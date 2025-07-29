import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from collections import Counter
from torch.utils.data import DataLoader

# Import all necessary functions and classes from our project
from src.data_processing.data_cleaner import clean_dataframe
from src.data_processing.feature_engineer import extract_features
from src.vectorization.tfidf_vectorizer import create_tfidf_vectorizer
from src.vectorization.word_embeddings import load_spacy_model, create_sentence_embeddings
from src.vectorization.feature_combiner import combine_features
from src.models.traditional_models import MODEL_CONFIG
from src.models.pytorch_models import SpamLSTM, train_pytorch_model
from pipeline import prepare_pytorch_data  # Re-use the helper from pipeline.py


def get_svc_predictions(model, texts, tfidf_vectorizer, nlp, engineered_cols):
    """Generates prediction probabilities from the SVC model for a set of texts."""
    # This function encapsulates the full feature pipeline for the SVC
    X_tfidf = tfidf_vectorizer.transform(texts)
    X_embed = create_sentence_embeddings(texts.tolist(), nlp)

    # Create a dummy DataFrame to run feature engineering
    df = pd.DataFrame({'text': texts})
    featured_df = extract_features(df, 'text')
    X_engineered = featured_df[engineered_cols]

    X_final = combine_features(X_tfidf, X_embed, X_engineered)
    # Return the probability of the 'spam' class
    return model.predict_proba(X_final)[:, 1]


# In fold_test.py

# ... (imports and get_svc_predictions function remain the same) ...

def get_lstm_predictions(model, texts, vocab_to_int, device):
    """Generates prediction probabilities from the BiLSTM model."""
    model.eval()

    # Create a dataloader specifically for inference (no shuffle, no drop_last)
    dummy_labels = pd.Series(np.zeros(len(texts)))
    dataset = prepare_pytorch_data(texts, dummy_labels, vocab_to_int, seq_length=50).dataset

    prediction_loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=32  # You can adjust the batch size
    )

    all_preds = []
    with torch.no_grad():
        for inputs, _ in prediction_loader:
            inputs = inputs.to(device)
            h = model.init_hidden(inputs.size(0), device)
            output, _ = model(inputs, h)
            all_preds.extend(output.cpu().numpy())

    return np.array(all_preds)


def main():
    """
    Performs k-fold cross-validation on a custom stacking ensemble
    of an SVC and a BiLSTM model.
    """
    print("--- Starting K-Fold Test for SVC + BiLSTM Stacking Ensemble ---")

    # --- 1. Load and Prepare Full Dataset ---
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df.columns = ['label', 'text', 'c3', 'c4', 'c5']
    df = df[['label', 'text']]
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    cleaned_df = clean_dataframe(df, 'text')

    X = cleaned_df['text']
    y = cleaned_df['label']

    # --- 2. Set up K-Fold Cross-Validation ---
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []

    nlp = load_spacy_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 3. Loop Through Each Fold ---
    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        print(f"\n--- Processing Fold {fold + 1}/{n_splits} ---")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # --- Train Base Model 1: SVC ---
        print("Training SVC...")
        svc_model = MODEL_CONFIG['svm']['model']
        tfidf_vectorizer = create_tfidf_vectorizer(X_train)
        engineered_cols = extract_features(pd.DataFrame({'text': X_train}), 'text').drop(['text', 'label'], axis=1,
                                                                                         errors='ignore').columns

        X_train_svc_features = combine_features(
            tfidf_vectorizer.transform(X_train),
            create_sentence_embeddings(X_train.tolist(), nlp),
            extract_features(pd.DataFrame({'text': X_train}), 'text')[engineered_cols]
        )
        svc_model.fit(X_train_svc_features, y_train)

        # --- Train Base Model 2: BiLSTM ---
        print("Training BiLSTM...")
        words = ' '.join(X_train).split()
        count_words = Counter(words)
        sorted_words = count_words.most_common(len(count_words))
        vocab_to_int = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}

        train_loader = prepare_pytorch_data(X_train, y_train, vocab_to_int, seq_length=50)

        lstm_model = SpamLSTM(vocab_size=len(vocab_to_int) + 1, embedding_dim=128, hidden_dim=256, output_dim=1,
                              n_layers=2).to(device)
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        # Using train_loader as validation loader for simplicity in this script
        train_pytorch_model(lstm_model, train_loader, train_loader, optimizer, criterion, epochs=5, device=device,
                            patience=2)

        # --- Train the Meta-Learner ---
        print("Training Meta-Learner...")
        svc_train_preds = get_svc_predictions(svc_model, X_train, tfidf_vectorizer, nlp, engineered_cols)
        lstm_train_preds = get_lstm_predictions(lstm_model, X_train, vocab_to_int, device)

        meta_features_train = np.column_stack((svc_train_preds, lstm_train_preds))

        meta_learner = LogisticRegression()
        meta_learner.fit(meta_features_train, y_train)

        # --- Evaluate the Ensemble on the Test Fold ---
        print("Evaluating ensemble on the test fold...")
        svc_test_preds = get_svc_predictions(svc_model, X_test, tfidf_vectorizer, nlp, engineered_cols)
        lstm_test_preds = get_lstm_predictions(lstm_model, X_test, vocab_to_int, device)

        meta_features_test = np.column_stack((svc_test_preds, lstm_test_preds))

        final_predictions = meta_learner.predict(meta_features_test)

        score = f1_score(y_test, final_predictions)
        fold_scores.append(score)
        print(f"F1-Score for Fold {fold + 1}: {score:.4f}")

    # --- 4. Report Final Results ---
    print("\n--- Final Cross-Validation Results ---")
    print(f"Scores for each fold: {fold_scores}")
    print(f"Average F1-Score: {np.mean(fold_scores):.4f}")
    print(f"Standard Deviation: {np.std(fold_scores):.4f}")


if __name__ == '__main__':
    main()