import pandas as pd
import numpy as np
from collections import Counter

# --- 1. Data Processing ---
from src.data_processing.data_cleaner import clean_dataframe
from src.data_processing.feature_engineer import extract_features
from src.data_processing.class_balancer import stratified_split

# --- 2. Vectorization ---
from src.vectorization.tfidf_vectorizer import create_tfidf_vectorizer, save_vectorizer
from src.vectorization.word_embeddings import load_spacy_model, create_sentence_embeddings
from src.vectorization.feature_combiner import combine_features

# --- 3. Model Training ---
from src.models.traditional_models import train_traditional_models, save_model
from src.models.pytorch_models import SpamLSTM, train_pytorch_model

# --- 4. PyTorch Specific Imports ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- 10. Ensemble model
from src.models.ensemble_models import create_voting_classifier, create_stacking_classifier

# --- 11. Cross-Validation
from src.evaluation.cross_validator import perform_cross_validation


# --- PyTorch Data Preparation Helper ---
def prepare_pytorch_data(texts, labels, vocab_to_int, seq_length):
    """Tokenizes, encodes, and pads text data for PyTorch."""
    # Convert text to sequences of integers
    features = []
    for text in texts:
        features.append([vocab_to_int.get(word, 0) for word in text.split()])

    # Pad sequences
    padded_features = np.zeros((len(features), seq_length), dtype=int)
    for i, row in enumerate(features):
        n = len(row)
        if n > 0:
            padded_features[i, -n:] = np.array(row)[:seq_length]

    # Create tensors
    feature_tensor = torch.from_numpy(padded_features)
    label_tensor = torch.from_numpy(labels.to_numpy())

    # Create DataLoader
    dataset = TensorDataset(feature_tensor, label_tensor)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32, drop_last=True)

    return dataloader


def run_pipeline():
    """Executes the full pipeline for all model types."""
    # --- Steps 1-3: Data Loading, Cleaning, and Splitting ---
    print("--- 1. Loading and Processing Data ---")
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    print("--- 2. Cleaning Text and Engineering Features ---")
    cleaned_df = clean_dataframe(df, 'text')
    featured_df = extract_features(cleaned_df, 'text')

    print("--- 3. Splitting Data into Training, Validation, and Test Sets ---")
    X = featured_df.drop('label', axis=1)
    y = featured_df['label']
    X_train_val, X_test, y_train_val, y_test = stratified_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = stratified_split(X_train_val, y_train_val, test_size=0.2)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_val = pd.DataFrame(X_val, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.Series(y_train, name='label')
    y_val = pd.Series(y_val, name='label')
    y_test = pd.Series(y_test, name='label')

    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}\n")

    # --- Steps 4-5: Feature Preparation for Traditional Models ---
    print("--- 4. Vectorizing for Traditional Models ---")
    tfidf_vectorizer = create_tfidf_vectorizer(X_train['text'])
    X_train_tfidf = tfidf_vectorizer.transform(X_train['text'])
    save_vectorizer(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    nlp = load_spacy_model()
    X_train_embed = create_sentence_embeddings(X_train['text'].tolist(), nlp)

    print("--- 5. Combining Features for Traditional Models ---")
    engineered_cols = [col for col in X_train.columns if col != 'text']
    X_train_engineered = X_train[engineered_cols]
    X_train_final = combine_features(X_train_tfidf, X_train_embed, X_train_engineered)

    # --- Step 6: Training Traditional Models ---
    print("--- 6. Training Traditional Models (SVC & Logistic Regression) ---")
    trained_models = train_traditional_models(X_train_final, y_train)
    for name, model in trained_models.items():
        save_model(model, f"{name}_model.pkl")

    # --- Step 7-8: PyTorch Model Preparation and Training ---
    print("\n--- 7. Preparing Data for PyTorch Model ---")
    all_text = ' '.join(X_train['text'])
    words = all_text.split()
    count_words = Counter(words)
    sorted_words = count_words.most_common(len(count_words))
    vocab_to_int = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}

    train_loader = prepare_pytorch_data(X_train['text'], y_train, vocab_to_int, seq_length=50)
    validation_loader = prepare_pytorch_data(X_val['text'], y_val, vocab_to_int, seq_length=50)

    print("\n--- 8. Training PyTorch BiLSTM Model ---")
    VOCAB_SIZE = len(vocab_to_int) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = SpamLSTM(vocab_size=VOCAB_SIZE, embedding_dim=128, hidden_dim=256, output_dim=1, n_layers=2).to(device)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    train_pytorch_model(lstm_model, train_loader, validation_loader, optimizer, criterion, epochs=20, device=device,
                        patience=3)
    torch.save(lstm_model.state_dict(), 'lstm_model.pth')
    print("Best BiLSTM model saved.")

    # --- Step 9: Prepare Test Data for Final Evaluation ---
    print("\n--- 9. Preparing Test Set for Final Evaluation ---")
    X_test_tfidf = tfidf_vectorizer.transform(X_test['text'])
    X_test_embed = create_sentence_embeddings(X_test['text'].tolist(), nlp)
    X_test_engineered = X_test[engineered_cols]
    X_test_final = combine_features(X_test_tfidf, X_test_embed, X_test_engineered)
    print("Test data prepared for evaluation.")

    # --- Step 10: Building and Evaluating Ensemble Models ---
    print("\n--- 10. Building and Evaluating Ensemble Models ---")
    from src.models.traditional_models import MODEL_CONFIG
    from src.evaluation.metrics_calculator import generate_classification_report, plot_confusion_matrix

    # We need to re-import these here if they are not at the top
    from src.models.traditional_models import load_model

    lr_model = load_model("logistic_regression_model.pkl")
    svm_model = load_model("svm_model.pkl")

    base_estimators = [('logistic_regression', lr_model), ('svm', svm_model)]

    voting_model = create_voting_classifier(base_estimators)
    voting_model.fit(X_train_final, y_train)
    save_model(voting_model, "voting_classifier_model.pkl")

    voting_preds = voting_model.predict(X_test_final)
    generate_classification_report(y_test, voting_preds, "Voting Classifier")
    plot_confusion_matrix(y_test, voting_preds, "Voting Classifier")

    unfitted_estimators = [
        ('logistic_regression', MODEL_CONFIG['logistic_regression']['model']),
        ('svm', MODEL_CONFIG['svm']['model'])
    ]
    stacking_model = create_stacking_classifier(unfitted_estimators)
    stacking_model.fit(X_train_final, y_train)
    stacking_preds = stacking_model.predict(X_test_final)
    generate_classification_report(y_test, stacking_preds, "Stacking Classifier")
    plot_confusion_matrix(y_test, stacking_preds, "Stacking Classifier")

    print("\n--- 11. Robustness Check with Cross-Validation ---")

    # We will perform cross-validation on our best model (Voting Classifier)
    # using the full training set to get a stable performance estimate.
    # Note: We do not use the test set for cross-validation.
    perform_cross_validation(voting_model, X_train_final, y_train)

    print("\n--- Full Pipeline Complete ---")


if __name__ == '__main__':
    run_pipeline()