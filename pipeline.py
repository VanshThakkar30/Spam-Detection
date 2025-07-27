import pandas as pd
import numpy as np

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

def run_pipeline():
    """
    Executes the full pipeline from data loading to model training.
    """
    # --- Step 1: Data Loading and Initial Processing ---
    print("--- 1. Loading and Processing Data ---")
    try:
        # Load the dataset with latin-1 encoding, which is common for this file
        df = pd.read_csv('spam.csv', encoding='latin-1')
    except FileNotFoundError:
        print("Error: spam.csv not found. Make sure it's in the project root directory.")
        return

    # Keep only the necessary columns and rename them
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']

    # Convert labels to binary (0 for 'ham', 1 for 'spam')
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    print("Dataset loaded and columns renamed.")
    print(f"Initial data shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts(normalize=True)}\n")

    # --- Step 2: Cleaning and Feature Engineering ---
    print("--- 2. Cleaning Text and Engineering Features ---")
    cleaned_df = clean_dataframe(df, 'text')
    featured_df = extract_features(cleaned_df, 'text')
    print("Text cleaned and features engineered.")
    print(f"Data shape after cleaning and feature engineering: {featured_df.shape}\n")

    # --- Step 3: Data Splitting ---
    print("--- 3. Splitting Data into Training and Testing Sets ---")
    # Separate features from labels
    X = featured_df.drop('label', axis=1)
    y = featured_df['label']
    
    # Perform a stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Training label distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test label distribution:\n{y_test.value_counts(normalize=True)}\n")

    # --- Step 4: Vectorization ---
    print("--- 4. Vectorizing Text Data ---")
    # TF-IDF Vectorization
    # Fit the vectorizer ONLY on the training data to prevent data leakage
    tfidf_vectorizer = create_tfidf_vectorizer(X_train['text'])
    X_train_tfidf = tfidf_vectorizer.transform(X_train['text'])
    X_test_tfidf = tfidf_vectorizer.transform(X_test['text'])
    print("TF-IDF vectorization complete.")
    
    # Save the fitted vectorizer for later use in prediction
    save_vectorizer(tfidf_vectorizer, "tfidf_vectorizer.pkl")

    # Word Embedding Vectorization
    nlp = load_spacy_model()
    X_train_embed = create_sentence_embeddings(X_train['text'].tolist(), nlp)
    X_test_embed = create_sentence_embeddings(X_test['text'].tolist(), nlp)
    print("Sentence embeddings created.\n")

    # --- Step 5: Feature Combination ---
    print("--- 5. Combining All Features ---")
    # Isolate the engineered numerical features (all columns except 'text')
    engineered_cols = [col for col in X_train.columns if col != 'text']
    X_train_engineered = X_train[engineered_cols]
    X_test_engineered = X_test[engineered_cols]

    # Combine all features into a final feature matrix
    X_train_final = combine_features(X_train_tfidf, X_train_embed, X_train_engineered)
    X_test_final = combine_features(X_test_tfidf, X_test_embed, X_test_engineered)
    print("All feature types combined.")
    print(f"Final training feature matrix shape: {X_train_final.shape}")
    print(f"Final testing feature matrix shape: {X_test_final.shape}\n")

    # --- Step 6: Model Training ---
    print("--- 6. Training Traditional Models ---")
    trained_models = train_traditional_models(X_train_final, y_train)
    
    # Save all trained models
    for name, model in trained_models.items():
        save_model(model, f"{name}_model.pkl")
        
    print("\n--- Pipeline Complete ---")
    print("Next step: Evaluate the trained models on the test set.")


if __name__ == '__main__':
    run_pipeline()