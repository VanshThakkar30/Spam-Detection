import pandas as pd

# Import all necessary functions from our project
from src.data_processing.data_cleaner import clean_dataframe
from src.data_processing.feature_engineer import extract_features
from src.data_processing.class_balancer import stratified_split
from src.vectorization.tfidf_vectorizer import load_vectorizer
from src.vectorization.word_embeddings import load_spacy_model, create_sentence_embeddings
from src.vectorization.feature_combiner import combine_features
from src.models.traditional_models import load_model, MODEL_CONFIG
from src.models.ensemble_models import create_voting_classifier
from src.evaluation.metrics_calculator import plot_confusion_matrix
from src.evaluation.cross_validator import perform_cross_validation


def main():
    """
    Loads trained models, evaluates them on the test set,
    and performs cross-validation on the training set.
    """
    print("--- Starting Model Evaluation ---")

    # --- 1. Recreate Data Splits ---
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    cleaned_df = clean_dataframe(df, 'text')
    featured_df = extract_features(cleaned_df, 'text')

    X = featured_df.drop('label', axis=1)
    y = featured_df['label']

    X_train_val, X_test, y_train_val, y_test = stratified_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = stratified_split(X_train_val, y_train_val, test_size=0.2)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.Series(y_train, name='label')
    y_test = pd.Series(y_test, name='label')
    print("Train and Test sets recreated successfully.")

    # --- 2. Prepare Feature Sets for Both Train and Test ---
    tfidf_vectorizer = load_vectorizer("tfidf_vectorizer.pkl")
    nlp = load_spacy_model()
    engineered_cols = [col for col in X_train.columns if col != 'text']

    # Prepare training features (for cross-validation)
    X_train_tfidf = tfidf_vectorizer.transform(X_train['text'])
    X_train_embed = create_sentence_embeddings(X_train['text'].tolist(), nlp)
    X_train_engineered = X_train[engineered_cols]
    X_train_final = combine_features(X_train_tfidf, X_train_embed, X_train_engineered)

    # Prepare testing features (for confusion matrix)
    X_test_tfidf = tfidf_vectorizer.transform(X_test['text'])
    X_test_embed = create_sentence_embeddings(X_test['text'].tolist(), nlp)
    X_test_engineered = X_test[engineered_cols]
    X_test_final = combine_features(X_test_tfidf, X_test_embed, X_test_engineered)
    print("Train and Test features prepared.")

    # --- 3. Plot Confusion Matrices on Test Data ---
    print("\n--- Confusion Matrix on Test Set ---")
    models_to_plot = {
        "SVC": "svm_model.pkl",
        "Logistic Regression": "logistic_regression_model.pkl",
        "Voting Classifier": "voting_classifier_model.pkl"
    }
    for model_name, path in models_to_plot.items():
        model = load_model(path)
        predictions = model.predict(X_test_final)
        plot_confusion_matrix(y_test, predictions, model_name)

    # --- 4. Perform Cross-Validation on Training Data ---
    print("\n--- Cross-Validation Performance on Training Set ---")
    # For cross-validation, we use unfitted models
    unfitted_svc = MODEL_CONFIG['svm']['model']
    unfitted_lr = MODEL_CONFIG['logistic_regression']['model']
    unfitted_voting = create_voting_classifier([
        ('logistic_regression', unfitted_lr),
        ('svm', unfitted_svc)
    ])

    models_to_cv = {
        "SVC": unfitted_svc,
        "Logistic Regression": unfitted_lr,
        "Voting Classifier": unfitted_voting
    }

    for model_name, model in models_to_cv.items():
        perform_cross_validation(model, X_train_final, y_train)


if __name__ == '__main__':
    main()