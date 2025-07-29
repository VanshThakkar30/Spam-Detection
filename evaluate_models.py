import pandas as pd
from sklearn.ensemble import StackingClassifier

# Import all necessary functions from our project
from src.data_processing.data_cleaner import clean_dataframe
from src.data_processing.feature_engineer import extract_features
from src.data_processing.class_balancer import stratified_split
from src.vectorization.tfidf_vectorizer import load_vectorizer
from src.vectorization.word_embeddings import load_spacy_model, create_sentence_embeddings
from src.vectorization.feature_combiner import combine_features
from src.models.traditional_models import load_model, MODEL_CONFIG
from src.evaluation.metrics_calculator import generate_classification_report, plot_confusion_matrix


def main():
    """
    Loads trained models, evaluates them on the test set,
    and compares them against a Stacking Classifier.
    """
    print("--- Starting Final Model Comparison ---")

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

    # --- 2. Prepare Feature Sets ---
    tfidf_vectorizer = load_vectorizer("tfidf_vectorizer.pkl")
    nlp = load_spacy_model()
    engineered_cols = [col for col in X_train.columns if col != 'text']

    X_train_final = combine_features(
        tfidf_vectorizer.transform(X_train['text']),
        create_sentence_embeddings(X_train['text'].tolist(), nlp),
        X_train[engineered_cols]
    )
    X_test_final = combine_features(
        tfidf_vectorizer.transform(X_test['text']),
        create_sentence_embeddings(X_test['text'].tolist(), nlp),
        X_test[engineered_cols]
    )
    print("Train and Test features prepared.")

    # --- 3. Evaluate Individual Models on Test Data ---
    print("\n--- Evaluating Individual Models on Test Set ---")
    models_to_evaluate = {
        "SVC": "svm_model.pkl",
        "Logistic Regression": "logistic_regression_model.pkl",
    }
    for model_name, path in models_to_evaluate.items():
        model = load_model(path)
        predictions = model.predict(X_test_final)
        generate_classification_report(y_test, predictions, model_name)
        plot_confusion_matrix(y_test, predictions, model_name)

    # --- 4. Train and Evaluate Stacking Classifier ---
    print("\n--- Training and Evaluating Stacking Classifier ---")
    # Define the base models (unfitted)
    estimators = [
        ('svm', MODEL_CONFIG['svm']['model']),
        ('logistic_regression', MODEL_CONFIG['logistic_regression']['model'])
    ]

    # The Stacking Classifier uses a final model to combine the predictions
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=MODEL_CONFIG['logistic_regression']['model'],  # Use LR as the meta-learner
        cv=5
    )

    # Train the stacking model on the full training data
    stacking_model.fit(X_train_final, y_train)

    # Evaluate on the test data
    stacking_preds = stacking_model.predict(X_test_final)
    generate_classification_report(y_test, stacking_preds, "Stacking Classifier")
    plot_confusion_matrix(y_test, stacking_preds, "Stacking Classifier")


if __name__ == '__main__':
    main()