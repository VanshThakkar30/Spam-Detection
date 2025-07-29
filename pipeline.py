import pandas as pd
from sklearn.ensemble import StackingClassifier

# --- Import necessary project functions ---
from src.data_processing.data_cleaner import clean_dataframe
from src.data_processing.feature_engineer import extract_features
from src.data_processing.class_balancer import stratified_split
from src.vectorization.tfidf_vectorizer import create_tfidf_vectorizer, save_vectorizer
from src.vectorization.word_embeddings import load_spacy_model, create_sentence_embeddings
from src.vectorization.feature_combiner import combine_features
from src.models.traditional_models import MODEL_CONFIG, save_model
from src.evaluation.metrics_calculator import generate_classification_report, plot_confusion_matrix
from src.evaluation.cross_validator import perform_cross_validation


def run_final_comparison():
    """
    A definitive pipeline to compare SVC and the Stacking Classifier using both
    k-fold cross-validation and a final test set evaluation.
    """
    # --- 1. Load and Process Data ---
    print("--- 1. Loading and Processing Data ---")
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df.columns = ['label', 'text', 'c3', 'c4', 'c5']
    df = df[['label', 'text']]
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    cleaned_df = clean_dataframe(df, 'text')
    featured_df = extract_features(cleaned_df, 'text')

    # --- 2. Split Data into Training and Testing Sets ---
    print("--- 2. Splitting Data into Training and Test Sets ---")
    X = featured_df.drop('label', axis=1)
    y = featured_df['label']
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.Series(y_train, name='label')
    y_test = pd.Series(y_test, name='label')
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}\n")

    # --- 3. Create Final Feature Sets ---
    print("--- 3. Preparing Feature Sets for Training and Testing ---")
    tfidf_vectorizer = create_tfidf_vectorizer(X_train['text'])
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
    print("Feature sets created.\n")

    # --- 4. K-Fold Cross-Validation on Training Data ---
    print("--- 4. K-Fold Cross-Validation Bake-Off ---")
    # Define the unfitted models for the test
    svc_model = MODEL_CONFIG['svm']['model']

    stacking_model = StackingClassifier(
        estimators=[
            ('svm', MODEL_CONFIG['svm']['model']),
            ('logistic_regression', MODEL_CONFIG['logistic_regression']['model'])
        ],
        final_estimator=MODEL_CONFIG['logistic_regression']['model'],
        cv=5
    )

    models_to_cv = {"SVC": svc_model, "Stacking Classifier": stacking_model}
    for name, model in models_to_cv.items():
        perform_cross_validation(model, X_train_final, y_train)

    # --- 5. Final Evaluation on Hold-Out Test Set ---
    print("\n--- 5. Final Evaluation on Hold-Out Test Set ---")
    for name, model in models_to_cv.items():
        print(f"\n--- Training and Evaluating {name} on final test set ---")
        # Train the model on the full training data
        model.fit(X_train_final, y_train)

        # Evaluate on the hold-out test set
        predictions = model.predict(X_test_final)
        generate_classification_report(y_test, predictions, name)
        plot_confusion_matrix(y_test, predictions, name)

    # --- 6. Save the Champion Model ---
    print("\n--- 6. Saving the Champion Model ---")
    # Based on the results, you can choose which model to save.
    # We'll default to the Stacking Classifier as it's typically more robust.
    champion_model = stacking_model
    save_model(champion_model, "champion_model.pkl")
    save_vectorizer(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    print("Final champion model and vectorizer have been saved.")

    print("\n--- Final Comparison Complete ---")


if __name__ == '__main__':
    run_final_comparison()