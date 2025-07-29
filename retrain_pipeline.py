import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import StackingClassifier
import time

# Import functions from our project
from src.data_processing.data_cleaner import clean_dataframe
from src.data_processing.feature_engineer import extract_features
from src.vectorization.tfidf_vectorizer import create_tfidf_vectorizer, save_vectorizer, load_vectorizer
from src.vectorization.word_embeddings import load_spacy_model, create_sentence_embeddings
from src.vectorization.feature_combiner import combine_features
from src.models.traditional_models import load_model, save_model, MODEL_CONFIG
from src.evaluation.cross_validator import perform_cross_validation


def run_kfold_retraining():
    """
    A robust retraining pipeline that uses k-fold cross-validation
    for the champion-challenger evaluation.
    """
    print("--- Starting K-Fold Retraining Pipeline for Stacking Classifier ---")

    # --- 1. Load All Available Data ---
    # Load new data from feedback
    engine = create_engine('sqlite:///spam_app.db')
    query = "SELECT pl.message, pl.prediction FROM prediction_log pl JOIN user_feedback uf ON pl.id = uf.log_id WHERE uf.is_correct = 0;"
    feedback_df = pd.read_sql(query, engine)

    if feedback_df.empty:
        print("No new incorrect feedback data to retrain on. Exiting.")
        return

    feedback_df['label'] = feedback_df['prediction'].apply(lambda x: 0 if x == 'spam' else 1)
    feedback_df = feedback_df[['message', 'label']].rename(columns={'message': 'text'})

    # Load original data
    original_df = pd.read_csv('spam.csv', encoding='latin-1')
    original_df.columns = ['label', 'text', 'c3', 'c4', 'c5']
    original_df = original_df[['label', 'text']]
    original_df['label'] = original_df['label'].map({'ham': 0, 'spam': 1})

    # Create the complete, up-to-date dataset
    combined_df = pd.concat([original_df, feedback_df], ignore_index=True).drop_duplicates()

    # --- 2. Prepare the Full Feature Set ---
    cleaned_df = clean_dataframe(combined_df, 'text')
    featured_df = extract_features(cleaned_df, 'text')

    X = featured_df.drop('label', axis=1)
    y = featured_df['label']

    # Vectorize the entire dataset
    tfidf_vectorizer = create_tfidf_vectorizer(X['text'])
    nlp = load_spacy_model()
    engineered_cols = [col for col in X.columns if col != 'text']

    X_final = combine_features(
        tfidf_vectorizer.transform(X['text']),
        create_sentence_embeddings(X['text'].tolist(), nlp),
        X[engineered_cols]
    )

    # --- 3. Evaluate Champion and Challenger with K-Fold ---
    print("\n--- Evaluating Models with K-Fold Cross-Validation ---")

    # Load the current production model (the "Champion")
    try:
        champion_model = load_model("stacking_model.pkl")
        champion_score, _ = perform_cross_validation(champion_model, X_final, y)
    except FileNotFoundError:
        champion_score = 0
        print("No champion model found. The new challenger will be promoted automatically.")

    # Define the "Challenger" model (unfitted)
    unfitted_estimators = [
        ('logistic_regression', MODEL_CONFIG['logistic_regression']['model']),
        ('svm', MODEL_CONFIG['svm']['model'])
    ]
    challenger_model = StackingClassifier(
        estimators=unfitted_estimators,
        final_estimator=MODEL_CONFIG['logistic_regression']['model'],
        cv=5
    )
    challenger_score, _ = perform_cross_validation(challenger_model, X_final, y)

    # --- 4. Promote Challenger if Better ---
    if challenger_score > champion_score:
        print("\nChallenger model is better! Promoting to production.")
        print("Retraining challenger on the full dataset before saving...")

        # Retrain the final model on ALL available data
        challenger_model.fit(X_final, y)

        # Overwrite the old production models
        save_model(challenger_model, "stacking_model.pkl")
        save_vectorizer(tfidf_vectorizer, "tfidf_vectorizer.pkl")
        print(f"New champion model saved to: stacking_model.pkl")
    else:
        print("\nChallenger model is not better. Keeping the current champion model.")


if __name__ == '__main__':
    run_kfold_retraining()