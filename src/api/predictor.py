import pickle
import pandas as pd

# Import all necessary functions from our project
from src.data_processing.data_cleaner import clean_text
from src.data_processing.feature_engineer import extract_features
from src.vectorization.word_embeddings import load_spacy_model, create_sentence_embeddings
from src.vectorization.feature_combiner import combine_features

# --- Load all necessary artifacts once when the server starts ---
try:
    # Load the best model (SVC)
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load the TF-IDF vectorizer
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    # Load the spaCy model for embeddings
    nlp = load_spacy_model()

except FileNotFoundError:
    model, tfidf_vectorizer, nlp = None, None, None
    print("Warning: One or more model/vectorizer artifacts not found. Prediction will fail.")

def predict_message(message: str) -> dict:
    """
    Takes a raw text message and returns a spam/ham prediction.
    """
    if not all([model, tfidf_vectorizer, nlp]):
        raise RuntimeError("Model artifacts are not loaded. Cannot make predictions.")

    # 1. Clean the raw text
    cleaned_message = clean_text(message)

    # 2. Create a DataFrame for feature engineering
    data = {'text': [cleaned_message]}
    df = pd.DataFrame(data)

    # 3. Engineer features
    featured_df = extract_features(df, 'text')
    engineered_features = featured_df.drop('text', axis=1)

    # 4. Vectorize text (TF-IDF and Embeddings)
    tfidf_features = tfidf_vectorizer.transform([cleaned_message])
    embedding_features = create_sentence_embeddings([cleaned_message], nlp)

    # 5. Combine all features into the final format
    final_features = combine_features(
        tfidf_features,
        embedding_features,
        engineered_features
    )

    # 6. Make prediction
    prediction = model.predict(final_features)[0]
    probability = model.predict_proba(final_features)[0].tolist()

    result = {
        "prediction": "spam" if prediction == 1 else "ham",
        "confidence": {
            "ham": round(probability[0], 4),
            "spam": round(probability[1], 4)
        }
    }
    return result