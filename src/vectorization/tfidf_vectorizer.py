import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Any
import pickle

# Custom stop words tailored for spam detection
# These are common words that might not be in standard stop word lists but are frequent in spam
CUSTOM_STOP_WORDS = [
    'click', 'free', 'win', 'winner', 'prize', 'claim', 'urgent', 'buy', 'now', 'subscribe'
]

def create_tfidf_vectorizer(
    documents: List[str],
    max_features: int = 5000,
    ngram_range: tuple = (1, 2)
) -> TfidfVectorizer:
    """
    Creates and fits a TF-IDF vectorizer.

    Args:
        documents (List[str]): A list of text documents to fit the vectorizer on.
        max_features (int): The maximum number of features (top words) to keep.
        ngram_range (tuple): The range of n-grams to consider (e.g., (1, 2) for unigrams and bigrams).

    Returns:
        TfidfVectorizer: A scikit-learn TfidfVectorizer object fitted to the documents.
    """
    # Combine standard English stop words with our custom list
    stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words()) + CUSTOM_STOP_WORDS

    # Initialize the vectorizer with specified parameters
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
        lowercase=True,
        # min_df=2 # Alternative: ignore terms that appear in less than 2 documents
    )

    # Fit the vectorizer to the provided text documents
    vectorizer.fit(documents)
    return vectorizer

def save_vectorizer(vectorizer: TfidfVectorizer, filepath: str):
    """Saves a trained vectorizer to a file using pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_vectorizer(filepath: str) -> TfidfVectorizer:
    """Loads a vectorizer from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # --- Example Usage ---
    # Sample documents for demonstration
    sample_docs = [
        "Congratulations! You win a free prize. Click here to claim.",
        "URGENT: Your account needs immediate attention. Please click the link.",
        "Hello, this is a friendly reminder about our meeting tomorrow.",
        "Buy now and get a special discount on all products.",
        "This is a legitimate email, not a scam."
    ]
    df = pd.DataFrame({'text': sample_docs})

    print("--- Creating and Fitting TF-IDF Vectorizer ---")
    # Create the vectorizer based on our sample documents
    tfidf_vectorizer = create_tfidf_vectorizer(df['text'])
    print(f"Vectorizer created with {len(tfidf_vectorizer.get_feature_names_out())} features.")
    print("Feature names:", tfidf_vectorizer.get_feature_names_out()[:20]) # Display first 20 features

    # Transform the documents into a TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.transform(df['text'])

    print("\n--- TF-IDF Matrix ---")
    print("Shape of matrix:", tfidf_matrix.shape)
    print("Transformed matrix (first 2 rows):")
    print(tfidf_matrix.toarray()[:2])

    # --- Saving and Loading ---
    filepath = "tfidf_vectorizer.pkl"
    save_vectorizer(tfidf_vectorizer, filepath)
    print(f"\nVectorizer saved to {filepath}")

    loaded_vectorizer = load_vectorizer(filepath)
    print("Vectorizer loaded successfully.")

    # Verify that the loaded vectorizer works
    retransformed_matrix = loaded_vectorizer.transform(df['text'])
    print("Retransformed matrix shape is the same:", retransformed_matrix.shape == tfidf_matrix.shape)