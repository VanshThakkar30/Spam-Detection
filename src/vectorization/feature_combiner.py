import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def combine_features(
    tfidf_matrix: csr_matrix,
    embedding_matrix: np.ndarray,
    engineered_features: pd.DataFrame,
    scale_engineered: bool = True
) -> csr_matrix:
    """
    Combines sparse TF-IDF features, dense embedding features, and engineered numerical features.

    Args:
        tfidf_matrix (csr_matrix): A sparse matrix from a TfidfVectorizer.
        embedding_matrix (np.ndarray): A dense numpy array of sentence embeddings.
        engineered_features (pd.DataFrame): A DataFrame of engineered numerical features.
        scale_engineered (bool): If True, applies StandardScaler to the engineered features.

    Returns:
        csr_matrix: A single sparse matrix containing all combined features.
    """
    if scale_engineered:
        # Initialize a scaler
        scaler = MinMaxScaler()
        # Fit and transform the engineered features
        engineered_scaled = scaler.fit_transform(engineered_features)
    else:
        engineered_scaled = engineered_features.values

    # Convert dense matrices (embeddings and engineered) to sparse format
    embedding_sparse = csr_matrix(embedding_matrix)
    engineered_sparse = csr_matrix(engineered_scaled)
    
    # Combine all feature matrices horizontally
    combined_matrix = hstack([tfidf_matrix, embedding_sparse, engineered_sparse])
    
    return combined_matrix.tocsr() # Ensure CSR format for efficiency


if __name__ == "__main__":
    from .tfidf_vectorizer import create_tfidf_vectorizer
    from .word_embeddings import load_spacy_model, create_sentence_embeddings
    
    # --- Example Usage ---
    # 1. Generate sample data and features
    sample_docs = [
        "WIN $1000!!! CLICK NOW.",
        "Contact me at test@email.com.",
        "Hello! Visit https://example.com now! A great offer awaits.",
        "This is a normal, friendly message.",
    ]
    
    # Create TF-IDF features
    tfidf_vectorizer = create_tfidf_vectorizer(sample_docs, max_features=20)
    sample_tfidf = tfidf_vectorizer.transform(sample_docs)
    
    # Create word embedding features
    nlp = load_spacy_model()
    sample_embeddings = create_sentence_embeddings(sample_docs, nlp)

    # Create engineered features (mock-up)
    sample_engineered = pd.DataFrame({
        'char_count': [23, 28, 62, 33],
        'word_count': [5, 5, 11, 6],
        'currency_symbols': [1, 0, 0, 0]
    })
    
    print("--- Feature Shapes Before Combination ---")
    print(f"TF-IDF matrix shape: {sample_tfidf.shape}")
    print(f"Embeddings matrix shape: {sample_embeddings.shape}")
    print(f"Engineered features shape: {sample_engineered.shape}")

    # 2. Combine all features
    final_features = combine_features(sample_tfidf, sample_embeddings, sample_engineered)
    
    print("\n--- Combined Features ---")
    print(f"Final combined matrix shape: {final_features.shape}")
    print("Combined matrix (first row, first 30 columns):")
    print(final_features.toarray()[0, :30])

    # Verification of shape
    expected_cols = sample_tfidf.shape[1] + sample_embeddings.shape[1] + sample_engineered.shape[1]
    print(f"\nShape verification: {final_features.shape[1] == expected_cols}")