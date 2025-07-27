import spacy
import numpy as np
import pandas as pd
from typing import List

# We use a small, efficient pre-trained model from spaCy.
# For higher accuracy, a larger model like 'en_core_web_lg' could be used.
NLP_MODEL_NAME = "en_core_web_sm"

def load_spacy_model(model_name: str = NLP_MODEL_NAME):
    """
    Loads a spaCy model, downloading it if it's not already installed.
    """
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Spacy model '{model_name}' not found. Downloading...")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    return nlp

def create_sentence_embeddings(documents: List[str], nlp) -> np.ndarray:
    """
    Creates sentence-level embeddings by averaging the word vectors of tokens in a document.

    Args:
        documents (List[str]): A list of text documents.
        nlp: A loaded spaCy NLP model object.

    Returns:
        np.ndarray: A 2D numpy array where each row is a sentence embedding.
    """
    embeddings = []
    for doc_text in documents:
        # Process the document with spaCy
        doc = nlp(doc_text)
        # Get the average vector of all tokens that have a vector
        # Using a zero vector for empty or out-of-vocabulary documents
        if doc.has_vector and np.any(doc.vector):
            embeddings.append(doc.vector)
        else:
            # Fallback to a zero vector of the correct dimension if doc has no vector
            embeddings.append(np.zeros(nlp.vocab.vectors_length))
            
    return np.array(embeddings)


if __name__ == "__main__":
    # --- Example Usage ---
    # Load the spaCy model first
    print("--- Loading spaCy Model ---")
    nlp_model = load_spacy_model()
    print(f"Model '{NLP_MODEL_NAME}' loaded successfully.")
    print(f"Vector dimension: {nlp_model.vocab.vectors_length}")


    # Sample documents for demonstration
    sample_docs = [
        "Congratulations! You win a free prize. Click here to claim.",
        "URGENT: Your account needs immediate attention.",
        "Hello, this is a friendly reminder about our meeting tomorrow.",
        "" # Test empty string
    ]
    df = pd.DataFrame({'text': sample_docs})

    print("\n--- Creating Sentence Embeddings ---")
    # Generate embeddings for the sample documents
    sentence_vectors = create_sentence_embeddings(df['text'], nlp_model)
    
    print("Shape of embeddings matrix:", sentence_vectors.shape)
    print("First sentence embedding vector (first 10 values):")
    print(sentence_vectors[0, :10])