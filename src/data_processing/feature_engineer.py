# src/data_processing/feature_engineer.py
import pandas as pd
import numpy as np

# Use a compiled regex for currency symbols for better performance
CURRENCY_SYMBOLS = r'[$€£₹]'

def extract_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Engineers a set of features from the text column of a DataFrame.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the DataFrame.")

    df_copy = df.copy()
    
    # Ensure the text column is of type string
    df_copy[text_column] = df_copy[text_column].astype(str)

    df_copy['char_count'] = df_copy[text_column].str.len()
    df_copy['word_count'] = df_copy[text_column].str.split().str.len().fillna(0).astype(int)
    
    df_copy['capital_letters'] = df_copy[text_column].str.count(r'[A-Z]')
    df_copy['exclamation_marks'] = df_copy[text_column].str.count(r'!')
    df_copy['currency_symbols'] = df_copy[text_column].str.count(CURRENCY_SYMBOLS)
    df_copy['sentence_count'] = df_copy[text_column].str.count(r'[.!?]') + 1
    
    # Safely calculate density to avoid division by zero
    special_chars = df_copy[text_column].str.count(r'[^\w\s]')
    df_copy['special_char_density'] = special_chars / (df_copy['char_count'] + 1e-9)

    # Word length statistics
    words = df_copy[text_column].str.split()
    df_copy['avg_word_length'] = words.apply(lambda x: np.mean([len(i) for i in x]) if x else 0)
    
    # Flags
    df_copy['url_flag'] = df_copy[text_column].str.contains(r'https?://|www\.', case=False, na=False).astype(int)
    df_copy['email_flag'] = df_copy[text_column].str.contains(r'\b[\w.-]+@[\w.-]+\.\w{2,4}\b', case=False, na=False).astype(int)
    
    return df_copy


if __name__ == "__main__":
    # Example usage
    data = {
        'text': [
            'WIN $1000!!! CLICK NOW.',
            'Contact me at test@email.com.',
            'Hello! Visit https://example.com now! A great offer awaits.',
            'This is a normal, friendly message.',
            '' # Empty string case
        ]
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    features_df = extract_features(df, 'text')
    print("\nDataFrame with Features:")
    print(features_df)