# src/data_processing/data_cleaner.py
import re
import pandas as pd
from typing import Optional

# Compiled regex patterns for efficiency
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN = re.compile(r'\b[\w.-]+@[\w.-]+\.\w{2,4}\b')
PHONE_PATTERN = re.compile(r'\b\+?\d[\d\s().-]{7,}\b')
EMOJI_PATTERN = re.compile(r'[\U0001F600-\U0001F64F\u2600-\u26FF\u2700-\u27BF]')
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s.,!?$€£₹]')
WHITESPACE_PATTERN = re.compile(r'\s+')

def clean_text(text: Optional[str]) -> str:
    """
    Cleans a single text string by removing URLs, emails, phone numbers, emojis, 
    and special characters, followed by normalization.
    """
    if not isinstance(text, str):
        return ''
    
    text = URL_PATTERN.sub('', text)
    text = EMAIL_PATTERN.sub('', text)
    text = PHONE_PATTERN.sub('', text)
    text = EMOJI_PATTERN.sub('', text)
    text = SPECIAL_CHARS_PATTERN.sub('', text)
    text = text.lower()
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    return text


def clean_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Cleans a DataFrame's text column, removes duplicates, and handles missing values.
    """
    df_copy = df.copy()
    # Ensure the text column is of type string, converting everything else to empty strings
    df_copy[text_column] = df_copy[text_column].astype(str).fillna('')
    
    df_copy[text_column] = df_copy[text_column].apply(clean_text)
    
    # Drop rows where the text column is now empty after cleaning
    df_copy = df_copy[df_copy[text_column] != '']

    # Handle duplicates and missing values
    df_copy = df_copy.drop_duplicates(subset=[text_column])
    df_copy = df_copy.dropna(subset=[text_column]) # Redundant but safe
    
    return df_copy.reset_index(drop=True)


if __name__ == "__main__":
    # Example usage
    data = {
        'text': [
            'Hello! Visit https://example.com now! 😊',
            'Contact me at test@email.com or +1-234-567-8901!',
            'WIN $1000!!!',
            None,
            12345, # Example of non-string data
            'Hello! Visit https://example.com now! 😊',  # duplicate
        ],
        'other_col': [1, 2, 3, 4, 5, 6]
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df, 'text')
    print("\nCleaned DataFrame:")
    print(cleaned_df)