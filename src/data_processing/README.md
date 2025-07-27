# Data Processing Module - README

This directory contains core modules for data preprocessing and feature engineering in the Spam Detection project. Each file is designed to be modular and reusable, making it easy for anyone to integrate or extend the functionality.

## 1. data_cleaner.py
**Purpose:**
- Cleans raw text data by removing unwanted elements and normalizing the text.

**Key Features:**
- Removes URLs, email addresses, and phone numbers from text.
- Handles special characters and emojis.
- Normalizes text (converts to lowercase, trims whitespace).
- Removes duplicate and missing entries.

**How to Use:**
- Use the `clean_dataframe(df, text_column)` function to clean a pandas DataFrame column containing text messages.
- Example usage is provided in the `__main__` block at the bottom of the file.

---

## 2. feature_engineer.py
**Purpose:**
- Extracts useful features from text data to improve model performance.

**Key Features:**
- Counts capital letters, exclamation marks, and currency symbols.
- Calculates word count, character count, and sentence count.
- Computes special character density.
- Flags presence of URLs and email addresses.

**How to Use:**
- Use the `extract_features(df, text_column)` function to add feature columns to your DataFrame.
- Example usage is provided in the `__main__` block at the bottom of the file.

---

## 3. class_balancer.py
**Purpose:**
- Handles class imbalance in the dataset, which is common in spam detection tasks.

**Key Features:**
- Computes class weights for imbalanced datasets.
- Applies SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.
- Provides stratified train-test splitting.
- Supports undersampling and oversampling techniques.

**How to Use:**
- Use `compute_weights(y)` to get class weights for your labels.
- Use `apply_smote(X, y)` to balance your feature set and labels.
- Use `stratified_split(df, label_col)` for a balanced train-test split.
- Use `undersample(df, label_col)` or `oversample(df, label_col)` for sampling.
- Example usage is provided in the `__main__` block at the bottom of the file.

---

**Tip:**
Each file is self-contained and includes a test/example block at the bottom. You can run any file directly to see how it works with sample data. 