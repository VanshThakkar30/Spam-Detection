# src/data_processing/class_balancer.py
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Dict, Union

def compute_weights(y: pd.Series) -> Dict[int, float]:
    """
    Computes class weights for imbalanced datasets.
    """
    if not isinstance(y, pd.Series):
        raise TypeError("Input 'y' must be a pandas Series.")
        
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

def apply_smote(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies SMOTE to balance classes. It automatically adjusts k_neighbors 
    for small minority classes to prevent errors.
    
    Note: SMOTE requires all input features in X to be numerical.
    """
    if not all(np.issubdtype(dtype, np.number) for dtype in X.dtypes):
        raise TypeError("All columns in X must be numeric to apply SMOTE.")

    minority_class_count = y.value_counts().min()
    # k_neighbors must be less than the number of samples in the minority class
    k_neighbors = max(1, minority_class_count - 1)
    
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)

def stratified_split(X, y, test_size: float = 0.2):
    """
    Performs a stratified train-test split on features (X) and labels (y).

    This is a wrapper for scikit-learn's train_test_split to ensure
    stratification on the label.
    """
    X_train, X_test, y_train, y_test = sk_train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,  # Stratify based on the labels
        random_state=42
    )
    return X_train, X_test, y_train, y_test

def undersample(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Undersamples the majority class to match the minority class size.
    """
    min_count = df[label_col].value_counts().min()
    # group_keys=False is important to avoid a multilevel index
    return df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(n=min_count, random_state=42)
    ).reset_index(drop=True)

def oversample(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Oversamples the minority class to match the majority class size.
    """
    max_count = df[label_col].value_counts().max()
    return df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(n=max_count, replace=True, random_state=42)
    ).reset_index(drop=True)

if __name__ == "__main__":
    # Create a more realistic imbalanced dataset
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'label': [1] * 10 + [0] * 90  # 10% spam, 90% ham
    }
    df = pd.DataFrame(data)

    print("--- Class Balancing Techniques ---")
    print(f"Original class distribution:\n{df['label'].value_counts()}\n")

    # 1. Class Weights
    weights = compute_weights(df['label'])
    print(f"Computed Class Weights: {weights}\n")

    # 2. Stratified Splitting
    train_df, test_df = stratified_split(df, 'label', test_size=0.2)
    print(f"Train set distribution:\n{train_df['label'].value_counts()}")
    print(f"Test set distribution:\n{test_df['label'].value_counts()}\n")
    
    # 3. Undersampling
    under_df = undersample(df, 'label')
    print(f"Undersampled distribution:\n{under_df['label'].value_counts()}\n")

    # 4. Oversampling
    over_df = oversample(df, 'label')
    print(f"Oversampled distribution:\n{over_df['label'].value_counts()}\n")

    # 5. SMOTE
    # SMOTE is typically applied only to the training set after splitting
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    
    print(f"Original training distribution: {Counter(y_train)}")
    X_res, y_res = apply_smote(X_train, y_train)
    print(f"SMOTE-resampled training distribution: {Counter(y_res)}\n")