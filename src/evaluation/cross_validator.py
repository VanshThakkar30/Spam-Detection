from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np


def perform_cross_validation(model, X, y, cv_folds=5, scoring='f1_macro'):
    """
    Performs stratified cross-validation on a model.

    Args:
        model: The machine learning model to evaluate.
        X: The feature set.
        y: The labels.
        cv_folds (int): The number of cross-validation folds.
        scoring (str): The scoring metric to use.

    Returns:
        The mean and standard deviation of the cross-validation scores.
    """
    print(f"\n--- Performing {cv_folds}-Fold Cross-Validation for {model.__class__.__name__} ---")

    # StratifiedKFold is essential for imbalanced datasets like ours
    # It ensures each fold has the same class distribution as the whole dataset
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Calculate the scores for each fold
    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)

    mean_score = np.mean(scores)
    std_dev = np.std(scores)

    print(f"Scores for each fold: {scores}")
    print(f"Average F1-Score: {mean_score:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")

    return mean_score, std_dev