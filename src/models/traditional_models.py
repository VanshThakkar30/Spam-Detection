import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any

# Define the models and their hyperparameter grids for tuning
MODEL_CONFIG = {
    "random_forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None]
        }
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        }
    },
    "svm": {                                       # <-- SVC is back in
        "model": SVC(probability=True, random_state=42),
        "params": {
            "C": [1, 10],
            "kernel": ['rbf', 'linear']
        }
    },
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [1, 10],
            "solver": ['liblinear']
        }
    }
}


def tune_hyperparameters(model, params: Dict[str, Any], X_train, y_train) -> Any:
    """
    Performs hyperparameter tuning using GridSearchCV.

    Args:
        model: The machine learning model instance.
        params (Dict[str, Any]): The dictionary of parameters to search.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        The best model found by GridSearchCV after fitting.
    """
    print(f"Tuning hyperparameters for {model.__class__.__name__}...")
    
    # Initialize GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=5,
        scoring='f1_macro', # F1-score is a good metric for potentially imbalanced classes
        n_jobs=-1, # Use all available CPU cores
        verbose=1
    )
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best F1-score: {grid_search.best_score_:.4f}\n")
    
    return grid_search.best_estimator_

def train_traditional_models(X_train, y_train) -> Dict[str, Any]:
    """
    Trains a suite of traditional machine learning models.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        A dictionary containing the trained model objects.
    """
    trained_models = {}
    for model_name, config in MODEL_CONFIG.items():
        best_model = tune_hyperparameters(
            config["model"],
            config["params"],
            X_train,
            y_train
        )
        trained_models[model_name] = best_model
        
    return trained_models

def save_model(model: Any, filepath: str):
    """Saves a model to a file using pickle."""
    print(f"Saving model to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath: str) -> Any:
    """Loads a model from a file."""
    print(f"Loading model from {filepath}...")
    with open(filepath, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    # --- Example Usage ---
    from sklearn.datasets import make_classification
    
    print("--- Generating sample data for demonstration ---")
    # Create a dummy dataset that mimics a text classification problem
    # It has many features, is sparse, and has two classes (spam/ham)
    X, y = make_classification(
        n_samples=1000, 
        n_features=500, 
        n_informative=50, 
        n_redundant=100,
        n_classes=2,
        random_state=42
    )
    
    # For MultinomialNB, features must be non-negative
    X[X < 0] = 0 

    print("\n--- Training Traditional Models ---")
    models = train_traditional_models(X, y)
    print(f"Trained {len(models)} models: {list(models.keys())}")

    # --- Saving and Loading Example ---
    # Get one of the trained models to demonstrate saving and loading
    lr_model = models.get("logistic_regression")
    model_path = "logistic_regression_model.pkl"
    
    if lr_model:
        save_model(lr_model, model_path)
        loaded_lr_model = load_model(model_path)
        
        # Verify the loaded model by making a prediction
        sample_prediction = loaded_lr_model.predict(X[:1])
        print(f"\nPrediction on first sample with loaded model: {sample_prediction}")
        print(f"Actual label for first sample: {y[:1]}")