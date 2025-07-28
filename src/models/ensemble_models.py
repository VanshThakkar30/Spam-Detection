from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from typing import List, Any, Tuple


def create_voting_classifier(estimators: List[Tuple[str, Any]]):
    """
    Creates a hard voting classifier from a list of trained models.
    'Hard' voting uses the majority vote from the classifiers.
    """
    print("Creating Voting Classifier...")
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='hard'  # Use majority rule
    )
    return voting_clf


def create_stacking_classifier(estimators: List[Tuple[str, Any]]):
    """
    Creates a stacking classifier. It uses a meta-classifier (Logistic Regression)
    to combine the outputs of the base models.
    """
    print("Creating Stacking Classifier...")
    # The final estimator is a simple model that learns from the base models' predictions
    final_estimator = LogisticRegression()

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5  # Use cross-validation to generate the predictions for the meta-learner
    )
    return stacking_clf