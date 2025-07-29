import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Import our project's functions and models
from src.api.predictor import predict_message  # We'll reuse our prediction pipeline logic
from src.models.pytorch_models import SpamLSTM
from src.data_processing.class_balancer import stratified_split  # To create a hold-out set for the meta-learner


class StackingEnsemble:
    """
    A custom stacking ensemble to combine the predictions of the SVC and BiLSTM models.
    """

    def __init__(self, svc_model_path, lstm_model_path, vocab_to_int_path):
        # Load the base models
        with open(svc_model_path, 'rb') as f:
            self.svc_model = pickle.load(f)

        # Load the vocabulary needed for the LSTM
        with open(vocab_to_int_path, 'rb') as f:
            vocab_to_int = pickle.load(f)

        # Initialize the LSTM model
        vocab_size = len(vocab_to_int) + 1
        self.lstm_model = SpamLSTM(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, output_dim=1, n_layers=2)
        self.lstm_model.load_state_dict(torch.load(lstm_model_path))
        self.lstm_model.eval()

        # The meta-learner that combines the predictions
        self.meta_learner = LogisticRegression()

    def _get_base_predictions(self, texts):
        """Helper function to get predictions from the base models."""
        svc_preds = []
        lstm_preds = []

        for text in texts:
            # Re-using the full SVC prediction pipeline for consistency
            svc_result = predict_message(text)
            svc_preds.append(svc_result['confidence']['spam'])

            # --- LSTM Prediction Logic (simplified from pipeline) ---
            # This part would need the full text-to-tensor pipeline
            # For a full implementation, we would build a dedicated predictor for the LSTM
            # Here we simulate a placeholder prediction
            lstm_preds.append(np.random.rand())  # Placeholder for LSTM prediction probability

        return np.column_stack((svc_preds, lstm_preds))

    def fit(self, X, y):
        """
        Trains the meta-learner on the predictions of the base models.

        Args:
            X (pd.Series): A series of raw text messages.
            y (pd.Series): The corresponding labels.
        """
        print("Training the meta-learner...")

        # Get the predictions from the base models to use as training data for the meta-learner
        base_model_predictions = self._get_base_predictions(X)

        # Train the meta-learner
        self.meta_learner.fit(base_model_predictions, y)
        print("Meta-learner training complete.")

    def predict(self, texts):
        """
        Makes a final prediction by combining the base model outputs.
        """
        # Get predictions from the base models
        base_model_predictions = self._get_base_predictions(texts)

        # Use the trained meta-learner to make the final prediction
        return self.meta_learner.predict(base_model_predictions)