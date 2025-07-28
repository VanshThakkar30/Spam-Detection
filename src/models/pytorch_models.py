import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from typing import Dict


class SpamLSTM(nn.Module):
    """
    A Bidirectional LSTM (BiLSTM) model for spam classification.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, n_layers: int,
                 drop_prob: float = 0.5):
        super(SpamLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # --- Key Change 1: Make the LSTM Bidirectional ---
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(drop_prob)

        # --- Key Change 2: Adjust the Fully Connected Layer ---
        # The input dimension is now hidden_dim * 2 because the outputs of the
        # forward and backward passes are concatenated.
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out[:, -1, :]

        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sigmoid(out)

        sig_out = sig_out.squeeze()
        return sig_out, hidden

    def init_hidden(self, batch_size: int, device):
        # --- Key Change 3: Adjust Hidden State Initialization ---
        # The first dimension is now num_layers * 2 for a bidirectional model.
        h0 = torch.zeros((self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size)).to(device)
        c0 = torch.zeros((self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size)).to(device)
        hidden = (h0, c0)
        return hidden


def train_pytorch_model(model, train_loader, validation_loader, optimizer, criterion, epochs, device, patience=3):
    """
    The main training loop with early stopping and F1 score calculation.
    """
    model.train()
    print("Starting PyTorch model training with early stopping...")

    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        h = model.init_hidden(train_loader.batch_size, device)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            h = tuple([each.data for each in h])

            model.zero_grad()
            output, h = model(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            val_h = model.init_hidden(validation_loader.batch_size, device)
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                val_h = tuple([each.data for each in val_h])

                output, val_h = model(inputs, val_h)
                preds = (output.squeeze() > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds)
        avg_train_loss = train_loss / len(train_loader)

        print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Validation F1: {val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict()
            print(f'Validation F1 improved! Saving model.')
        else:
            patience_counter += 1
            print(f'Validation F1 did not improve. Patience: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print('Early stopping triggered!')
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
    print(f"\nTraining complete. Best validation F1-score: {best_val_f1:.4f}")