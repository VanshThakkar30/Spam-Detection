import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict


class SpamLSTM(nn.Module):
    """
    An LSTM model for spam classification.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, n_layers: int,
                 drop_prob: float = 0.5):
        """
        Initializes the model layers.

        Args:
            vocab_size (int): The size of the vocabulary (number of unique words).
            embedding_dim (int): The size of the word embeddings.
            hidden_dim (int): The size of the LSTM's hidden state.
            output_dim (int): The size of the output (1 for binary classification).
            n_layers (int): The number of LSTM layers.
            drop_prob (float): The dropout probability.
        """
        super(SpamLSTM, self).__init__()

        # 1. Embedding Layer: Converts word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. LSTM Layer: Processes the sequence of embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # 3. Dropout Layer: For regularization
        self.dropout = nn.Dropout(drop_prob)

        # 4. Fully Connected Layer: Maps LSTM output to a final prediction
        self.fc = nn.Linear(hidden_dim, output_dim)

        # 5. Sigmoid Activation: Squashes output to a probability (0 to 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Defines the forward pass of the model.
        """
        # Convert word indices to embeddings
        embeds = self.embedding(x)

        # Pass embeddings to LSTM
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Get the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        # Apply dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # Apply sigmoid activation
        sig_out = self.sigmoid(out)

        # Squeeze to remove extra dimensions
        sig_out = sig_out.squeeze()

        return sig_out, hidden

    def init_hidden(self, batch_size: int, device):
        """Initializes hidden state."""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size)).to(device)
        c0 = torch.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size)).to(device)
        hidden = (h0, c0)
        return hidden


def train_pytorch_model(model, train_loader, optimizer, criterion, epochs, device):
    """
    The main training loop for a PyTorch model.
    """
    model.train()
    print("Starting PyTorch model training...")

    for epoch in range(epochs):
        h = model.init_hidden(train_loader.batch_size, device)

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            model.zero_grad()
            output, h = model(inputs, h)

            # Calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}')

    print("Training complete.")


if __name__ == '__main__':
    # --- Example Usage for Demonstration ---
    # This block shows how to use the model. In our real pipeline,
    # the data will come from our pre-processing steps.

    # 1. Define Hyperparameters and Device
    VOCAB_SIZE = 1000  # Size of our dummy vocabulary
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 100
    OUTPUT_DIM = 1
    N_LAYERS = 2
    EPOCHS = 3
    BATCH_SIZE = 16

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Create Dummy Data
    # In a real scenario, this data would be tokenized, padded text
    dummy_features = torch.randint(0, VOCAB_SIZE, (128, 50))  # 128 samples, 50 words each
    dummy_labels = torch.randint(0, 2, (128,))  # 128 labels (0 or 1)

    # Create DataLoader
    train_data = TensorDataset(dummy_features, dummy_labels)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

    # 3. Initialize Model, Optimizer, and Loss Function
    model = SpamLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification

    # 4. Train the Model
    train_pytorch_model(model, train_loader, optimizer, criterion, EPOCHS, device)