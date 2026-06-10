"""
Stage C - the recurrent model.

DrowsinessLSTM reads a sequence of (EAR, MAR) feature vectors and classifies
the whole window into one of the driver states.
"""
import torch
import torch.nn as nn

from config import NUM_FEATURES, NUM_CLASSES


class DrowsinessLSTM(nn.Module):
    def __init__(self, input_size=NUM_FEATURES, hidden_size=64,
                 num_layers=1, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        last = out[:, -1, :]        # take the final time-step's hidden state
        last = self.dropout(last)
        return self.fc(last)        # raw logits -> CrossEntropyLoss


if __name__ == "__main__":
    from config import SEQUENCE_LENGTH
    model = DrowsinessLSTM()
    dummy = torch.randn(4, SEQUENCE_LENGTH, NUM_FEATURES)
    print("output shape:", model(dummy).shape)  # expect (4, NUM_CLASSES)
