"""
Stage C - the recurrent model.

DrowsinessLSTM reads a sequence of (EAR, MAR) feature vectors and classifies
the whole window into one of the driver states.
"""
import torch
import torch.nn as nn

from config import NUM_FEATURES, NUM_CLASSES


class DrowsinessLSTM(nn.Module):
    """LSTM-based sequence classifier for driver state detection.

    Architecture:
        LSTM (num_layers stacked) → Dropout → Linear → logits

    Input:  (batch, SEQUENCE_LENGTH, NUM_FEATURES)
            Each row is one frame: [EAR, MAR, yaw, pitch, roll, Δ_EAR, Δ_MAR]
    Output: (batch, num_classes) — raw logits, passed to CrossEntropyLoss.

    The LSTM sees the full 60-frame (~2 s) window before producing a single
    class prediction per window.  Dropout after the final hidden state
    regularises the classifier head and reduces overfitting.
    """

    def __init__(self, input_size=NUM_FEATURES, hidden_size=64,
                 num_layers=1, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # dropout between stacked layers (ignored when num_layers == 1)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Run the full sequence through the LSTM, then classify using only
        # the hidden state at the last time step (summarises the whole window).
        out, _ = self.lstm(x)           # (batch, seq_len, hidden_size)
        last = out[:, -1, :]            # (batch, hidden_size)
        last = self.dropout(last)
        return self.fc(last)            # (batch, num_classes) raw logits


if __name__ == "__main__":
    from config import SEQUENCE_LENGTH
    dummy = torch.randn(4, SEQUENCE_LENGTH, NUM_FEATURES)
    print("LSTM output:", DrowsinessLSTM()(dummy).shape)
