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


class _TemporalBlock(nn.Module):
    """One residual block of two causal dilated Conv1d layers."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self._pad = pad
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        p = self._pad
        def trim(t): return t[:, :, :-p] if p else t  # causal: discard right-pad
        out = self.drop(self.relu(trim(self.conv1(x))))
        out = self.drop(self.relu(trim(self.conv2(out))))
        res = self.downsample(x) if self.downsample is not None else x
        return self.relu(out + res)


class DrowsinessTCN(nn.Module):
    """Dilated causal TCN for sequence classification.

    Default: 4 blocks with dilations [1, 2, 4, 8], kernel_size=3.
    Receptive field = 1 + 2*(k-1)*(1+2+4+8) = 61 frames — covers the full
    60-frame window with one frame to spare.
    """

    def __init__(self, input_size=NUM_FEATURES, num_channels=None,
                 kernel_size=3, dropout=0.2, num_classes=NUM_CLASSES):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 128, 128]

        blocks = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            blocks.append(_TemporalBlock(in_ch, out_ch, kernel_size, 2 ** i, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*blocks)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size) — permute for Conv1d
        out = self.network(x.permute(0, 2, 1))  # (batch, channels, seq_len)
        return self.fc(out[:, :, -1])            # classify from last time step


if __name__ == "__main__":
    from config import SEQUENCE_LENGTH
    dummy = torch.randn(4, SEQUENCE_LENGTH, NUM_FEATURES)
    print("LSTM output:", DrowsinessLSTM()(dummy).shape)
    print("TCN  output:", DrowsinessTCN()(dummy).shape)
