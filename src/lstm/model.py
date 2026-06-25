import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    Stacked LSTM with attention-weighted output.
    """

    def __init__(
        self, n_features, seq_len, pred_len, out_dim=2, hidden=128, lstm_layers=2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=lstm_layers, batch_first=True, dropout=0.2
        )
        # Attention
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len * out_dim),
        )
        self.pred_len = pred_len
        self.out_dim = out_dim

    def forward(self, x):
        out, _ = self.lstm(x)  # (B, T, hidden)
        weights = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
        ctx = (weights * out).sum(dim=1)  # (B, hidden)
        return self.head(ctx).view(-1, self.pred_len, self.out_dim)
