import torch
import torch.nn as nn


class RCNNForecaster(nn.Module):
    """
    RCNN: CNN feature extractor → GRU sequence encoder → FC head.
    """

    def __init__(
        self, n_features, seq_len, pred_len, out_dim=2, hidden=64, gru_layers=2
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            64, hidden, num_layers=gru_layers, batch_first=True, dropout=0.2
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len * out_dim),
        )
        self.pred_len = pred_len
        self.out_dim = out_dim

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, n_feat, T)
        x = self.cnn(x).permute(0, 2, 1)  # (B, T, 64)
        _, h = self.gru(x)  # h: (layers, B, hidden)
        return self.head(h[-1]).view(-1, self.pred_len, self.out_dim)
