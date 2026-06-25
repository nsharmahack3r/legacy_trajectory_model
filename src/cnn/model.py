import torch.nn as nn


class CNNForecaster(nn.Module):
    """
    1-D CNN over the time dimension.
    Architecture: 3 conv layers → global avg pool → FC head
    """

    def __init__(self, n_features, seq_len, pred_len, out_dim=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len * out_dim),
        )
        self.pred_len = pred_len
        self.out_dim = out_dim

    def forward(self, x):  # x: (B, seq_len, n_feat)
        x = x.permute(0, 2, 1)  # → (B, n_feat, seq_len)
        x = self.conv(x)
        x = self.pool(x)
        return self.head(x).view(-1, self.pred_len, self.out_dim)
