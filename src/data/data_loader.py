import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src.config import FEATURE_COLS


class TrajectoryDataset(Dataset):
    """
    For each elephant separately, builds (X, y) sliding windows:
        X : [seq_len, n_features]
        y : [pred_len, 2]  (lon, lat)
    """

    def __init__(self, df, scaler, seq_len, pred_len, fit_scaler=False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples = []

        all_data = []
        for _, grp in df.groupby("elephant_id"):
            vals = grp[FEATURE_COLS].values.astype(np.float32)
            all_data.append(vals)

        combined = np.vstack(all_data)
        if fit_scaler:
            scaler.fit(combined)

        for vals in all_data:
            scaled = scaler.transform(vals)
            n = len(scaled)
            for i in range(n - seq_len - pred_len + 1):
                X = scaled[i : i + seq_len]
                # Targets: raw lon/lat from the *unscaled* future steps
                # (we scale targets separately using lon/lat indices)
                y_raw = vals[i + seq_len : i + seq_len + pred_len, :2]
                self.samples.append((X, y_raw))

        # Fit target scaler on lon/lat only
        self.target_scaler = StandardScaler()
        all_y = np.vstack([s[1] for s in self.samples])
        self.target_scaler.fit(all_y)
        self.samples = [(X, self.target_scaler.transform(y)) for X, y in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.tensor(X), torch.tensor(y.astype(np.float32))


def load_data(path, sample_frac=1.0):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["elephant_id", "timestamp"]).reset_index(drop=True)

    # Encode time_of_day
    tod_map = {"night": 0, "morning": 1, "afternoon": 2, "evening": 3}
    df["time_of_day_code"] = df["time_of_day"].map(tod_map).fillna(0).astype(int)

    # Fill NaN with forward fill per elephant, then 0
    df[FEATURE_COLS] = (
        df.groupby("elephant_id")[FEATURE_COLS]
        .transform(lambda g: g.ffill().bfill())
        .fillna(0)
    )

    if sample_frac < 1.0:
        df = df.groupby("elephant_id", group_keys=False).apply(
            lambda g: g.sample(frac=sample_frac, random_state=42).sort_values(
                "timestamp"
            )
        )
        df = df.reset_index(drop=True)

    print(f"Loaded {len(df):,} rows | {df['elephant_id'].nunique()} elephants")
    print(f"Date range: {df['timestamp'].min()}  ->  {df['timestamp'].max()}")
    return df
