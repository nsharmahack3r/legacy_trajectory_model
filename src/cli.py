import argparse

from src.config import DataPaths


def get_args():
    p = argparse.ArgumentParser(
        description="Elephant trajectory forecasting — CNN / RCNN / LSTM"
    )
    p.add_argument(
        "--mode",
        choices=["test", "real"],
        default="test",
        help="'test' uses sample data; 'real' uses full dataset (default: test)",
    )
    p.add_argument("--data", default=None, help="Override: path to CSV data")
    p.add_argument("--seq_len", type=int, default=20, help="Input sequence length")
    p.add_argument("--pred_len", type=int, default=1, help="Steps ahead to predict")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="Fraction of data to use (per elephant). Use <1 to speed up.",
    )
    p.add_argument("--out_dir", default="./outputs")
    args = p.parse_args()

    # Resolve data path: explicit --data wins, otherwise use mode
    if args.data is None:
        paths = DataPaths()
        args.data = paths.actual_data if args.mode == "real" else paths.sample_data

    return args
