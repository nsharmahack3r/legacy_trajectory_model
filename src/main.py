import warnings

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from src.cli import get_args
from src.config import FEATURE_COLS
from src.data import TrajectoryDataset, load_data
from src.cnn import CNNForecaster
from src.rcnn import RCNNForecaster
from src.lstm import LSTMForecaster
from src.utils.plotting import plot_results, print_summary, save_model_plot, save_model_csv
from src.utils.training import run_model

warnings.filterwarnings("ignore")


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    df = load_data(args.data, args.sample_frac)

    scaler = StandardScaler()

    # Build full dataset to fit scalers
    full_ds = TrajectoryDataset(
        df, scaler, args.seq_len, args.pred_len, fit_scaler=True
    )
    target_scaler = full_ds.target_scaler
    n = len(full_ds)
    print(f"Total windows: {n:,}  (seq_len={args.seq_len}, pred_len={args.pred_len})")

    idx = list(range(n))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.3, random_state=42)
    val_idx, te_idx = train_test_split(tmp_idx, test_size=0.5, random_state=42)

    train_loader = DataLoader(Subset(full_ds, tr_idx), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=args.batch)
    test_loader = DataLoader(Subset(full_ds, te_idx), batch_size=args.batch)

    n_feat = len(FEATURE_COLS)

    models = {
        "CNN": CNNForecaster(n_feat, args.seq_len, args.pred_len),
        "RCNN": RCNNForecaster(n_feat, args.seq_len, args.pred_len),
        "LSTM": LSTMForecaster(n_feat, args.seq_len, args.pred_len),
    }

    histories = {}
    all_preds = {}
    all_trues = {}
    results = []
    colors = {"CNN": "#2196F3", "RCNN": "#4CAF50", "LSTM": "#FF5722"}

    for name, model in models.items():
        model = model.to(device)
        h, preds, trues, metrics = run_model(
            name, model, train_loader, val_loader, test_loader,
            args, device, target_scaler,
        )
        histories[name] = h
        all_preds[name] = preds
        all_trues[name] = trues
        results.append(metrics)

        save_model_plot(name, h, preds, trues, args.out_dir, colors[name])
        save_model_csv(name, preds, trues, args.out_dir)

    print_summary(results)
    plot_results(histories, results, all_preds, all_trues, args.out_dir)


if __name__ == "__main__":
    main()
