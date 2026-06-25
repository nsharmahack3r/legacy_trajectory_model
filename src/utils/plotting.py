import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(histories, results, all_preds, all_trues, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "Elephant Trajectory Forecasting — Model Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"CNN": "#2196F3", "RCNN": "#4CAF50", "LSTM": "#FF5722"}
    names = list(histories.keys())

    # ── Row 0: training curves ──────────────────────────────
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[0, i])
        h = histories[name]
        ax.plot(h["train"], label="Train loss", color=colors[name], lw=1.5)
        ax.plot(
            h["val_loss"], label="Val loss", color=colors[name], lw=1.5, linestyle="--"
        )
        ax.set_title(f"{name} — Loss Curve", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Huber Loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # ── Row 1: displacement error over training ─────────────
    ax_dist = fig.add_subplot(gs[1, :2])
    for name in names:
        h = histories[name]
        ax_dist.plot(
            [d * 1000 for d in h["val_dist"]], label=name, color=colors[name], lw=2
        )
    ax_dist.set_title("Validation Displacement Error (m) per Epoch", fontweight="bold")
    ax_dist.set_xlabel("Epoch")
    ax_dist.set_ylabel("Mean dist error (m)")
    ax_dist.legend()
    ax_dist.grid(alpha=0.3)

    # ── Row 1, col 2: bar chart of final test errors ─────────
    ax_bar = fig.add_subplot(gs[1, 2])
    model_names = [r["model"] for r in results]
    errors_m = [r["mean_dist_error_m"] for r in results]
    bars = ax_bar.bar(
        model_names, errors_m, color=[colors[n] for n in model_names], width=0.5
    )
    ax_bar.bar_label(bars, fmt="%.1f m", padding=3, fontsize=9)
    ax_bar.set_title("Test Mean Displacement Error", fontweight="bold")
    ax_bar.set_ylabel("Error (m)")
    ax_bar.grid(axis="y", alpha=0.3)

    # ── Row 2: predicted vs true trajectory (first 200 pts) ──
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[2, i])
        preds = all_preds[name][:200]
        trues = all_trues[name][:200]
        ax.plot(trues[:, 0], trues[:, 1], "k.-", lw=1, ms=2, label="True", alpha=0.7)
        ax.plot(
            preds[:, 0],
            preds[:, 1],
            ".-",
            color=colors[name],
            lw=1,
            ms=2,
            label="Predicted",
            alpha=0.8,
        )
        ax.set_title(f"{name} — Predicted vs True Path", fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.savefig(
        f"{out_dir}/elephant_trajectory_results.png", dpi=150, bbox_inches="tight"
    )
    print(f"\n  Plot saved -> {out_dir}/elephant_trajectory_results.png")
    plt.close()


def save_model_plot(name, history, preds, trues, out_dir, color):
    """Save an individual model plot (loss curve + predicted vs true path)."""
    model_dir = os.path.join(out_dir, name)
    os.makedirs(model_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{name} — Training & Prediction", fontsize=14, fontweight="bold")

    # Loss curve
    ax1.plot(history["train"], label="Train loss", color=color, lw=1.5)
    ax1.plot(history["val_loss"], label="Val loss", color=color, lw=1.5, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Huber Loss")
    ax1.set_title("Loss Curve", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Predicted vs true path
    n = min(200, len(preds))
    ax2.plot(trues[:n, 0], trues[:n, 1], "k.-", lw=1, ms=2, label="True", alpha=0.7)
    ax2.plot(preds[:n, 0], preds[:n, 1], ".-", color=color, lw=1, ms=2, label="Predicted", alpha=0.8)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_title("Predicted vs True Path", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    path = os.path.join(model_dir, f"{name.lower()}_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Individual plot saved -> {path}")


def save_model_csv(name, preds, trues, out_dir):
    """Save model predictions and ground truth as CSV."""
    model_dir = os.path.join(out_dir, name)
    os.makedirs(model_dir, exist_ok=True)

    df = pd.DataFrame({
        "pred_lon": preds[:, 0],
        "pred_lat": preds[:, 1],
        "true_lon": trues[:, 0],
        "true_lat": trues[:, 1],
    })
    path = os.path.join(model_dir, f"{name.lower()}_predictions.csv")
    df.to_csv(path, index=False)
    print(f"  Predictions saved -> {path}")


def print_summary(results):
    sep = "=" * 65
    print(f"\n{sep}")
    print("  FINAL MODEL COMPARISON")
    print(sep)
    header = f"{'Model':<8} {'Test Loss':>12} {'Dist Error (m)':>16} {'Params':>10} {'Time (s)':>10}"
    print(header)
    print("-" * 65)
    for r in sorted(results, key=lambda x: x["mean_dist_error_m"]):
        print(
            f"{r['model']:<8} {r['test_loss']:>12.6f} "
            f"{r['mean_dist_error_m']:>16.2f} "
            f"{r['params']:>10,} {r['train_time_s']:>10.1f}"
        )
    print(sep)
    best = min(results, key=lambda x: x["mean_dist_error_m"])
    print(
        f"\n  Best model: {best['model']}  "
        f"({best['mean_dist_error_m']:.1f} m avg displacement error)\n"
    )
