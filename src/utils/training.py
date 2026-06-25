import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.metrics import haversine_km


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, target_scaler):
    model.eval()
    total_loss = 0
    all_pred, all_true = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_loss += criterion(pred, y).item() * len(X)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())

    preds = np.concatenate(all_pred).reshape(-1, 2)
    trues = np.concatenate(all_true).reshape(-1, 2)

    preds_inv = target_scaler.inverse_transform(preds)
    trues_inv = target_scaler.inverse_transform(trues)

    dist_km = haversine_km(
        preds_inv[:, 0], preds_inv[:, 1], trues_inv[:, 0], trues_inv[:, 1]
    )
    return (total_loss / len(loader.dataset)), np.mean(dist_km), preds_inv, trues_inv


def run_model(
    name, model, train_loader, val_loader, test_loader, args, device, target_scaler
):
    print(f"\n{'=' * 55}")
    print(f"  Training: {name}")
    print(f"{'=' * 55}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.HuberLoss()

    history = {"train": [], "val_loss": [], "val_dist": []}
    best_val = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dist, _, _ = evaluate(
            model, val_loader, criterion, device, target_scaler
        )
        scheduler.step(val_loss)
        history["train"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_dist"].append(val_dist)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{args.epochs}  "
                f"train={tr_loss:.5f}  val={val_loss:.5f}  "
                f"dist={val_dist * 1000:.1f} m"
            )

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    test_loss, test_dist, preds, trues = evaluate(
        model, test_loader, criterion, device, target_scaler
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\n  >> Test loss: {test_loss:.5f}  |  "
        f"Mean displacement error: {test_dist * 1000:.1f} m  |  "
        f"Params: {params:,}  |  Time: {elapsed:.1f}s"
    )

    return (
        history,
        preds,
        trues,
        {
            "model": name,
            "test_loss": round(test_loss, 6),
            "mean_dist_error_m": round(test_dist * 1000, 2),
            "params": params,
            "train_time_s": round(elapsed, 1),
        },
    )
