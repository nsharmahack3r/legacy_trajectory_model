# Elephant Movement Trajectory Forecasting

Benchmarking CNN, RCNN, and LSTM architectures for forecasting the next geographic position of African elephants from GPS tracking data.

---

## Table of Contents

- [Research Overview](#research-overview)
- [Data](#data)
- [Model Architectures](#model-architectures)
- [Training Protocol](#training-protocol)
- [Evaluation Metrics](#evaluation-metrics)
- [Setup](#setup)
- [Running Experiments](#running-experiments)
- [Outputs](#outputs)
- [Project Structure](#project-structure)

---

## Research Overview

This project compares three deep learning sequence-forecasting architectures on a **spatiotemporal trajectory prediction** task. Given a sliding window of *T* past movement steps (each described by 18 features), each model predicts the next *(longitude, latitude)* position.

The three architectures represent distinct design philosophies for time-series forecasting:

| Model | Philosophy | Citation Archetype |
|-------|-----------|-------------------|
| **CNN** | Pure convolutional — treats the time axis as a 1-D signal, stacks Conv1d layers with growing then shrinking channel dimensions, then pools across time | WaveNet / TC-ResNet |
| **RCNN** | Hybrid CNN+RNN — a convolutional feature extractor feeds into a GRU sequence encoder, merging local pattern detection with temporal state tracking | ConvLSTM / LRCN |
| **LSTM** | Pure recurrent with attention — a stacked LSTM encodes the full sequence, then a learnable attention layer weights each time step's contribution to the prediction | DA-RNN / Bahdanau-style |

All models use the same data splits, loss function, and evaluation pipeline so that comparisons are controlled.

---

## Data

### Source

GPS collar tracking data from African elephants in the Okavango Delta / Etosha region (Botswana / Namibia). The raw data has ~780,000 rows in the full set.

### Mode switching

| `--mode` | Data used | Rows |
|----------|-----------|------|
| `test` | `sample_data.csv` | ~80,000 |
| `real` | `actual_data.csv` | ~780,000 |

### Preprocessing pipeline

1. **Sort** by `(elephant_id, timestamp)` to ensure temporal order per animal.
2. **Encode** `time_of_day` as ordinal: `night=0, morning=1, afternoon=2, evening=3`.
3. **Fill NaN** with forward-fill then backward-fill per elephant; remaining NaNs → 0.
4. **Subsample** (optional `--sample_frac < 1.0`) per elephant for fast prototyping.

### Features (18 columns)

| Group | Features |
|-------|----------|
| **Position** | `longitude`, `latitude` |
| **Movement** | `speed_kmh`, `step_dist_m`, `turning_angle`, `bearing`, `dir_persistence`, `acceleration` |
| **Temporal** | `hour`, `season_code`, `time_of_day_code` |
| **Environmental** | `NDVI`, `EVI`, `LST_celsius`, `elevation_m`, `slope_deg`, `water_occ_1km`, `NDWI` |

**Target**: `longitude`, `latitude` (next step).

### Window construction

- Data is grouped by `elephant_id` — windows never cross between animals.
- A sliding window of `seq_len` (default 20) steps becomes `X`; the next `pred_len` (default 1) steps become `y`.
- **Feature scaling**: `StandardScaler` fit on all features combined.
- **Target scaling**: Separate `StandardScaler` fit on `(lon, lat)` only. Predictions are inverse-transformed before evaluation.

### Train / validation / test split

- **70 %** train, **15 %** validation, **15 %** test (fixed `random_state=42` per scikit-learn).
- Split is applied at the *window* level, not the *row* level.

---

## Model Architectures

### CNN Forecaster

A purely convolutional approach that treats the sequence as a 1-D signal.

```
Input:  (B, T=20, F=18)
        │
        ▼  permute(0, 2, 1)  →  (B, F=18, T=20)
        │
   ┌─────────────────────────────┐
   │  Conv1d(18 → 64,  k=3, P=1) │
   │  ReLU                       │
   │  Conv1d(64 → 128, k=3, P=1) │
   │  ReLU                       │
   │  Conv1d(128 → 64, k=3, P=1) │
   │  ReLU                       │
   └─────────────────────────────┘
        │  (B, C=64, T=20)
        ▼
   AdaptiveAvgPool1d(1)
        │  (B, C=64, 1)
        ▼
   Flatten
        │  (B, 64)
        ▼
   ┌──────────────────────┐
   │  Linear(64 → 64)      │
   │  ReLU                 │
   │  Linear(64 → 1 × 2)   │
   └──────────────────────┘
        │
        ▼  view(B, 1, 2)
   Output: (B, pred_len=1, 2)   →  (lon, lat)
```

**Parameters** (at defaults): ~23,000

**Key design choice**: No padding-based preservation of sequence length; the model pools the entire time dimension into a single vector per channel, then decodes to the prediction. This works best when the relevant signal is distributed across the whole window rather than localised at the end.

---

### RCNN Forecaster

Hybrid architecture: CNN layers extract local temporal features, then a GRU encodes the sequence dynamics.

```
Input:  (B, T=20, F=18)
        │
        ▼  permute(0, 2, 1)  →  (B, 18, 20)
        │
   ┌─────────────────────────────┐
   │  Conv1d(18 → 64, k=3, P=1)  │
   │  ReLU                       │
   │  Conv1d(64 → 64,  k=3, P=1) │
   │  ReLU                       │
   └─────────────────────────────┘
        │  (B, C=64, T=20)
        ▼  permute(0, 2, 1)  →  (B, T=20, 64)
        │
   ┌───────────────────────────────────┐
   │  GRU(64 → hidden=64)              │
   │  num_layers=2, batch_first=True   │
   │  dropout=0.2                      │
   └───────────────────────────────────┘
        │  h: (2, B, 64)
        ▼  h[-1]  (last layer's final hidden state)
   ┌──────────────────────┐
   │  Linear(64 → 64)      │
   │  ReLU                 │
   │  Linear(64 → 1 × 2)   │
   └──────────────────────┘
        │
        ▼  view(B, 1, 2)
   Output: (B, 1, 2)
```

**Parameters** (at defaults): ~59,000

**Key design choice**: Two Conv1d layers with the *same* channel count (64 → 64) act as a feature projector, not a dimensionality reducer. The GRU then receives a feature-rich 64-dim vector per time step. Using `h[-1]` (final layer hidden state) discards the per-step outputs, forcing the GRU to compress the sequence into a single state vector before decoding.

---

### LSTM Forecaster

Stacked LSTM with a learnable attention mechanism.

```
Input:  (B, T=20, F=18)
        │
        ▼
   ┌──────────────────────────────────────┐
   │  LSTM(18 → hidden=128)               │
   │  num_layers=2, batch_first=True      │
   │  dropout=0.2                         │
   └──────────────────────────────────────┘
        │  out: (B, T=20, 128)
        ▼
   ┌──────────────────────────────┐
   │  Score: attn(out) → (B,T,1)  │
   │  Softmax over dim=1          │
   │  Context = Σ(weight × out)   │  →  (B, 128)
   └──────────────────────────────┘
        │
   ┌──────────────────────┐
   │  Linear(128 → 64)     │
   │  ReLU                 │
   │  Linear(64 → 1 × 2)   │
   └──────────────────────┘
        │
        ▼  view(B, 1, 2)
   Output: (B, 1, 2)
```

**Parameters** (at defaults): ~154,000

**Key design choice**: The attention mechanism learns a scalar weight per time step via a small linear layer (`128 → 1`), then uses softmax to produce a convex combination over the time axis. This contrasts with the CNN (which pools uniformly) and the RCNN (which uses only the final hidden state). Attention lets the model *learn which past positions matter most* for predicting the next step — a useful property for animal movement where recent context may have varying relevance depending on behaviour state (foraging vs. travelling).

---

## Training Protocol

All three models share an identical training loop:

| Hyperparameter | Value |
|---------------|-------|
| Loss function | `nn.HuberLoss` (robust to GPS outliers) |
| Optimizer | Adam (`lr=1e-3`) |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5, monitor=val_loss) |
| Gradient clipping | max_norm = 1.0 |
| Epochs | 40 (configurable) |
| Batch size | 64 (configurable) |
| Model selection | Checkpoint with lowest validation loss |

### Per-epoch loop

```
for each epoch:
    train_epoch:
        for each batch:
            pred = model(X)
            loss = HuberLoss(pred, y)
            backward + clip_grads(1.0) + optimizer.step

    evaluate (val):
        for each batch:
            pred = model(X), collect all preds + trues
        inverse-transform preds/trues via target_scaler
        val_loss  = mean HuberLoss
        val_dist  = mean Haversine(km) × 1000  (metres)

    scheduler.step(val_loss)
    save best state dict if val_loss improved
```

## Evaluation Metrics

### Loss

`nn.HuberLoss` (smooth L1) — quadratic for small residuals, linear for large ones. This is less sensitive to GPS outliers than MSE.

### Displacement error

**Haversine distance** in kilometres (converted to metres for reporting), computed between predicted and true `(lon, lat)`:

```
a = sin²(Δlat/2) + cos(lat₁)·cos(lat₂)·sin²(Δlon/2)
d = R · 2 · arcsin(√a)                     where R = 6371 km
```

The Haversine formula accounts for Earth's curvature, giving physically meaningful displacement errors regardless of latitude.

### Reported metrics (per model)

- Test loss
- Mean displacement error (metres)
- Number of trainable parameters
- Training wall-clock time (seconds)

---

## Setup

### Prerequisites

- Python ≥ 3.11
- CUDA 12.6 (optional — falls back to CPU automatically)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Environment

```bash
# Clone
cd legacy_models_movement

# Create virtual environment + install deps with uv
uv sync

# Activate
.venv\Scripts\activate    # Windows
source .venv/bin/activate  # macOS / Linux
```

Or with pip:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install pandas scikit-learn matplotlib
```

---

## Running Experiments

### Quick test run (sample data, ~80K rows)

```bash
python main.py --mode test
```

Uses `outputs_test/` as the output directory by default (set with `--out_dir`).

### Full experiment (all ~780K rows)

```bash
python main.py --mode real --epochs 40 --out_dir ./outputs
```

### Customising the run

```bash
python main.py --mode real               \
    --seq_len 20                          \
    --pred_len 1                          \
    --batch 64                            \
    --epochs 40                           \
    --lr 1e-3                             \
    --sample_frac 0.5                     \  # use 50% of data per elephant
    --data /path/to/custom_data.csv       \  # override data file
    --out_dir ./outputs
```

### Available arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `test` | `test` or `real` — selects sample or full dataset |
| `--data` | `None` | Override data CSV path |
| `--seq_len` | 20 | Input window length |
| `--pred_len` | 1 | Forecast horizon |
| `--batch` | 64 | Batch size |
| `--epochs` | 40 | Training epochs |
| `--lr` | 1e-3 | Learning rate |
| `--sample_frac` | 1.0 | Fraction of rows per elephant to use |
| `--out_dir` | `./outputs` | Output directory |

---

## Outputs

After a run completes, the output directory contains:

```
outputs/
├── elephant_trajectory_results.png    # 3×3 comparison grid
│
├── CNN/
│   ├── cnn_results.png                # Individual loss + prediction plots
│   └── cnn_predictions.csv            # pred_lon, pred_lat, true_lon, true_lat
│
├── RCNN/
│   ├── rcnn_results.png
│   └── rcnn_predictions.csv
│
└── LSTM/
    ├── lstm_results.png
    └── lstm_predictions.csv
```

The **comparison grid** (`elephant_trajectory_results.png`) is a 3×3 figure:

| Row | Content |
|-----|---------|
| 1 | Training & validation loss curves (one per model) |
| 2 | Mean displacement error bar chart + test loss bar chart |
| 3 | Predicted vs. true trajectories (one panel per model) |

The **console summary** ranks models by ascending mean displacement error.

---

## Project Structure

```
legacy_models_movement/
├── main.py                        # Entry point
├── data.py                        # Re-exports DataPaths
├── pyproject.toml                 # Project config + dependencies
├── README.md
│
├── src/
│   ├── cli.py                     # Argument parser
│   ├── config.py                  # Feature/target column lists, data paths
│   ├── main.py                    # Orchestrator: data → split → train → compare
│   │
│   ├── cnn/model.py               # CNNForecaster
│   ├── lstm/model.py              # LSTMForecaster (with attention)
│   ├── rcnn/model.py              # RCNNForecaster
│   │
│   ├── data/data_loader.py        # TrajectoryDataset, load_data
│   │
│   └── utils/
│       ├── training.py            # train_epoch, evaluate, run_model
│       ├── metrics.py             # haversine_km
│       └── plotting.py            # plot_results, save_model_plot, save_model_csv
│
├── outputs/                       # Full-data experiment outputs
├── outputs_test/                  # Sample-data test outputs
│
└── .venv/                         # Virtual environment
```

---

## Citation

If you use this code in academic work, please cite the repository and the original data source.

```
@software{elephant_trajectory_forecasting,
  title     = {Elephant Movement Trajectory Forecasting},
  author    = {Penn State University},
  year      = {2025},
  url       = {https://github.com/...}
}
```
