"""
src/trainnew.py

ABLATION: basis-functions-only DeepKriging (no covariates).

Identical to train.py except the 18 covariate columns are dropped
from X before training — only the 411 Wendland RBF basis columns
are kept. Purpose: isolate how much of the model's predictive power
comes from spatial basis structure alone vs. the covariate stack
(bg_csi, C13 brightness temp, met variables, etc).

Compare outputs/validation_nocov/loso_summary.txt against
outputs/validation/loso_summary.txt (from train.py) to get the
delta attributable to covariates.

Leave-One-Station-Out (LOSO) cross-validation training for DeepKriging.

4 rounds:
    Round 0  →  hold out S1,  train on S2 S3 P2
    Round 1  →  hold out S2,  train on S1 S3 P2
    Round 2  →  hold out S3,  train on S1 S2 P2
    Round 3  →  hold out P2,  train on S1 S2 S3

Within each round:
    • StandardScaler fit on training samples only
    • Last 20 % of each training station's timesteps → validation
    • Early stopping on validation Huber loss (patience = 20)
    • Best model weights saved per fold

Outputs (outputs/models_nocov/ and outputs/validation_nocov/):
    fold_{k}_best.pt             best model weights
    fold_{k}_scaler_mean.npy     scaler mean  (411,)
    fold_{k}_scaler_std.npy      scaler std   (411,)
    fold_{k}_predictions.csv     test predictions with timestamps
    fold_{k}_history.csv         epoch-by-epoch train/val loss
    loso_summary.txt             RMSE / R² for all folds
    fig_loss_curves.png          training loss curves (outputs/figures_nocov/)

Run:
    python src/trainnew.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no popup window
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import (
    TRAIN_DIR, MODEL_DIR, VAL_DIR, BG_DIR, FIG_DIR,
    STATIONS,
    HIDDEN_SIZE, DROPOUT, WEIGHT_DECAY,
    LEARNING_RATE, BATCH_SIZE, MAX_EPOCHS,
)
from src.model import DeepKriging, count_parameters

# ── HYPERPARAMETERS ───────────────────────────────────────────
HUBER_DELTA      = 0.1
EARLY_STOP_PAT   = 20
VAL_FRACTION     = 0.20
CLEARSKY_MIN     = 10.0

STATION_NAMES    = list(STATIONS.keys())
DEVICE           = torch.device('cpu')

FOLD_COLORS = {
    'S1': '#e63946',
    'S2': '#2a9d8f',
    'S3': '#e76f51',
    'P2': '#264653',
}


# ── UTILITIES ─────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def standardise(X_train, X_val, X_test, n_basis=411):
    """
    Standardize ONLY the covariate columns (columns after n_basis).
    Leave the basis function columns unchanged — they are
    already in [0,1] from Phi_stations_scaled.npy.

    ABLATION NOTE: when X has exactly n_basis columns (no covariates
    left, as in trainnew.py), there are no columns to standardize —
    this function becomes a no-op identity pass-through, which is
    correct since basis values are already in [0,1].
    """
    mean = np.zeros(X_train.shape[1], dtype=np.float32)
    std  = np.ones(X_train.shape[1],  dtype=np.float32)

    if X_train.shape[1] > n_basis:
        cov_mean = X_train[:, n_basis:].mean(axis=0)
        cov_std  = X_train[:, n_basis:].std(axis=0)
        cov_std[cov_std < 1e-8] = 1.0

        mean[n_basis:] = cov_mean
        std[n_basis:]  = cov_std

    def scale(X):
        return (X - mean) / std

    return scale(X_train), scale(X_val), scale(X_test), mean, std


def temporal_val_split(X, y, ts, fraction=VAL_FRACTION):
    """
    Split (X, y) into train / val by time.
    Takes the LAST `fraction` of timesteps as validation.
    """
    n     = len(y)
    n_val = max(1, int(n * fraction))
    n_tr  = n - n_val

    order  = np.argsort(ts)
    X, y, ts = X[order], y[order], ts[order]

    return (X[:n_tr], y[:n_tr], ts[:n_tr],
            X[n_tr:], y[n_tr:], ts[n_tr:])


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ── TRAINING LOOP ─────────────────────────────────────────────

def train_one_fold(fold_k, X_tr, y_tr, X_val, y_val):
    """
    Train DeepKriging for one LOSO fold.
    Returns best model and epoch-by-epoch history list.
    """
    model     = DeepKriging(X_tr.shape[1], HIDDEN_SIZE, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    tr_loader  = make_loader(X_tr,  y_tr,  BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    best_val_loss  = float('inf')
    best_weights   = None
    patience_count = 0
    history        = []

    for epoch in range(1, MAX_EPOCHS + 1):

        # ── Train ─────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(y_tr)

        # ── Validate ──────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * len(yb)
        val_loss /= len(y_val)

        history.append({'epoch': epoch,
                        'train_loss': tr_loss,
                        'val_loss':   val_loss})

        # ── Early stopping ────────────────────────────────────
        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            best_weights   = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:>3d}  "
                  f"train={tr_loss:.5f}  val={val_loss:.5f}  "
                  f"patience={patience_count}/{EARLY_STOP_PAT}")

        if patience_count >= EARLY_STOP_PAT:
            print(f"    Early stop at epoch {epoch}  "
                  f"(best val={best_val_loss:.5f})")
            break

    model.load_state_dict(best_weights)
    return model, history


# ── PREDICT ───────────────────────────────────────────────────

def predict(model, X):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
    return out.cpu().numpy()


# ── PLOT LOSS CURVES ──────────────────────────────────────────

def plot_loss_curves(val_dir, fig_dir, station_names):
    """
    Reads fold_*_history.csv files saved during training and
    generates two figures:
      fig_loss_curves.png   — 4-panel, one per fold
      fig_loss_combined.png — all folds overlaid
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('DeepKriging — Training & Validation Loss per Fold',
                 fontsize=13, fontweight='bold')

    for ax, (k, station) in zip(axes.flat, enumerate(station_names)):
        hist     = pd.read_csv(val_dir / f"fold_{k}_history.csv")
        color    = FOLD_COLORS[station]
        best_idx = hist['val_loss'].idxmin()
        best_ep  = int(hist.loc[best_idx, 'epoch'])
        best_val = hist.loc[best_idx, 'val_loss']
        best_tr  = hist.loc[best_idx, 'train_loss']

        ax.plot(hist['epoch'], hist['train_loss'],
                color=color, lw=1.8, label='Train loss')
        ax.plot(hist['epoch'], hist['val_loss'],
                color=color, lw=1.8, ls='--', label='Val loss')

        # Mark best epoch
        ax.axvline(best_ep, color='black', lw=1.0, ls=':', alpha=0.5)
        ax.scatter([best_ep], [best_val], color='black', s=60, zorder=5)
        ax.annotate(f"best val={best_val:.5f}\nepoch {best_ep}",
                    (best_ep, best_val), xytext=(6, 4),
                    textcoords='offset points', fontsize=7.5)

        # Stats box
        gap = best_tr - best_val
        ax.text(0.98, 0.96,
                f"Train={best_tr:.5f}\nVal  ={best_val:.5f}\nGap ={gap:+.5f}",
                transform=ax.transAxes, fontsize=7.5,
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

        ax.set_title(f"Fold {k} — hold out {station}  "
                     f"({len(hist)} epochs)",
                     fontsize=10, fontweight='bold', color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Huber Loss')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out1 = fig_dir / "fig_loss_curves.png"
    plt.savefig(out1, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out1.name}")

    # ── Combined: all folds overlaid ─────────────────────────
    fig, (ax_tr, ax_val) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('DeepKriging — Loss Curves (All Folds)',
                 fontsize=13, fontweight='bold')

    for k, station in enumerate(station_names):
        hist  = pd.read_csv(val_dir / f"fold_{k}_history.csv")
        color = FOLD_COLORS[station]
        best_ep = int(hist.loc[hist['val_loss'].idxmin(), 'epoch'])

        ax_tr.plot(hist['epoch'], hist['train_loss'],
                   color=color, lw=1.8, label=f"Train {station}")
        ax_tr.axvline(best_ep, color=color, lw=0.8, ls=':', alpha=0.5)

        ax_val.plot(hist['epoch'], hist['val_loss'],
                    color=color, lw=1.8, label=f"Val {station}")
        ax_val.axvline(best_ep, color=color, lw=0.8, ls=':', alpha=0.5)
        ax_val.scatter([best_ep],
                       [hist.loc[hist['val_loss'].idxmin(), 'val_loss']],
                       color=color, s=60, zorder=5)

    for ax, title in [(ax_tr, 'Training Loss'), (ax_val, 'Validation Loss')]:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Huber Loss')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    out2 = fig_dir / "fig_loss_combined.png"
    plt.savefig(out2, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out2.name}")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  train.py — DeepKriging LOSO Cross-Validation")
    print("=" * 60)

    # ── Load training matrix ──────────────────────────────────
    print("\n[1/3] Loading training matrix...")
    X        = np.load(TRAIN_DIR / "X.npy")
    y        = np.load(TRAIN_DIR / "y.npy")
    fold_ids = np.load(TRAIN_DIR / "fold_ids.npy")
    ts_ns    = np.load(TRAIN_DIR / "timestamps.npy")

    # ── ABLATION: drop the 18 covariate columns ───────────────
    # X is assembled in training_matrix.py as [phi (411) | covariates (18)]
    # so covariates are always the LAST 18 columns. Slicing them off
    # leaves a pure basis-function input.
    N_COV = 18
    print(f"\n  ⚠ ABLATION MODE — dropping last {N_COV} covariate columns")
    print(f"  X shape before: {X.shape}")
    X = X[:, :-N_COV]
    print(f"  X shape after : {X.shape}  (basis functions only)")

    # ── Redirect outputs so the real train.py results are untouched
    MODEL_DIR = MODEL_DIR.parent / (MODEL_DIR.name + "_nocov")
    VAL_DIR   = VAL_DIR.parent   / (VAL_DIR.name   + "_nocov")
    FIG_DIR   = FIG_DIR.parent   / (FIG_DIR.name   + "_nocov")

    print(f"  X shape    : {X.shape}")
    print(f"  y shape    : {y.shape}")
    print(f"  y range    : [{y.min():.3f}, {y.max():.3f}]")
    for i, s in enumerate(STATION_NAMES):
        print(f"  Fold {i} ({s}) : {(fold_ids == i).sum()} samples")

    print("\n  Loading background CSI and clearsky for GHI eval...")
    bg_csi   = pd.read_parquet(BG_DIR / "bg_csi_stations.parquet")
    bg_clear = pd.read_parquet(BG_DIR / "clearsky_pvlib_stations.parquet")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True,   exist_ok=True)

    # ── LOSO loop ─────────────────────────────────────────────
    print(f"\n[2/3] LOSO training ({len(STATION_NAMES)} folds)...")
    model_tmp = DeepKriging(X.shape[1], HIDDEN_SIZE, DROPOUT)
    print(f"  Architecture : input={X.shape[1]} → "
          f"3×Dense({HIDDEN_SIZE},ReLU,Drop{DROPOUT}) → 1")
    print(f"  Parameters   : {count_parameters(model_tmp):,}")
    print(f"  Device       : {DEVICE}")
    print(f"  Loss         : MSE (peak-sensitive)")
    print(f"  Optimizer    : Adam  lr={LEARNING_RATE}  wd={WEIGHT_DECAY}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Max epochs   : {MAX_EPOCHS}  patience={EARLY_STOP_PAT}")

    fold_results = []

    for k, test_station in enumerate(STATION_NAMES):

        print(f"\n{'─'*60}")
        print(f"  FOLD {k} — Test station: {test_station}")
        train_stations = [s for s in STATION_NAMES if s != test_station]
        print(f"           Train stations: {train_stations}")
        print(f"{'─'*60}")

        # Separate test data
        test_mask = fold_ids == k
        X_test    = X[test_mask]
        y_test    = y[test_mask]
        ts_test   = ts_ns[test_mask]

        # Build train + val from remaining 3 stations
        X_rem   = X[~test_mask]
        y_rem   = y[~test_mask]
        ts_rem  = ts_ns[~test_mask]
        fid_rem = fold_ids[~test_mask]

        X_tr_list, y_tr_list, X_val_list, y_val_list = [], [], [], []

        for j, s in enumerate(STATION_NAMES):
            if s == test_station:
                continue
            s_mask = fid_rem == j
            Xs, ys, tss = X_rem[s_mask], y_rem[s_mask], ts_rem[s_mask]
            Xtr, ytr, _, Xvl, yvl, _ = temporal_val_split(Xs, ys, tss)
            X_tr_list.append(Xtr);  y_tr_list.append(ytr)
            X_val_list.append(Xvl); y_val_list.append(yvl)

        X_tr  = np.vstack(X_tr_list);   y_tr  = np.concatenate(y_tr_list)
        X_val = np.vstack(X_val_list);  y_val = np.concatenate(y_val_list)

        # Oversample cloud enhancement events (residual > 0.2)
        enhance_mask = y_tr > 0.85
        if enhance_mask.sum() > 0:
            X_enh = np.tile(X_tr[enhance_mask], (3, 1))
            y_enh = np.tile(y_tr[enhance_mask], 3)
            X_tr = np.vstack([X_tr, X_enh])
            y_tr = np.concatenate([y_tr, y_enh])
            print(f"  Oversampled {enhance_mask.sum()} enhancement events → "
                  f"{len(y_enh)} extra samples added")

        print(f"  Train samples : {len(y_tr)}")
        print(f"  Val samples   : {len(y_val)}")
        print(f"  Test samples  : {len(y_test)}")

        # Normalise (fit on train only)
        X_tr_sc, X_val_sc, X_test_sc, sc_mean, sc_std = \
            standardise(X_tr, X_val, X_test)

        # Train
        t0 = time.time()
        model, history = train_one_fold(k, X_tr_sc, y_tr, X_val_sc, y_val)
        elapsed = time.time() - t0
        print(f"  Training time : {elapsed/60:.1f} min")

        # ── Save training history ─────────────────────────────
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(VAL_DIR / f"fold_{k}_history.csv", index=False)
        print(f"  History saved : fold_{k}_history.csv  "
              f"({len(history)} epochs  "
              f"best epoch={hist_df['val_loss'].idxmin()+1})")

        # Predict on test station
        csi_pred_raw = predict(model, X_test_sc)
        csi_true = y_test

        rmse_csi_raw = rmse(csi_true, csi_pred_raw)
        r2_csi_raw   = r2_score(csi_true, csi_pred_raw)

        # Reconstruct GHI
        ts_dt    = pd.to_datetime(ts_test, unit='ns', utc=True) \
                     .tz_convert('America/Los_Angeles')
        bg_clear_s = bg_clear[test_station].reindex(ts_dt).values
        bg_csi_s = bg_csi[test_station].reindex(ts_dt).values
        csi_pred = np.clip(csi_pred_raw, 0.0, 1.3)  # predicted CSI

        low_sun = bg_clear_s < 200.0
        if low_sun.any():
            blend_w = np.clip(bg_clear_s[low_sun] / 200.0, 0.0, 1.0)
            csi_pred[low_sun] = (blend_w * csi_pred_raw[low_sun] +
                                 (1 - blend_w) * np.clip(bg_csi_s[low_sun], 0.0, 1.0))
        final_cap = np.where(bg_clear_s < 100, 0.85,
                             np.where(bg_clear_s < 200, 0.90,
                                      np.where(bg_clear_s < 350, 0.95, 1.3)))
        csi_pred = np.minimum(csi_pred, final_cap)
        csi_pred = np.clip(csi_pred, 0.0, 1.3)  # overall ceiling unchanged for daytime

        csi_true = np.clip(csi_true, 0.0, 1.3)  # measured CSI
        ghi_pred = csi_pred * bg_clear_s
        ghi_true = csi_true * bg_clear_s

        day      = bg_clear_s >= CLEARSKY_MIN
        rmse_ghi = rmse(ghi_true[day], ghi_pred[day])
        r2_ghi   = r2_score(ghi_true[day], ghi_pred[day])

        print(f"\n  ── Test Results ({test_station}) ──────────────────")
        print(f"  CSI (raw out) RMSE={rmse_csi_raw:.4f}  R²={r2_csi_raw:.4f}  "
              f"← unclipped model output vs measured CSI")
        print(f"  CSI (clipped) RMSE={rmse(csi_true[day], csi_pred[day]):.4f}  "
              f"R²={r2_score(csi_true[day], csi_pred[day]):.4f}")
        print(f"  GHI (W/m²)    RMSE={rmse_ghi:.2f}    R²={r2_ghi:.4f}")

        # Save model + scaler
        torch.save(model.state_dict(), MODEL_DIR / f"fold_{k}_best.pt")
        np.save(MODEL_DIR / f"fold_{k}_scaler_mean.npy", sc_mean)
        np.save(MODEL_DIR / f"fold_{k}_scaler_std.npy",  sc_std)

        # Save predictions
        pred_df = pd.DataFrame({
            'datetime_local': ts_dt,
            'station': test_station,
            'csi_true': csi_true,
            'csi_pred': csi_pred,  # clipped
            'ghi_true': ghi_true,
            'ghi_pred': ghi_pred,
            'bg_clearsky': bg_clear_s,
        })

        pred_df.to_csv(VAL_DIR / f"fold_{k}_{test_station}_predictions.csv",
                       index=False)

        fold_results.append({
            'fold'        : k,
            'test_station': test_station,
            'n_train'     : len(y_tr),
            'n_test'      : len(y_test),
            'rmse_csi_raw'  : rmse_csi_raw,
            'r2_csi_raw'    : r2_csi_raw,
            'rmse_ghi'    : rmse_ghi,
            'r2_ghi'      : r2_ghi,
            'epochs_run'  : len(history),
        })

    # ── Summary ───────────────────────────────────────────────
    print(f"\n[3/3] LOSO Summary")
    print(f"{'─'*60}")
    print(f"{'Fold':<6} {'Station':<8} {'RMSE_CSI':>10} {'R²_CSI':>8} "
          f"{'RMSE_GHI':>10} {'R²_GHI':>8}")
    print(f"{'─'*60}")

    results_df = pd.DataFrame(fold_results)
    for _, row in results_df.iterrows():
        print(f"  {int(row.fold):<4} {row.test_station:<8} "
              f"{row.rmse_csi_raw:>10.4f} {row.r2_csi_raw:>8.4f} "
              f"{row.rmse_ghi:>10.2f} {row.r2_ghi:>8.4f}")
    print(f"{'─'*60}")
    print(f"  {'Mean':<12} "
          f"{results_df.rmse_csi_raw.mean():>10.4f} "
          f"{results_df.r2_csi_raw.mean():>8.4f} "
          f"{results_df.rmse_ghi.mean():>10.2f} "
          f"{results_df.r2_ghi.mean():>8.4f}")

    summary_lines = [
        "DeepKriging LOSO Cross-Validation Results", "=" * 50,
        results_df.to_string(index=False), "",
        f"Mean RMSE (CSI residual) : {results_df.rmse_csi_raw.mean():.4f}",
        f"Mean R²   (CSI residual) : {results_df.r2_csi_raw.mean():.4f}",
        f"Mean RMSE (GHI W/m²)     : {results_df.rmse_ghi.mean():.2f}",
        f"Mean R²   (GHI)          : {results_df.r2_ghi.mean():.4f}",
    ]
    (VAL_DIR / "loso_summary.txt").write_text('\n'.join(summary_lines))
    results_df.to_csv(VAL_DIR / "loso_results.csv", index=False)

    # ── Plot loss curves ──────────────────────────────────────
    print("\nGenerating loss curves...")
    plot_loss_curves(VAL_DIR, FIG_DIR, STATION_NAMES)

    print(f"\n✓ Training complete")
    print(f"  Models  → {MODEL_DIR}")
    print(f"  Results → {VAL_DIR}")
    print(f"  Figures → {FIG_DIR}")