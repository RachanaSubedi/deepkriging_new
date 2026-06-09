"""
src/train.py

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

Outputs (outputs/models/ and outputs/validation/):
    fold_{k}_best.pt             best model weights
    fold_{k}_scaler_mean.npy     scaler mean  (426,)
    fold_{k}_scaler_std.npy      scaler std   (426,)
    fold_{k}_predictions.csv     test predictions with timestamps
    loso_summary.txt             RMSE / R² for all folds

Run:
    python src/train.py
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

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import (
    TRAIN_DIR, MODEL_DIR, VAL_DIR, BG_DIR,
    STATIONS,
    HIDDEN_SIZE, DROPOUT, WEIGHT_DECAY,
    LEARNING_RATE, BATCH_SIZE, MAX_EPOCHS,
)
from src.model import DeepKriging, count_parameters

# ── HYPERPARAMETERS ───────────────────────────────────────────
HUBER_DELTA      = 0.1    # Huber loss δ — robust to cloud-enhancement outliers
EARLY_STOP_PAT   = 20     # epochs without val improvement before stopping
VAL_FRACTION     = 0.20   # last 20 % of each training station → validation
CLEARSKY_MIN     = 10.0   # W/m² — used only for GHI reconstruction check

STATION_NAMES    = list(STATIONS.keys())   # ['S1', 'S2', 'S3', 'P2']
DEVICE           = torch.device('cpu')     # CPU is sufficient for this model size


# ── UTILITIES ─────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def standardise(X_train, X_val, X_test, n_basis=411):
    """
    Standardize ONLY the covariate columns (last 15).
    Leave the 411 basis function columns unchanged — they are
    already min-max scaled to [0,1] in Phi_stations_scaled.npy.

    Re-standardizing phi values with per-fold statistics causes
    phi(test_station) to fall outside the training distribution,
    producing extreme network outputs and catastrophic test RMSE.
    """
    mean = np.zeros(X_train.shape[1], dtype=np.float32)
    std  = np.ones(X_train.shape[1],  dtype=np.float32)

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
    Operates on a single station's data (already sorted by ts).
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
    Returns best model and training history.
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
    ts_ns    = np.load(TRAIN_DIR / "timestamps.npy")   # int64 ns since epoch

    print(f"  X shape    : {X.shape}")
    print(f"  y shape    : {y.shape}")
    print(f"  y range    : [{y.min():.3f}, {y.max():.3f}]")
    for i, s in enumerate(STATION_NAMES):
        n = (fold_ids == i).sum()
        print(f"  Fold {i} ({s}) : {n} samples")

    # ── Load background field for GHI reconstruction ──────────
    print("\n  Loading background CSI and clearsky for GHI eval...")
    bg_csi   = pd.read_parquet(BG_DIR / "bg_csi_stations.parquet")
    bg_clear = pd.read_parquet(BG_DIR / "bg_clearsky_stations.parquet")

    # ── Setup output dirs ─────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True,   exist_ok=True)

    # ── LOSO loop ─────────────────────────────────────────────
    print(f"\n[2/3] LOSO training ({len(STATION_NAMES)} folds)...")
    print(f"  Architecture : input={X.shape[1]} → "
          f"BN → 3×Dense({HIDDEN_SIZE},ReLU,Drop{DROPOUT}) → 1")
    model_tmp = DeepKriging(X.shape[1], HIDDEN_SIZE, DROPOUT)
    print(f"  Parameters   : {count_parameters(model_tmp):,}")
    print(f"  Device       : {DEVICE}")
    print(f"  Loss         : Huber (δ={HUBER_DELTA})")
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

        # ── Separate test data ────────────────────────────────
        test_mask  = fold_ids == k
        X_test     = X[test_mask]
        y_test     = y[test_mask]
        ts_test    = ts_ns[test_mask]

        # ── Build train + val from remaining 3 stations ───────
        train_mask = ~test_mask
        X_rem      = X[train_mask]
        y_rem      = y[train_mask]
        ts_rem     = ts_ns[train_mask]
        fid_rem    = fold_ids[train_mask]

        X_tr_list, y_tr_list = [], []
        X_val_list, y_val_list = [], []

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

        print(f"  Train samples : {len(y_tr)}")
        print(f"  Val samples   : {len(y_val)}")
        print(f"  Test samples  : {len(y_test)}")

        # ── Normalise (fit on train only) ─────────────────────
        X_tr_sc, X_val_sc, X_test_sc, sc_mean, sc_std = \
            standardise(X_tr, X_val, X_test)

        # ── Train ─────────────────────────────────────────────
        t0    = time.time()
        model, history = train_one_fold(k, X_tr_sc, y_tr,
                                           X_val_sc, y_val)
        elapsed = time.time() - t0
        print(f"  Training time : {elapsed/60:.1f} min")

        # ── Predict on test station ───────────────────────────
        resid_pred = predict(model, X_test_sc)
        resid_true = y_test

        rmse_resid = rmse(resid_true, resid_pred)
        r2_resid   = r2_score(resid_true, resid_pred)

        # ── Reconstruct GHI for evaluation ───────────────────
        ts_dt    = pd.to_datetime(ts_test, unit='ns', utc=True) \
                     .tz_convert('America/Los_Angeles')

        bg_csi_s   = bg_csi[test_station].reindex(ts_dt).values
        bg_clear_s = bg_clear[test_station].reindex(ts_dt).values

        csi_pred  = np.clip(bg_csi_s + resid_pred, 0.0, 2.0)
        csi_true  = np.clip(bg_csi_s + resid_true, 0.0, 2.0)
        ghi_pred  = csi_pred * bg_clear_s
        ghi_true  = csi_true * bg_clear_s

        # Only evaluate GHI where clearsky is meaningful
        day = bg_clear_s >= CLEARSKY_MIN
        rmse_ghi = rmse(ghi_true[day], ghi_pred[day])
        r2_ghi   = r2_score(ghi_true[day], ghi_pred[day])

        print(f"\n  ── Test Results ({test_station}) ──────────────────")
        print(f"  CSI residual  RMSE={rmse_resid:.4f}  R²={r2_resid:.4f}")
        print(f"  GHI (W/m²)    RMSE={rmse_ghi:.2f}    R²={r2_ghi:.4f}")

        # ── Save model + scaler ───────────────────────────────
        torch.save(model.state_dict(),
                   MODEL_DIR / f"fold_{k}_best.pt")
        np.save(MODEL_DIR / f"fold_{k}_scaler_mean.npy", sc_mean)
        np.save(MODEL_DIR / f"fold_{k}_scaler_std.npy",  sc_std)

        # ── Save predictions ──────────────────────────────────
        pred_df = pd.DataFrame({
            'datetime_local'  : ts_dt,
            'station'         : test_station,
            'resid_true'      : resid_true,
            'resid_pred'      : resid_pred,
            'csi_true'        : csi_true,
            'csi_pred'        : csi_pred,
            'ghi_true'        : ghi_true,
            'ghi_pred'        : ghi_pred,
            'bg_clearsky'     : bg_clear_s,
        })
        pred_df.to_csv(VAL_DIR / f"fold_{k}_{test_station}_predictions.csv",
                       index=False)

        fold_results.append({
            'fold'         : k,
            'test_station' : test_station,
            'n_train'      : len(y_tr),
            'n_test'       : len(y_test),
            'rmse_resid'   : rmse_resid,
            'r2_resid'     : r2_resid,
            'rmse_ghi'     : rmse_ghi,
            'r2_ghi'       : r2_ghi,
            'epochs_run'   : len(history),
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
              f"{row.rmse_resid:>10.4f} {row.r2_resid:>8.4f} "
              f"{row.rmse_ghi:>10.2f} {row.r2_ghi:>8.4f}")

    print(f"{'─'*60}")
    print(f"  {'Mean':<12} "
          f"{results_df.rmse_resid.mean():>10.4f} "
          f"{results_df.r2_resid.mean():>8.4f} "
          f"{results_df.rmse_ghi.mean():>10.2f} "
          f"{results_df.r2_ghi.mean():>8.4f}")

    # Save summary
    summary_lines = [
        "DeepKriging LOSO Cross-Validation Results",
        "=" * 50,
        results_df.to_string(index=False),
        "",
        f"Mean RMSE (CSI residual) : {results_df.rmse_resid.mean():.4f}",
        f"Mean R²   (CSI residual) : {results_df.r2_resid.mean():.4f}",
        f"Mean RMSE (GHI W/m²)     : {results_df.rmse_ghi.mean():.2f}",
        f"Mean R²   (GHI)          : {results_df.r2_ghi.mean():.4f}",
    ]
    (VAL_DIR / "loso_summary.txt").write_text('\n'.join(summary_lines))
    results_df.to_csv(VAL_DIR / "loso_results.csv", index=False)

    print(f"\n✓ Training complete")
    print(f"  Models  → {MODEL_DIR}")
    print(f"  Results → {VAL_DIR}")