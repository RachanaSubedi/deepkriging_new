"""
check_model_architecture.py

Diagnostic: inspect the saved DeepKriging checkpoint files directly to
determine whether they are single-point-estimate (output dim = 1) or
quantile (output dim = 3, for q10/q50/q90) models.

This does NOT import model.py (root or src/) at all — it reads the
raw state_dict from each fold_k_best.pt file and reports the shape of
the final linear layer's weight matrix. This sidesteps any ambiguity
about which model.py (root vs src/) predict.py is actually resolving
on this machine, since the checkpoint itself is unambiguous ground
truth for what was actually trained and saved.

Run from the repo root:
    python check_model_architecture.py

Or edit MODEL_DIR below if your models live somewhere else.
"""

import torch
import sys
from pathlib import Path

# Make repo root importable, regardless of where this script is run from
# (matches the sys.path.append pattern used by your other src/ scripts)
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ── EDIT IF NEEDED: point this at your actual model directory ────
try:
    from configs.config import MODEL_DIR
    print(f"Loaded MODEL_DIR from configs.config: {MODEL_DIR}\n")
except ImportError as e:
    print(f"Could not import MODEL_DIR from configs.config ({e}) —")
    print("edit MODEL_DIR below manually and re-run.\n")
    MODEL_DIR = Path("models")  # <-- change this if needed

N_FOLDS = 4


def inspect_checkpoint(path: Path):
    """Load a state_dict and report the final layer's output shape."""
    state_dict = torch.load(path, map_location="cpu")

    # Find all Linear-layer weight keys (they end in '.weight' and
    # are 2-D). The final one in the sequence is the output layer.
    linear_keys = [
        k for k, v in state_dict.items()
        if k.endswith(".weight") and v.dim() == 2
    ]

    if not linear_keys:
        print(f"  ⚠ No 2-D weight tensors found — unexpected state_dict structure.")
        print(f"    Keys present: {list(state_dict.keys())}")
        return None

    # Sort by the numeric index embedded in the key name
    # (e.g. 'net.0.weight', 'net.4.weight', ... 'net.12.weight')
    def key_index(k):
        parts = [p for p in k.split(".") if p.isdigit()]
        return int(parts[0]) if parts else -1

    linear_keys.sort(key=key_index)
    final_key = linear_keys[-1]
    final_shape = state_dict[final_key].shape  # (out_dim, in_dim)

    print(f"  All linear layer weight shapes, in order:")
    for k in linear_keys:
        print(f"    {k:20s} {tuple(state_dict[k].shape)}")

    out_dim = final_shape[0]
    print(f"\n  Final layer ({final_key}) output dim: {out_dim}")

    if out_dim == 1:
        print(f"  → SINGLE POINT ESTIMATE model (matches root model.py)")
    elif out_dim == 3:
        print(f"  → QUANTILE model, 3 outputs (matches src/model.py, q10/q50/q90)")
    else:
        print(f"  → UNEXPECTED output dim ({out_dim}) — neither known architecture matches")

    return out_dim


if __name__ == "__main__":
    print("=" * 60)
    print("  Checkpoint architecture inspector")
    print("=" * 60)

    results = []
    for k in range(N_FOLDS):
        ckpt_path = Path(MODEL_DIR) / f"fold_{k}_best.pt"
        print(f"\n[Fold {k}] {ckpt_path}")
        if not ckpt_path.exists():
            print(f"  ⚠ File not found — skipping.")
            continue
        out_dim = inspect_checkpoint(ckpt_path)
        results.append(out_dim)

    print("\n" + "=" * 60)
    if results and len(set(results)) == 1:
        print(f"  All {len(results)} folds consistent: output dim = {results[0]}")
    elif results:
        print(f"  ⚠ INCONSISTENT across folds: {results}")
        print(f"    Some folds may have been trained with a different")
        print(f"    model.py version than others — worth investigating")
        print(f"    which fold(s) are stale before trusting predict.py's")
        print(f"    ensemble average across them.")
    else:
        print("  No checkpoints could be inspected.")
    print("=" * 60)