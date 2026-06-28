"""
src/diagnose_c13_nulls.py

Standalone check on the RAW GOES-18 pixel CSVs (before any resampling),
to find the actual source of the nulls in c13_bt_*/c02_refl_*.parquet.

Does NOT import from c13_features.py — deliberately re-reads the raw
files independently so this diagnostic can't inherit a bug from the
production code it's trying to investigate.

Run:
    python src/diagnose_c13_nulls.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import C13_PIXEL_DIR

NATIVE_GAP_THRESHOLD = pd.Timedelta(minutes=20)  # nominal cadence is ~10 min


def load_raw(fpath):
    df = pd.read_csv(fpath, low_memory=False)
    if '.geo' in df.columns:
        df = df.drop(columns=['.geo'])
    if 'system:index' in df.columns:
        df = df.drop(columns=['system:index'])
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
    df = df.set_index('datetime_utc').sort_index()
    pixel_id = (df['pixel_id'].dropna().iloc[0]
                if 'pixel_id' in df.columns else fpath.stem)
    return df, pixel_id


def main():
    files = sorted(C13_PIXEL_DIR.glob("*.csv"))
    print(f"Found {len(files)} raw pixel CSVs in {C13_PIXEL_DIR}\n")

    summary = []
    for f in files:
        df, pid = load_raw(f)
        idx = df.index
        n_rows = len(idx)
        t_min, t_max = idx.min(), idx.max()

        gaps = idx.to_series().diff()
        big_gaps = gaps[gaps > NATIVE_GAP_THRESHOLD]

        summary.append({
            'pixel_id': pid,
            'n_rows': n_rows,
            't_min': t_min,
            't_max': t_max,
            'n_big_gaps': len(big_gaps),
        })

        print(f"{pid:45s}  rows={n_rows:6d}  "
              f"range=[{t_min}  →  {t_max}]  big_gaps={len(big_gaps)}")
        for ts, gap in big_gaps.items():
            print(f"      gap of {gap}  ending at {ts}  "
                  f"(started {ts - gap})")

    summ_df = pd.DataFrame(summary)

    print("\n" + "=" * 70)
    print("CROSS-FILE CHECK")
    print("=" * 70)
    print(f"Unique t_min values : {summ_df['t_min'].nunique()}")
    if summ_df['t_min'].nunique() > 1:
        print(summ_df.groupby('t_min').size().sort_index())
        print("  ^ pixels do NOT all start at the same raw timestamp.")
        print("    This alone explains NaN after concat in load_all_pixels.")

    print(f"\nUnique t_max values : {summ_df['t_max'].nunique()}")
    if summ_df['t_max'].nunique() > 1:
        print(summ_df.groupby('t_max').size().sort_index())
        print("  ^ pixels do NOT all end at the same raw timestamp.")
        print("    This alone explains NaN after concat in load_all_pixels.")

    print(f"\nPixels with >0 native gaps >20min : "
          f"{(summ_df['n_big_gaps'] > 0).sum()} / {len(summ_df)}")
    print("  These gaps get silently ffilled in load_pixel_csv (no limit) —")
    print("  they will NOT show as NaN in the final parquet, but they ARE")
    print("  stale repeated values sitting in your training data right now.")


if __name__ == "__main__":
    main()