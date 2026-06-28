"""
gee/merge_s1_gap.py

Splices the gap-fix extraction (goes18_c13_S1_GAPFIX.csv) into the
original S1 pixel CSV, filling the ~95-row gap at the start of the
year (2023-12-31 16:00 -> 2024-01-01 00:00 local) with real recovered
data instead of leaving it NaN or copying from a different pixel.

EDIT THE TWO PATHS BELOW before running.

Run:
    python gee/merge_s1_gap.py
"""

import pandas as pd
from pathlib import Path

# ── EDIT THESE TWO PATHS ───────────────────────────────────────
ORIGINAL_S1_CSV = Path(r"C:\Users\C838122727\Documents\CSU\research\deepkriging\deepkriging_solar_Copy\data\raw\goes_c13\extracted_pixels\goes18_c13_c02_px_46p594595_n119p155673.csv")
GAPFIX_CSV      = Path(r"C:\Users\C838122727\Documents\CSU\research\deepkriging\deepkriging_solar_Copy\data\raw\goes_c13\extracted_pixels\goes18_c13_S1_GAPFIX.csv")
# ────────────────────────────────────────────────────────────────


def load_raw(fpath):
    df = pd.read_csv(fpath, low_memory=False)
    if '.geo' in df.columns:
        df = df.drop(columns=['.geo'])
    if 'system:index' in df.columns:
        df = df.drop(columns=['system:index'])
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  merge_s1_gap.py — Splicing S1 gap-fix into original CSV")
    print("=" * 60)

    if not ORIGINAL_S1_CSV.exists():
        raise FileNotFoundError(
            f"Original S1 CSV not found: {ORIGINAL_S1_CSV}\n"
            f"Edit the path at the top of this script."
        )
    if not GAPFIX_CSV.exists():
        raise FileNotFoundError(
            f"Gap-fix CSV not found: {GAPFIX_CSV}\n"
            f"Download it from Drive first, then edit the path above."
        )

    original = load_raw(ORIGINAL_S1_CSV)
    gapfix   = load_raw(GAPFIX_CSV)

    print(f"\n  Original : {len(original)} rows, "
          f"{original['datetime_utc'].min()} → {original['datetime_utc'].max()}")
    print(f"  Gap-fix  : {len(gapfix)} rows, "
          f"{gapfix['datetime_utc'].min()} → {gapfix['datetime_utc'].max()}")

    # Combine, keeping gap-fix rows where timestamps overlap with original
    # (gap-fix is the more targeted/recent extraction for that window —
    #  in practice these should be identical values where they overlap,
    #  this just avoids depending on which file "wins" being ambiguous)
    combined = pd.concat([original, gapfix], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset='datetime_utc', keep='last')
    after = len(combined)
    combined = combined.sort_values('datetime_utc').reset_index(drop=True)

    print(f"\n  Combined : {after} rows ({before - after} duplicate timestamps resolved)")

    # Sanity check: confirm the original gap is now filled
    full_range = pd.date_range(
        combined['datetime_utc'].min(), combined['datetime_utc'].max(),
        freq='10min', tz='UTC'  # GOES MCMIPC native cadence
    )
    missing = full_range.difference(combined['datetime_utc'])
    print(f"  Remaining missing timestamps in expected range: {len(missing)}")
    if len(missing) > 0 and len(missing) < 20:
        print(f"    {list(missing)}")

    # Overwrite the original file with the merged, gap-filled version
    backup_path = ORIGINAL_S1_CSV.with_suffix('.csv.bak')
    ORIGINAL_S1_CSV.rename(backup_path)
    print(f"\n  Original backed up to: {backup_path}")

    combined.to_csv(ORIGINAL_S1_CSV, index=False)
    print(f"  ✓ Merged file written to: {ORIGINAL_S1_CSV}")
    print(f"\nNext: re-run src/c13_features.py to regenerate parquets with the gap closed.")