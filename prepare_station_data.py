# ════════════════════════════════════════════════════════════
# prepare_station_data.py
#
# Harmonizes all 4 stations into identical Deep Kriging input format.
#   S1, S2, S3 : raw 5-min Ambient Weather  → 30-min averaged
#   P2         : imputed model output (already 30-min)
#
# Output: one CSV per station, each [datetime, GHI], 30-min PST grid,
#         plus one combined wide table.
#
# Timezone: PST (UTC-8, local standard, no DST) — matches NSRDB.
#
# ── HOW TO RUN ───────────────────────────────────────────────
#   Place this file in:  deepkriging_solar/
#   Your raw files are in: deepkriging_solar/data/raw/stations/
#   From PyCharm, just run this file (right-click → Run), OR in terminal:
#       cd deepkriging_solar
#       python prepare_station_data.py
#   Outputs land in: deepkriging_solar/data/processed/stations/
# ════════════════════════════════════════════════════════════

import os
import numpy as np
import pandas as pd

# ── Resolve paths relative to THIS script's location ─────────
# So it works no matter what directory you launch it from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR    = os.path.join(SCRIPT_DIR, "data", "raw", "stations")
OUT_DIR    = os.path.join(SCRIPT_DIR, "data", "processed", "stations")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Canonical 30-min PST grid (full year 2024) ───────────────
GRID = pd.date_range("2024-01-01 00:00:00", "2024-12-31 23:30:00", freq="30min")

# ── Station registry ─────────────────────────────────────────
# kind="raw"     → 5-min Ambient Weather, needs averaging
# kind="imputed" → 30-min model output, just reformat
#
# NOTE: filenames use the exact names in your folder (with spaces).
STATIONS = [
    {"name": "S1", "kind": "raw",
     "file": "46.59, -119.15 2024.csv", "lat": 46.594029, "lon": -119.152367},
    {"name": "S2", "kind": "raw",
     "file": "46.82, -119.15 2024.csv", "lat": 46.823242, "lon": -119.163197},
    {"name": "S3", "kind": "raw",
     "file": "46.82, -119.16 2024.csv", "lat": 46.821036, "lon": -119.150761},
    {"name": "P2", "kind": "imputed",
     "file": "46.78, -119.22 2024.csv", "lat": 46.780547, "lon": -119.228783},
]

# Column names
RAW_DATE_COL = "Date"                     # ISO 8601 with -08:00 (PST)
RAW_GHI_COL  = "Solar Radiation (W/m^2)"
IMP_DATE_COL = "datetime"                 # naive PST
IMP_GHI_COL  = "GHI_imputed"


def process_raw(path, name):
    """Raw 5-min Ambient Weather → 30-min averaged GHI on PST grid."""
    df = pd.read_csv(path)

    # Parse ISO timestamp (contains -08:00). utc=True → correct UTC,
    # then subtract 8h → fixed PST, then drop tz to get naive PST.
    df["dt_utc"]   = pd.to_datetime(df[RAW_DATE_COL], utc=True)
    df["datetime"] = (df["dt_utc"] - pd.Timedelta(hours=8)).dt.tz_localize(None)
    df["GHI"]      = pd.to_numeric(df[RAW_GHI_COL], errors="coerce")

    df = (df[["datetime", "GHI"]]
          .sort_values("datetime")
          .set_index("datetime"))

    # 5-min → 30-min mean (averages the 6 readings in each half hour)
    df30 = df["GHI"].resample("30min").mean()

    # Snap onto canonical grid so every station shares identical timestamps
    df30 = (df30.reindex(GRID)
                 .rename_axis("datetime")
                 .reset_index(name="GHI"))

    n_missing = df30["GHI"].isna().sum()
    df30["GHI"] = df30["GHI"].interpolate(limit=2).clip(lower=0)
    print(f"  [{name}] raw 5-min → 30-min | rows={len(df30):,} | "
          f"gaps_before_interp={n_missing}")
    return df30


def process_imputed(path, name):
    """Imputed 30-min model output → [datetime, GHI] on PST grid."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df[IMP_DATE_COL])
    df["GHI"]      = pd.to_numeric(df[IMP_GHI_COL], errors="coerce")

    df = (df[["datetime", "GHI"]]
          .sort_values("datetime")
          .set_index("datetime")["GHI"]
          .reindex(GRID)
          .rename_axis("datetime")
          .reset_index(name="GHI"))

    n_missing = df["GHI"].isna().sum()
    df["GHI"] = df["GHI"].interpolate(limit=2).clip(lower=0)
    print(f"  [{name}] imputed 30-min | rows={len(df):,} | "
          f"gaps_on_grid={n_missing}")
    return df


# ── Run ──────────────────────────────────────────────────────
print("=" * 60)
print("PREPARING STATION DATA FOR DEEP KRIGING")
print(f"Raw dir : {RAW_DIR}")
print(f"Out dir : {OUT_DIR}")
print("=" * 60 + "\n")

all_dfs = {}
for st in STATIONS:
    path = os.path.join(RAW_DIR, st["file"])
    if not os.path.exists(path):
        print(f"  [{st['name']}] ⚠️  FILE NOT FOUND: {path}")
        continue
    print(f"[{st['name']}] {st['file']}")
    if st["kind"] == "raw":
        df = process_raw(path, st["name"])
    else:
        df = process_imputed(path, st["name"])

    out_path = os.path.join(OUT_DIR, f"{st['name']}_GHI_30min_PST.csv")
    df_out = df.copy()
    df_out["GHI"] = df_out["GHI"].round(2)
    df_out.to_csv(out_path, index=False)
    print(f"        saved → {out_path}\n")
    all_dfs[st["name"]] = df

# ── Combined wide table ──────────────────────────────────────
print("Building combined wide table...")
combined = pd.DataFrame({"datetime": GRID})
for name, df in all_dfs.items():
    combined = combined.merge(
        df.rename(columns={"GHI": f"GHI_{name}"}),
        on="datetime", how="left")
combined_path = os.path.join(OUT_DIR, "all_stations_GHI_30min_PST.csv")
for c in combined.columns:
    if c != "datetime":
        combined[c] = combined[c].round(2)
combined.to_csv(combined_path, index=False)
print(f"  saved → {combined_path}\n")

# ── Alignment check ──────────────────────────────────────────
print("=" * 60)
print("ALIGNMENT CHECK — June 17 daylight window per station")
print("=" * 60)
for name, df in all_dfs.items():
    jun = df[(df["datetime"].dt.month == 6) & (df["datetime"].dt.day == 17)]
    day = jun[jun["GHI"] > 5]
    if len(day) > 0:
        peak = jun.loc[jun["GHI"].idxmax(), "datetime"].strftime("%H:%M")
        first = day["datetime"].iloc[0].strftime("%H:%M")
        last  = day["datetime"].iloc[-1].strftime("%H:%M")
        print(f"  {name}: daylight {first}-{last} PST | peak {peak} | "
              f"max {jun['GHI'].max():.0f} W/m²")

# ── Annual summary ───────────────────────────────────────────
print("\n" + "=" * 60)
print("ANNUAL MEAN DAYTIME GHI (sanity)")
print("=" * 60)
for name, df in all_dfs.items():
    dayvals = df[df["GHI"] > 5]["GHI"]
    print(f"  {name}: mean={dayvals.mean():.1f} W/m²  max={df['GHI'].max():.0f} W/m²")

print(f"\nAll stations on identical {len(GRID):,}-row 30-min PST grid.")
print("Ready for Deep Kriging.")