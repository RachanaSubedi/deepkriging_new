"""
check_p2_outage.py

Determine the FULL extent of the P2 sensor outage found around
Nov 13-15, 2024 (the earlier diagnostic only caught rows where
clearsky > 300 W/m², which could miss the edges of the outage).
Also scans the full year for any OTHER similar outages.

Run from repo root:
    python check_p2_outage.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from configs.config import BG_DIR, STATION_DIR

ghi = pd.read_csv(STATION_DIR / "all_stations_GHI_30min_PST_filled.csv",
                   sep=None, engine='python', encoding='utf-8-sig')
ghi = ghi.dropna(subset=['datetime'])
ghi['datetime'] = pd.to_datetime(ghi['datetime'], format='mixed')
ghi = ghi.set_index('datetime')
ghi.index = ghi.index.tz_localize('Etc/GMT+8').tz_convert('America/Los_Angeles')
ghi.columns = [c.replace('GHI_', '') for c in ghi.columns]
ghi = ghi[~ghi.index.duplicated(keep='first')]

clearsky = pd.read_parquet(BG_DIR / "clearsky_pvlib_stations.parquet")
common = ghi.index.intersection(clearsky.index)
ghi_c = ghi.loc[common]
cs_c  = clearsky.loc[common]

# ── Part 1: full extent of the known Nov 13-15 outage ──────────
window = ghi_c.loc['2024-11-12':'2024-11-16', 'P2']
cs_window = cs_c.loc['2024-11-12':'2024-11-16', 'P2']

print("=== P2 GHI vs clearsky, Nov 12-16, all daytime hours (clearsky > 5) ===")
daytime = cs_window > 5
combined = pd.DataFrame({'P2_GHI': window[daytime], 'P2_clearsky': cs_window[daytime]})
combined['ratio'] = combined['P2_GHI'] / combined['P2_clearsky']
print(combined.to_string())

print()
print("=== Flagging suspiciously low P2/clearsky ratio (< 0.05) on Nov 12-16 ===")
suspicious = combined[combined['ratio'] < 0.05]
print(suspicious.to_string())

# ── Part 2: full-year scan for ANY other similar outages ──────
print()
print("=== Full-year scan: stretches of 3+ consecutive zero-GHI readings")
print("    where clearsky > 200 (i.e. should clearly be daytime/sunny) ===")
full = pd.DataFrame({'P2_GHI': ghi_c['P2'], 'P2_clearsky': cs_c['P2']})
full['is_suspect'] = (full['P2_GHI'] == 0) & (full['P2_clearsky'] > 200)
full['block'] = (full['is_suspect'] != full['is_suspect'].shift()).cumsum()
runs = full[full['is_suspect']].groupby('block').agg(
    start=('P2_GHI', lambda x: x.index[0]),
    end=('P2_GHI', lambda x: x.index[-1]),
    n=('P2_GHI', 'size')
)
runs = runs[runs['n'] >= 3]
print(f"Found {len(runs)} suspect run(s) of 3+ consecutive zero-GHI/high-clearsky readings:")
print(runs.to_string())