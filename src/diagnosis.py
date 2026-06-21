"""
Check raw vs corrected GHI on April 22 to isolate whether the
afternoon overshoot comes from the model or the quantile correction.

Run:
    python src/diagnose_apr22.py
"""
import pandas as pd
import numpy as np

ghi_raw  = pd.read_parquet(r"C:\Users\C838122727\Documents\CSU\research\deepkriging_solar_Copy\outputs\predictions\ghi_pvs.parquet")
ghi_corr = pd.read_parquet(r"C:\Users\C838122727\Documents\CSU\research\deepkriging_solar_Copy\outputs\predictions\ghi_pvs_corrected.parquet")

for d in [ghi_raw, ghi_corr]:
    d.index = pd.to_datetime(d.index, utc=True).tz_convert('America/Los_Angeles')

day = pd.Timestamp('2024-04-22').date()
raw_day  = ghi_raw[ghi_raw.index.date == day]
corr_day = ghi_corr[ghi_corr.index.date == day]

print(f"{'time':>6}  {'raw_median':>11}  {'corr_median':>12}  {'cf_applied':>10}")
for t in raw_day.between_time('11:00', '16:00').index:
    r = raw_day.loc[t].median()
    c = corr_day.loc[t].median()
    cf = c / r if r > 0 else np.nan
    print(f"{t.strftime('%H:%M'):>6}  {r:>11.1f}  {c:>12.1f}  {cf:>10.3f}")

cf_table = pd.read_csv(r"outputs\predictions\correction_factors.csv")
print("\nCorrection factors for April (month=4), hours 11-16:")
print(cf_table[(cf_table['month']==4) & (cf_table['hour'].between(11,16))].to_string(index=False))