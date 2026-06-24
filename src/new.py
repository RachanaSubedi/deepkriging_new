import pandas as pd

bg_csi = pd.read_parquet(r"C:\Users\C838122727\Documents\CSU\research\deepkriging_solar_Copy\data\processed\background_field\bg_csi_stations.parquet")  # adjust path if different
s1 = bg_csi['S1']

worst_times = [
    "2024-05-05 08:00", "2024-05-04 08:00", "2024-04-04 09:00",
    "2024-04-05 08:30", "2024-12-03 10:00", "2024-12-04 10:00",
    "2024-04-04 08:30", "2024-12-05 10:00", "2024-05-05 07:30",
    "2024-05-04 07:30",
]
for t in worst_times:
    ts = pd.Timestamp(t).tz_localize("America/Los_Angeles")
    if ts in s1.index:
        print(f"{t}  bg_csi={s1.loc[ts]:.3f}")
    else:
        print(f"{t}  still not found — index may use a different exact minute offset")