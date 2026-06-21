import pandas as pd

df = pd.read_parquet(
    r"C:\Users\C838122727\Documents\CSU\research\deepkriging_solar\outputs\predictions\ghi_pvs.parquet",
    engine='pyarrow'
)

# Convert index to PDT and format like station file
df.index = pd.to_datetime(df.index, utc=True).tz_convert('America/Los_Angeles')
df.index = df.index.strftime('%m/%d/%Y %H:%M')
df.index.name = 'datetime'

df.to_csv(
    r"C:\Users\C838122727\Documents\CSU\research\deepkriging_solar\outputs\predictions\ghi_pvs.csv",
    index=True
)
print(f"Saved: {df.shape}  first datetime: {df.index[0]}")