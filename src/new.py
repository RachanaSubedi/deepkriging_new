import pandas as pd
df = pd.read_csv(r'C:\Users\C838122727\Documents\CSU\research\deepkriging\deepkriging_solar_Copy\outputs\validation\baseline_comparison_overall.csv')
pivot = df[df['method'].isin(['idw_stations','deepkriging_corrected'])].pivot(
    index='station', columns='method', values=['rmse','mae','r2'])
print(pivot.round(3).to_string())