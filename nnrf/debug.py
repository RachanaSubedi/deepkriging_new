import pandas as pd
m = pd.read_csv('outputs/nnrf_dk/nnrf_metrics.csv')
print(m['gof'].describe())
print('\nWorst 5 PVs:')
print(m.sort_values('gof').head(5)[['pv_name','rmse','gof']])
print('\nBest 5 PVs:')
print(m.sort_values('gof', ascending=False).head(5)[['pv_name','rmse','gof']])