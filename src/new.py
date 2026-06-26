import pandas as pd

st = pd.read_csv(r"C:\Users\C838122727\Documents\CSU\research\deepkriging_solar_Copy\data\raw\stations\all_stations_GHI_30min_PST_filled.csv",
                  sep=None, engine='python', encoding='utf-8-sig', index_col=0, parse_dates=True)
st.index = (pd.to_datetime(st.index)
            .tz_localize('Etc/GMT+8')
            .tz_convert('America/Los_Angeles')
            .tz_localize(None))
st.columns = [c.replace('GHI_', '') for c in st.columns]

print("st.index length:", len(st.index))
print("st.index unique:", st.index.nunique())
print("Duplicated timestamps in st.index:")
print(st.index[st.index.duplicated(keep=False)])