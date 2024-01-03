import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("train_model/data/bitcoin_2017_to_2023.csv")

df = df.dropna()
df = df.reset_index(drop=True)

df["timestamp"] = df["timestamp"].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

df = df.sort_values(by=["timestamp"])
# df = df.drop(columns=["timestamp"])
df = df.reset_index(drop=True)

df_len = len(df)
test_size = 0.2
cut_idx = int(df_len* (1 - test_size))
train, val = df.iloc[:cut_idx], df.iloc[cut_idx:]

numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
       'number_of_trades', 'taker_buy_base_asset_volume',
       'taker_buy_quote_asset_volume']



from tslearn.clustering import TimeSeriesKMeans
import time

start = time.time()

data = train[numeric_cols].values

model = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=50, random_state=42)
model.fit(data)
model.to_hdf5('./trained_model.hdf5')

print(time.time() - start)