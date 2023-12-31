import pandas as pd

import datetime
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("crypto/data/bitcoin_2017_to_2023.csv")

print("total data", len(df))
df = df.dropna()
df = df.reset_index(drop=True)
print("after dropna", len(df))

df["timestamp"] = df["timestamp"].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

# prev = None
# discrete_ids = []
# for idx, date in enumerate(df["timestamp"]):
#     if prev is None:
#         prev = date
#         continue

#     if (prev - date) != datetime.timedelta(minutes=1):
#         discrete_ids.append(idx)

#     prev = date


# print("discrete datas", len(discrete_ids))

# df = df.drop(index=discrete_ids)
df = df.drop(columns=["timestamp"])
df = df.reset_index(drop=True)
print("after drop", len(df))

df_len = len(df)
test_size = 0.2
cut_idx = int(df_len*test_size)
train, val = df.iloc[cut_idx:], df.iloc[:cut_idx]
print("train data", len(train))
print("test data", len(val))

scaler = MinMaxScaler(feature_range=(-1, 1))
train = train.copy()
train[train.columns] = scaler.fit_transform(train[train.columns])
val = val.copy()
val[val.columns] = scaler.transform(val[val.columns])

print(train)
print(val)

window_size = 30
for index in range(len(train)-2):
    x = train.iloc[index: index + window_size].values
    y = train.iloc[index + window_size][["open", "high", "low", "close"]].values
