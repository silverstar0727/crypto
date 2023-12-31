import torch

import model

net = model.Time2Vec(
    input_size=9,
    activation="sin",
    hidden_dim=200,
    out_dim=4,
    batch_size=32,
    lstm_hidden_dim=20,
    lstm_layer=1
)

ckpt = torch.load("best.ckpt")
w = {k[4:]: v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}

net.load_state_dict(w)
net.eval()

import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("../train_model/data/bitcoin_2017_to_2023.csv")

df = df.dropna()
df = df.reset_index(drop=True)
df["timestamp"] = df["timestamp"].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

df = df.drop(columns=["timestamp"])
df = df.reset_index(drop=True)

df_len = len(df)
test_size = 0.2
cut_idx = int(df_len*test_size)
train, val = df.iloc[cut_idx:], df.iloc[:cut_idx]

scaler = MinMaxScaler(feature_range=(-1, 1))
train = train.copy()
train[train.columns] = scaler.fit_transform(train[train.columns])
val = val.copy()
val[val.columns] = scaler.transform(val[val.columns])


window_size = 30
for index in range(len(val) - window_size - 2):
    x = val.iloc[index: index + window_size].values
    y = val.iloc[index + window_size][["open", "high", "low", "close"]].values

    x = torch.Tensor(x).unsqueeze(dim=0)
    y = torch.Tensor(y).unsqueeze(dim=0)

    pred = net(x)
    print(pred, y)