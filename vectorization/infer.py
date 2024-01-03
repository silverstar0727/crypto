from torch.utils.data import Subset, DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch

import datetime
from tqdm import tqdm

import model


def load_model(model_path):
    net = model.Time2Vec(
        input_size=9,
        activation="sin",
        hidden_dim=200,
        out_dim=1,
        batch_size=32,
        lstm_hidden_dim=20,
        lstm_layer=1
    )

    ckpt = torch.load(model_path)
    w = {k[4:]: v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}

    net.load_state_dict(w)
    net.eval()

    return net.tsv

def prep_data(data_path):
    df = pd.read_csv(data_path)

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

    return val


class CryptoDataset(Dataset):
    def __init__(self, df, mode="train", window_size=30) -> None:
        super().__init__()
        self.df = df
        self.window_size = window_size

    def __len__(self):
        return len(self.df) - self.window_size - 2

    def __getitem__(self, index):
        x = self.df.iloc[index: index + self.window_size].values
        target_cols = ["close"]
        y = self.df.iloc[index + self.window_size][target_cols].values

        x = torch.Tensor(x)
        y = torch.Tensor(y)
        return x, y


def main(model_path, data_path, crypto_pinecone):
    tsv = load_model(model_path=model_path)
    data_df = prep_data(data_path=data_path)
    dataset = CryptoDataset(data_df, window_size=30)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    tsv_preds = []
    tsv_labels = []
    with tqdm(dataloader) as iterator:
        for batch_idx, (x, y) in enumerate(iterator):
            preds = tsv(x).unbind(dim=0)

            for data_idx, pred in enumerate(preds):
                pred = pred.flatten()

                upsert_response = crypto_pinecone.upsert(
                    vectors=[(f"{batch_idx}_{data_idx}", pred.tolist())]
                )
            


if __name__ == "__main__":
    import pinecone

    pinecone.init(api_key="9882a900-ced3-4f2c-a5a5-bea615c330f3", environment="asia-southeast1-gcp-free")
    crypto_pinecone = pinecone.Index("crypto")

    main(
        model_path="best.ckpt",
        data_path="../train_model/data/bitcoin_2017_to_2023.csv",
        crypto_pinecone=crypto_pinecone
    )