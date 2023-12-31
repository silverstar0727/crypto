from typing import Union

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Subset, DataLoader, Dataset
import pandas as pd
import datetime
import torch

import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from transforms.base import BaseTransforms

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


def preprocessing_df(df, val_split): 
    # drop null values
    df = df.dropna()
    df = df.reset_index(drop=True)

    # apply strptime
    df["timestamp"] = df["timestamp"].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    )

    df = df.reset_index(drop=True)
    df = df.drop(columns=["timestamp"])

    df_len = len(df)
    cut_idx = int(df_len*val_split)
    train, val = df.iloc[cut_idx:], df.iloc[:cut_idx]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train = train.copy()
    train[train.columns] = scaler.fit_transform(train[train.columns])
    val = val.copy()
    val[val.columns] = scaler.transform(val[val.columns])

    return train, val

class CryptoDataModule(L.LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        batch_size: int,
        val_split: Union[int, float],
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.val_split = val_split
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        self.df = pd.read_csv(self.csv_path)
        self.train_df, self.val_df = preprocessing_df(df=self.df, val_split=self.val_split)

    def setup(self, stage: str) -> None:
        self.train_dataset = CryptoDataset(self.train_df, mode="train")
        self.val_dataset = CryptoDataset(self.val_df, mode="val")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    # def test_dataloader(self) -> EVAL_DATALOADERS:
    #     return DataLoader(
    #         self.test_dataset, batch_size=100, shuffle=False, num_workers=4
    #     )
