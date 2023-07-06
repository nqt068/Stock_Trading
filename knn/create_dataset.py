import logging
import os

import numpy as np
import pandas as pd
import parameters as pr
import yfinance
from torch.utils.data import DataLoader
from utils.dataset import UpDownStockDataset
from utils.helper import *


def create(*, code: str = pr.code,
           start_day: str = pr.start_day,
           end_day: str = pr.end_day,
           cols: list[str] = pr.cols,
           prediction_step: int = pr.prediction_step,
           profit_rate: float = pr.profit_rate,
           use_median: bool = pr.use_median,
           seq_length: int = pr.seq_length,
           transform = pr.transform,
           split_func = pr.split_func,
           split_ratio: list[int] = pr.split_ratio,
           batch_size: int = pr.batch_size):
    logger = logging.getLogger("CreateDataset")
    data_folder = "./data"
    data_path = os.path.join(data_folder, f"{code}.csv")
    os.makedirs(data_folder, exist_ok=True)
    if not os.path.isfile(data_path):
        logger.info(f"Downloading {code} data from yfinance...")
        dat = yfinance.download(code, start=start_day, end=end_day, progress=False)
        dat.to_csv(data_path)
    else:
        logger.info(f"Loading data from {data_path}")
        dat = pd.read_csv(data_path)
    data = np.array(dat[cols])
    data = data.flatten() if data.shape[-1] == 1 else data
    logger.info(f"Data downloaded.")

    profit = [(data[idx+prediction_step]/data[idx]) - 1 for idx in range(len(data) - prediction_step)]
    profit_rate = np.median(profit) if use_median else profit_rate
    logger.info(f"Median of profit rate: {profit_rate}")

    logger.info(f"Create dataset object with parameters: {dict(seq_length=seq_length, trainsforms=transform.__name__ if transform is not None else None, prediction_step=prediction_step, profit_rate=profit_rate)}")
    dataset = UpDownStockDataset(data=data,
                                 seq_length=seq_length,
                                 transforms=transform,
                                 prediction_step=prediction_step,
                                 profit_rate = float(profit_rate))
    logger.info("Done")

    logger.info(f"Create Subset and DataLoader objects with {split_func.__name__} using split_ratio = {split_ratio} and batch_size = {batch_size}")
    train_set, val_set = split_func(dataset, split_ratio)


    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  drop_last=False,
                                  shuffle=False)
    val_dataloader = DataLoader(dataset=val_set,
                                batch_size=batch_size,
                                drop_last=False,
                                shuffle=True)
    logger.info("Done.")

    logger.info(f"Number of training samples: {len(train_dataloader) * batch_size}")
    logger.info(f"Number of validating samples: {len(val_dataloader) * batch_size}")
    for x, y in train_dataloader:
        logger.info(f"X's shape: {x.shape}")
        logger.info(f"Y's shape: {y.shape}")
        break


    logger.info("Creating an array of data")
    array_trainX, array_trainY = [], []
    for val in train_dataloader:
        array_trainX.extend(list(val[0].numpy()))
        array_trainY.extend(list(val[1].numpy()))
    array_trainX, array_trainY = np.array(array_trainX), np.array(array_trainY)
    logger.info("Done.")
    return train_dataloader, val_dataloader, array_trainX, array_trainY

train_dataloader, val_dataloader, array_trainX, array_trainY = create()