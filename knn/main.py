import logging
logging.basicConfig(level=logging.INFO)
import os

import numpy as np
import parameters as pr
import torch
from create_dataset import array_trainX, array_trainY, val_dataloader, create
from models.knn import WeightedKNearestNeighbors
from utils.helper import get_file_name, metric

weights = 0
knn = WeightedKNearestNeighbors(x=array_trainX,
                                y=array_trainY,
                                k=1000,
                                similarity='cosine',
                                weights=weights,
                                learning_rate=10**-1,
                                device=pr.device,
                                train_split_ratio=pr.wknn_train_split_ratio)
def report():
    pred = []
    logits = []
    targ = []
    for (x, y) in val_dataloader:
        prediction = knn.predict(x, reduction="score")
        pred.extend(prediction[0])
        logits.extend(prediction[1])
        targ.extend(y.tolist())
    pred = torch.tensor(pred)
    logits = torch.tensor(logits)
    targ = torch.tensor(targ)

    confusion_matrix = np.zeros((2, 2), int)
    for idx, (x, y) in enumerate(zip(logits, targ)):
        confusion_matrix[x, y] += 1
    print("*"*os.get_terminal_size().columns)
    metric(confusion_matrix, verbose=True)
    print(confusion_matrix)

    limit_n = [5, 10, 25, 100]
    res = []
    for val in limit_n:
        lim_score = np.percentile(pred, 100-val)
        confusion_matrix = np.zeros((2, 2), int)
        for idx, (x, y) in enumerate(zip(logits, targ)):
            if pred[idx] < lim_score:
                continue
            confusion_matrix[x, y] += 1
        res.extend(metric(confusion_matrix, verbose=False)[:-1])
    print(*[f"{val:.3f}" for val in res])

report()
knn.train(100, 10)
report()