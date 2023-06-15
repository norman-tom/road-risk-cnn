import os
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import dataset
from src.custom_types import Grid
import numpy as np


class Dataset(dataset.Dataset):
    def __init__(
        self, label: str, 
        gridX: Grid, X: np.ndarray, 
        gridY: Grid = None, Y: np.ndarray = None, 
        mean = None,
        std = None
    ):
        self._label = label
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.long)
        self._X = X
        self._Y = Y
        self._gridX = gridX
        self._gridY = gridY
        self._mean = mean
        self._std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self._Y is None:
            return self._X[idx]
        else:
            return self._X[idx], self._Y[idx]

    @property
    def label(self):
        return self._label

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def gridX(self):
        return self._gridX

    @property
    def gridY(self):
        return self._gridY
    
    @property
    def W(self):
        return self._W
    
    @property
    def mean(self):
        return self._mean
    
    @property
    def std(self):
        return self._std

    def split_dataset(self, seed=0):
        raise NotImplementedError()
    
    def weights(self, W_0: float = 1, mulitplier: int = 1, func="log"):
        """Default behaviour is a loss weight matrix of 1's.
        The weight of the first class (W_0) and the scaling of the weights (multiplier) 
        are hyperparameters of the model.
        """
        classes = np.unique(self.Y)
        if func == "log":
            W = log(W_0, mulitplier, classes)   # this calculates a geometric series for the class weights. 
        else:
            W = geometric(W_0, mulitplier, classes) # This calculates log scaling for class weights.
        return np.array(W, dtype=np.float32)

# Use these for loss weights, either geometric or log scaling.     
def geometric(W0, multi, classes):
    return [multi ** i * W0 for i in classes]

def log(W0, multi, classes):
    return [multi * np.log(i+1) + W0 for i in classes]