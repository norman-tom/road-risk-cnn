import gc
import torch
import numpy as np
from src.models.vgg16 import VGG16
from src.models.resnet18 import RESNET18
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
import os
from pathlib import Path
from ray import tune
from src.custom_types import Dataset
from torch.nn import CrossEntropyLoss
from torcheval.metrics.functional import multiclass_accuracy

import re

ROOT_DIR = Path(os.path.abspath(__file__)).parents[2]
RAY_DIR = os.path.join(ROOT_DIR, "ray")

N_SAMPLES = 10000
BATCH_SIZE = 500
DEVICE = "cuda"
N_CLASSES = 3

# TODO Make this script automatic. Remove all hardcoded elements. '

class Reporter():
    """Reporter takes a metric that we want to find the best model for.
    All the reporting will be relevant to this model. So since we tuned on acc_val_macro, 
    if we are saying that the best model is the one with the highest acc_val_macro then pass
    that metric to the constructor."""
    _generator = np.random.default_rng(seed=42)
        
    def __init__(self, architecture, criteria) -> None:
        assert criteria in ["loss_tr", "loss_val", "acc_val_macro", "acc_val_micro"]
        results = tune.ExperimentAnalysis(os.path.join(RAY_DIR, "sydney", architecture.__name__))
        mode = "min" if criteria == "loss_tr" or criteria == "loss_val" else "max"
        trial = results.get_best_trial(metric=criteria, 
                                        mode=mode, 
                                        scope="all")
        checkpoint = results.get_best_checkpoint(trial, metric=criteria, mode=mode)
        self._checkpoint = checkpoint.path[-4:]
        model_state, _ = torch.load(os.path.join(checkpoint.path, "checkpoint"))
        self._features = self._feature_label(checkpoint)
        self._model = architecture(self._n_channels(self._features), N_CLASSES)
        self._model.load_state_dict(model_state)
        self._dataset = None
        self._model.to(DEVICE)

    @property
    def features(self):
        return self._features
    
    def confusion(self) -> np.ndarray:
        idx = self._generator.choice(range(self._dataset.X.size()[0]), N_SAMPLES)
        return confusion_matrix(
            self._dataset.Y[idx], 
            self._predictions(self._model, self._dataset.X[idx]).numpy()
        )

    def accuracy(self, average) -> np.ndarray:
        return multiclass_accuracy(
            self._predictions(self._model, self._dataset.X),
            self._dataset.Y,
            average=average, 
            num_classes=N_CLASSES
        )
    
    def recall(self) -> np.ndarray:
        return recall_score(
            self._dataset.Y, 
            self._predictions(self._model, self._dataset.X).numpy(),
            average="micro"
        )
    
    def precision(self) -> np.ndarray:
        return precision_score(
            self._dataset.Y, 
            self._predictions(self._model, self._dataset.X).numpy(),
            average="micro"
        )

    def report(self) -> np.ndarray:
        idx = self._generator.choice(range(self._dataset.X.size()[0]), N_SAMPLES)
        return classification_report(
            self._dataset.Y[idx], 
            self._predictions(self._model, self._dataset.X[idx]).numpy()
        )
    
    def loss(self):
        weights = torch.tensor([0.42, 2.39, 4.36])
        criterion = CrossEntropyLoss(weights)
        #idx = self._generator.choice(range(self._dataset.X.size()[0]), N_SAMPLES)
        loss = criterion(self._predictions(self._model, self._dataset.X, classes=False), self._dataset.Y)
        return loss

    def load_dataset(self, dataset: Dataset):
        self._dataset = dataset

    def _predictions(self, model, X, classes=True):
        h = 0
        t = BATCH_SIZE
        L = X.size()[0] 
        N = int(L / BATCH_SIZE) * BATCH_SIZE # Discard the remainder examples. 
        out = torch.zeros((L, N_CLASSES))
        with torch.no_grad():
            for i in range(N // BATCH_SIZE):
                X_batch = X[h:t].to(DEVICE)
                out[h:t] = model(X_batch)
                h = t
                t += BATCH_SIZE
            # Pick up the remainder examples. 
            X_batch = X[h:L].to(DEVICE)
            out[h:L] = model(X_batch)
            return torch.argmax(out, 1) if classes else out
        
    def _feature_label(self, checkpoint):
        pattern = r"features=[gslh]+_"
        features = re.search(pattern=pattern, string=checkpoint.path)
        features = features.group(0)[9:-1]
        return features
    
    def _n_channels(self, features):
        if features == "g":
            return 1
        elif features == "gs":
            return 2
        elif features == "gh":
            return 6
        elif features == "gl":
            return 2
        elif features == "gsh":
            return 7
        elif features == "gsl":
            return 3
        elif features == "glh":
            return 7
        elif features == "gslh":
            return 8
        else:
            raise ValueError()
