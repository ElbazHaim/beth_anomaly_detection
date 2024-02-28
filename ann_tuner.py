import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import nltk
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torchmetrics import AUROC

from datamodules import BETHDataModule
from models.ann import ANN

tracking_uri = "http://127.0.0.1:5000/"
experiment_name = "beth_anomaly_detection"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name=experiment_name)
mlflow.autolog()

input_directory = Path("data/np_arrays")

X_train = np.load(input_directory / "X_train.npy", allow_pickle=True).astype(
    np.float32
)
y_train = np.load(input_directory / "y_train.npy", allow_pickle=True).astype(
    np.float32
)
X_column_names = np.load(
    input_directory / "X_column_names.npy", allow_pickle=True
)

X_val = np.load(input_directory / "X_val.npy", allow_pickle=True).astype(
    np.float32
)
y_val = np.load(input_directory / "y_val.npy", allow_pickle=True).astype(
    np.float32
)

X_test = np.load(input_directory / "X_test.npy", allow_pickle=True).astype(
    np.float32
)
y_test = np.load(input_directory / "y_test.npy", allow_pickle=True).astype(
    np.float32
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


def objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 216])
    batch_size = trial.suggest_categorical(
        "batch_size", [64, 128, 216, 512, 1024]
    )
    learning_rate = trial.suggest_loguniform("learning_rate", 3e-5, 3e-3)
    model = ANN(40, hidden_dim, 1, learning_rate=learning_rate)

    pl_logger = MLFlowLogger(experiment_name, tracking_uri=tracking_uri)
    datamodule = BETHDataModule(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=batch_size,
    )
    trainer = pl.Trainer(
        logger=pl_logger,
        fast_dev_run=False,
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.00001,
                patience=5,
                verbose=True,
                mode="min",
            )
        ],
    )
    hyperparameters = dict(
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    val_loss = trainer.callback_metrics["val_loss"].item()
    return val_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, timeout=60 * 60 * 8)

print("Best trial:")
trial = study.best_trial
print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")
