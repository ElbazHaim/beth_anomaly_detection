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
from torch import nn
from torch.nn import functional as F
from torchmetrics import AUROC

from datamodules import BETHDataModule
from models.ann import ANN
from models.vae import VAE

tracking_uri = "http://127.0.0.1:5000/"
experiment_name = "beth_anomaly_detection"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name=experiment_name)
mlflow.autolog()
pl_logger = MLFlowLogger(experiment_name, tracking_uri=tracking_uri)

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

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class VAE(pl.LightningModule):
    def __init__(
        self,
        input_height=40,
        latent_dim=128,
        enc_out_dim=40,
    ):
        super().__init__()
        self.input_height = input_height
        self.latent_dim = latent_dim
        self.enc_out_dim = enc_out_dim

        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(self.input_height, self.latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.latent_dim, self.enc_out_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.latent_dim, self.input_height),
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=-1)

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(std)
        )
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):

        x, _ = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)

        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = kl - recon_loss
        elbo = elbo.mean()

        self.log_dict(
            {
                "elbo": elbo,
                "kl": kl.mean(),
                "recon_loss": recon_loss.mean(),
                "reconstruction": recon_loss.mean(),
                "kl": kl.mean(),
            }
        )

        return elbo


def objective(trial):
    latent_dim_choices = [2**i for i in range(7, 11)]
    latent_dim = trial.suggest_categorical("latent_dim", latent_dim_choices)
    batch_size_choices = [2**i for i in range(11, 14)]
    batch_size = trial.suggest_categorical("batch_size", batch_size_choices)

    datamodule = BETHDataModule(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=batch_size,
    )
    model = VAE(input_height=40, latent_dim=latent_dim, enc_out_dim=40)
    trainer = pl.Trainer(
        logger=MLFlowLogger(experiment_name, tracking_uri=tracking_uri),
        fast_dev_run=False,
        max_epochs=50,
        accelerator="gpu",
        callbacks=[
            EarlyStopping(
                monitor="elbo",
                min_delta=0.00001,
                patience=10,
                verbose=True,
                mode="min",
            )
        ],
        devices=1,
    )
    hyperparameters = dict(
        batch_size=batch_size,
        enc_out_dim=40,
        latent_dim=latent_dim,
        input_height=40,
    )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    val_loss = trainer.callback_metrics["elbo"].item()
    return val_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5, timeout=60 * 60 * 8)

print("Best trial:")
trial = study.best_trial
print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")
