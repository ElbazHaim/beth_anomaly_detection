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
from sklearn.linear_model import SGDOneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torch import nn
from torch.nn import functional as F
from torchmetrics import AUROC

from datamodules import BETHDataModule
from models.vae import VAE
