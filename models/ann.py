
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchmetrics import AUROC


class ANN(LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        
        self.metric = AUROC("binary")
        self.loss_fn = nn.BCELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def _step(self, batch, batch_idx):
        inputs, labels = batch
        y_hat = self(inputs)
        loss = self.loss_fn(y_hat, labels.unsqueeze(1))
        auroc = self.metric(y_hat, labels)
        return loss, auroc

    def training_step(self, batch, batch_idx):
        train_loss, train_auroc = self._step(batch, batch_idx)
        self.log("train_loss", train_loss, on_epoch=True)
        self.log("train_auroc", train_auroc, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss, val_auroc = self._step(batch, batch_idx)
        self.log("val_loss", val_loss, on_epoch=True)
        self.log("val_auroc", val_auroc, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss, test_auroc = self._step(batch, batch_idx)
        self.log("test_loss", test_loss, on_epoch=True)
        self.log("test_auroc", test_auroc, on_epoch=True)
        return test_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
