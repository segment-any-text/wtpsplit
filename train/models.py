import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils import data
from text_data import MemoryMapDataset
from dataset import SplitDataset


def _freeze_bias(lstm):
    for name, param in lstm.named_parameters():
        if name.startswith("bias"):
            param.requires_grad = False
            param[:] = 0


class Network(pl.LightningModule):
    def __init__(self, hparams={}):
        super().__init__()
        # init datasets
        text_data = MemoryMapDataset("texts.txt", "slices.pkl")
        dataset = SplitDataset(text_data, 500, 800, 20)

        train_indices, valid_indeces = train_test_split(
            np.arange(len(dataset)), test_size=0.1
        )
        self.train_dataset = data.Subset(dataset, train_indices)
        self.valid_dataset = data.Subset(dataset, valid_indeces)

        # init network
        self.embedding = nn.Embedding(127 + 2, 32)
        self.lstm1 = nn.LSTM(32, 128, bidirectional=True, batch_first=True)
        _freeze_bias(self.lstm1)
        self.lstm2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True)
        _freeze_bias(self.lstm2)
        self.out = nn.Linear(128, 1)

    def get_keras_equivalent(self):
        from tensorflow.keras import layers, models

        k_model = models.Sequential()
        k_model.add(layers.Input(shape=(None,)))

        k_model.add(layers.Embedding(127 + 2, 32))
        k_model.layers[-1].set_weights([self.embedding.weight.detach().cpu().numpy()])

        k_model.add(
            layers.Bidirectional(
                layers.LSTM(128, return_sequences=True, use_bias=False)
            )
        )
        k_model.layers[-1].set_weights(
            [
                np.transpose(x.detach().cpu().numpy())
                for name, x in self.lstm1.named_parameters()
                if not name.startswith("bias")
            ]
        )

        k_model.add(
            layers.Bidirectional(layers.LSTM(64, return_sequences=True, use_bias=False))
        )
        k_model.layers[-1].set_weights(
            [
                np.transpose(x.detach().cpu().numpy())
                for name, x in self.lstm2.named_parameters()
                if not name.startswith("bias")
            ]
        )

        k_model.add(layers.Dense(2))
        k_model.layers[-1].set_weights(
            [np.transpose(x.detach().cpu().numpy()) for x in self.out.parameters()]
        )
        return k_model

    def forward(self, x):
        h = self.embedding(x.long())
        h, _ = self.lstm1(h)
        h, _ = self.lstm2(h)
        h = self.out(h)
        return h

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(
            y_hat, y.float(), pos_weight=torch.tensor(10.0)
        )
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        val_loss = F.binary_cross_entropy_with_logits(
            y_hat, y.float(), pos_weight=torch.tensor(10.0)
        )

        threshold = 0.5
        n_labels = y.shape[-1]

        y_flat = y.view((-1, n_labels))
        pred_flat = y_hat.view((-1, n_labels)) > threshold

        tp = ((pred_flat == 1) & (y_flat == 1)).sum(dim=0)
        fp = ((pred_flat == 1) & (y_flat == 0)).sum(dim=0)
        fn = ((pred_flat == 0) & (y_flat == 1)).sum(dim=0)

        return {"val_loss": val_loss, "tp": tp, "fp": fp, "fn": fn}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tp = torch.stack([x["tp"] for x in outputs]).sum(dim=0)
        fp = torch.stack([x["fp"] for x in outputs]).sum(dim=0)
        fn = torch.stack([x["fn"] for x in outputs]).sum(dim=0)

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        for i in range(len(f1)):
            print(
                f"f1={f1[i]:.3f}\tprecision={precision[i]:.3f}\trecall={recall[i]:.3f}"
            )

        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=6,
            collate_fn=SplitDataset.collate_fn,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.valid_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=6,
            collate_fn=SplitDataset.collate_fn,
        )
