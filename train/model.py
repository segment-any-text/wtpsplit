import logging
from pathlib import Path
import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from sklearn.model_selection import train_test_split
from torch.utils import data
from dataset import SplitDataset
from argparse import ArgumentParser


class Network(pl.LightningModule):
    TORCHSCRIPT_CPU_NAME = "torchscript_cpu_model.pt"
    TORCHSCRIPT_CUDA_NAME = "torchscript_cuda_model.pt"
    ONNX_NAME = "model.onnx"

    def __init__(self, text_dataset, labeler, hparams):
        super().__init__()
        self.text_dataset = text_dataset
        self.labeler = labeler
        self.hparams = hparams

        self.embedding = nn.Embedding(256, 32)
        self.lstm1 = nn.LSTM(32, 128, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True)
        self.out = nn.Linear(128, 2)

    def prepare_data(self):
        dataset = SplitDataset(self.text_dataset, self.labeler, 500, 800, 20)

        train_indices, valid_indices = train_test_split(
            np.arange(len(dataset)), test_size=self.hparams.test_size, random_state=1234
        )
        self.train_dataset = data.Subset(dataset, train_indices)
        self.valid_dataset = data.Subset(dataset, valid_indices)

    def forward(self, x):
        h = self.embedding(x.long())
        h, _ = self.lstm1(h)
        h, _ = self.lstm2(h)
        h = self.out(h)
        return h

    @staticmethod
    def loss(y_hat, y):
        weight = torch.tensor([2.0, 0.1]).view((1, 1, 2)).to(y_hat.device)

        return F.binary_cross_entropy_with_logits(
            y_hat, y.float(), pos_weight=torch.tensor(10.0), weight=weight
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        val_loss = self.loss(y_hat, y)

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

        print()

        for i in range(len(f1)):
            print(
                f"f1={f1[i]:.3f}\tprecision={precision[i]:.3f}\trecall={recall[i]:.3f}"
            )

        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        # define 1 epoch = n random samples from train data
        # multiprocessing with spacy leaks memory so could go OOM without a sample limit
        # reload_dataloaders_every_epoch must be True in trainer
        # so that memory is cleaned up after each epoch

        epoch_indices = np.random.choice(
            np.arange(len(self.train_dataset)), self.hparams.train_size
        )
        epoch_sample = data.Subset(self.train_dataset, epoch_indices)

        return data.DataLoader(
            epoch_sample,
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

    def store(self, directory):
        store_directory = Path(directory)
        store_directory.mkdir(exist_ok=True, parents=True)

        sample = torch.zeros([1, 100], dtype=torch.uint8)
        # model is trained with fp16, so it can be safely quantized to 16 bit
        # CPU model is quantized to 8 bit, with minimal loss in accuracy
        quantized_model = Network(self.text_dataset, self.labeler, self.hparams)
        quantized_model.load_state_dict(self.state_dict())
        quantized_model = torch.quantization.quantize_dynamic(
            quantized_model, {nn.LSTM, nn.Linear}, dtype=torch.qint8, inplace=True
        )
        traced = torch.jit.trace(quantized_model, sample)
        traced.save(str(store_directory / self.TORCHSCRIPT_CPU_NAME))

        if torch.cuda.is_available():
            traced = torch.jit.trace(self.half().cuda(), sample.cuda())
            traced.save(str(store_directory / self.TORCHSCRIPT_CUDA_NAME))
        else:
            logging.warn(
                "CUDA is not available. CUDA version of model could not be stored."
            )

        torch.onnx.export(
            self.float().cpu(),
            sample.cpu(),
            store_directory / self.ONNX_NAME,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 1: "length"},
                "output": {0: "batch", 1: "length"},
            },
        )

    @staticmethod
    def get_parser():
        parser = ArgumentParser()
        parser.add_argument(
            "--test_size", type=int, help="Number of samples for test set."
        )
        parser.add_argument(
            "--train_size",
            type=int,
            help="Number of samples to train on for one epoch. "
            "Will be sampled without replacement from the text dataset.",
        )
        parser = Trainer.add_argparse_args(parser)

        parser.set_defaults(
            train_size=1_000_000,
            test_size=50_000,
            max_epochs=1,
            reload_dataloaders_every_epoch=True,
        )
        return parser
