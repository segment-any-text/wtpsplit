import math
import numpy as np
import torch
from torch import nn


def _freeze_bias(lstm):
    for name, param in lstm.named_parameters():
        if name.startswith("bias"):
            param.requires_grad = False
            param[:] = 0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

    def forward(self, x, positions):
        pe = torch.zeros(
            [positions.size(0), self.max_len, self.d_model], device=x.device
        )

        pe[:, :, 0::2] = torch.sin(positions.unsqueeze(-1) * self.div_term)
        pe[:, :, 1::2] = torch.cos(positions.unsqueeze(-1) * self.div_term)

        x = x + pe
        return x


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(127 + 2, 32)
        self.pe = PositionalEncoding(32, 100)
        self.lstm1 = nn.LSTM(32, 128, bidirectional=True, batch_first=True)
        _freeze_bias(self.lstm1)
        self.lstm2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True)
        _freeze_bias(self.lstm2)
        self.out = nn.Linear(128, 2)

    def get_keras_equivalent(self):
        from tensorflow.keras import layers, models

        k_model = models.Sequential()
        k_model.add(layers.Input(shape=(None,)))

        k_model.add(layers.Embedding(127 + 2, 25))
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
        h = self.embedding(x[:, :, 0].long())
        h = self.pe(h, x[:, :, 1])
        h, _ = self.lstm1(h)
        h, _ = self.lstm2(h)
        h = self.out(h)
        return h
