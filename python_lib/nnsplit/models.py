import numpy as np
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(127 + 2, 25)
        self.lstm1 = nn.LSTM(25, 128, bidirectional=True, batch_first=True, bias=False)
        self.out = nn.Linear(256, 2)

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
            [np.transpose(x.detach().cpu().numpy()) for x in self.lstm1.parameters()]
        )

        k_model.add(layers.Dense(2))
        k_model.layers[-1].set_weights(
            [np.transpose(x.detach().cpu().numpy()) for x in self.out.parameters()]
        )
        return k_model

    def forward(self, x):
        h = self.embedding(x.long())
        h, _ = self.lstm1(h)
        h = self.out(h)
        return h
