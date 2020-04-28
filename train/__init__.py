import numpy as np
from pytorch_lightning.trainer import Trainer
from models import Network

model = Network()
n_params = np.sum([np.prod(x.shape) for x in model.parameters()])

trainer = Trainer(gpus=1, max_epochs=1)
print(f"Training model with {n_params} parameters.")
trainer.fit(model)
