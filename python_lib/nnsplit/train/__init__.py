from models import Network
from pytorch_lightning.trainer import Trainer

model = Network()
trainer = Trainer(gpus=1, max_epochs=1)

trainer.fit(model)
