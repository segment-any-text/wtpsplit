from pathlib import Path
import shutil
from glob import glob
import numpy as np
from argparse import ArgumentParser
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from models import Network


def store_code(run):
    dst = Path(run.dir) / "code" / "train"
    src = (Path(__file__) / "..").resolve()

    for f in glob(str(src / "*.py")):
        shutil.copy(f, dst)


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="nnsplit")

    parser = ArgumentParser()
    parser = Network.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=1, max_epochs=1, reload_dataloaders_every_epoch=True, logger=wandb_logger,
    )

    hparams = parser.parse_args()

    if hparams.logger:
        store_code(wandb_logger.experiment)

    model = Network(hparams)
    n_params = np.sum([np.prod(x.shape) for x in model.parameters()])

    trainer = Trainer.from_argparse_args(hparams)
    print(f"Training model with {n_params} parameters.")
    trainer.fit(model)

    if hparams.logger:
        model.store(Path(wandb_logger.experiment.dir) / "model")
