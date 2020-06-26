from pathlib import Path
import shutil
from glob import glob
import numpy as np
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from model import Network
from text_data import MemoryMapDataset
from labeler import Labeler, SpacySentenceTokenizer, SpacyWordTokenizer


def store_code(run):
    dst = Path(run.dir) / "code" / "train"
    dst.mkdir(parents=True, exist_ok=True)
    src = (Path(__file__) / "..").resolve()

    for f in glob(str(src / "*.py")):
        shutil.copy(f, dst)


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="nnsplit")

    parser = Network.get_parser()
    parser.set_defaults(logger=wandb_logger)
    hparams = parser.parse_args()

    if hparams.logger:
        store_code(wandb_logger.experiment)

    labeler = Labeler(
        [
            SpacySentenceTokenizer(
                "de_core_news_sm", lower_start_prob=0.7, remove_end_punct_prob=0.7
            ),
            SpacyWordTokenizer("de_core_news_sm"),
        ]
    )

    model = Network(MemoryMapDataset("texts.txt", "slices.pkl"), labeler, hparams)
    n_params = np.sum([np.prod(x.shape) for x in model.parameters()])

    trainer = Trainer.from_argparse_args(hparams)
    print(f"Training model with {n_params} parameters.")
    trainer.fit(model)

    if hparams.logger:
        model.store(Path(wandb_logger.experiment.dir) / "model")
