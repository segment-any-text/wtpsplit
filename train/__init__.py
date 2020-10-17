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
    parser.add_argument(
        "--spacy_model",
        help="Name of the spacy model to use for labelling.",
        required=True,
    )
    parser.add_argument(
        "--text_path",
        help="Path to the text file to use for training.",
        required=True,
    )
    parser.add_argument(
        "--slice_path",
        help="Path to the slice file to use for training.",
        required=True,
    )
    parser.add_argument(
        "--model_path",
        help="Directory to store the model at.",
    )

    hparams = parser.parse_args()

    if hparams.logger:
        store_code(wandb_logger.experiment)

    labeler = Labeler(
        [
            SpacySentenceTokenizer(
                hparams.spacy_model, lower_start_prob=0.7, remove_end_punct_prob=0.7
            ),
            SpacyWordTokenizer(hparams.spacy_model),
        ]
    )

    model = Network(
        MemoryMapDataset(hparams.text_path, hparams.slice_path),
        labeler,
        hparams,
    )
    n_params = np.sum([np.prod(x.shape) for x in model.parameters()])

    trainer = Trainer.from_argparse_args(hparams)
    print(f"Training model with {n_params} parameters.")
    trainer.fit(model)

    if hparams.logger:
        model.store(Path(wandb_logger.experiment.dir) / "model")

    if hparams.model_path:
        model.store(Path(hparams.model_path))
