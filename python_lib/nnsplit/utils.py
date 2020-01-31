from pathlib import Path
import torch

ROOT = (Path(__file__) / "..").resolve()
DEFAULT_DATA_DIRECTORY = ROOT / "data"


def text_to_id(char):
    x = ord(char)
    return x + 2 if x <= 127 else 1


def id_to_text(x):
    return chr(x - 2) if (x - 2) <= 127 and x > 1 else "X"


def store_model(learner, path):
    path.parents[0].mkdir(exist_ok=True, parents=True)

    # always store on CPU for compatibility, can still convert to CUDA after loading
    traced = torch.jit.trace(learner.model.cpu(), learner.data.train_ds[:1][0])
    traced.save(str(path))


def load_model(name_or_path):
    if isinstance(name_or_path, Path) or "." in name_or_path:  # assume path
        traced = torch.jit.load(str(name_or_path))
    else:  # assume name
        traced = torch.jit.load(str(DEFAULT_DATA_DIRECTORY / f"{name_or_path}.pt"))

    return traced
