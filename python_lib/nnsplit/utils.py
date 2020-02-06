import io
import logging
from pathlib import Path
import pkgutil
import torch
import numpy as np
from .defaults import DEVICE


def text_to_id(char):
    x = ord(char)
    return x + 2 if x <= 127 else 1


def id_to_text(x):
    return chr(x - 2) if (x - 2) <= 127 and x > 1 else "X"


def store_model(learner, store_directory):
    store_directory = Path(store_directory)
    store_directory.mkdir(exist_ok=True, parents=True)

    # model is trained with fp16, so it can be safely quantized to 16 bit
    # CPU tensors do not support 16 bit embeddings yet so ts_cpu.pt has 32 bit weights
    traced = torch.jit.trace(learner.model.float().cpu(), learner.data.train_ds[:1][0])
    traced.save(str(store_directory / "ts_cpu.pt"))

    if torch.cuda.is_available():
        traced = torch.jit.trace(
            learner.model.half().cuda(), learner.data.train_ds[:1][0].cuda()
        )
        traced.save(str(store_directory / "ts_cuda.pt"))
    else:
        logging.warn(
            "CUDA is not available. CUDA version of model could not be stored."
        )

    import tensorflowjs as tfjs  # noqa: F401

    tfjs.converters.save_keras_model(
        learner.model.get_keras_equivalent(),
        str(store_directory / "tfjs_model"),
        quantization_dtype=np.uint16,
    )


def _get_filename(device):
    filename = (
        "ts_cpu.pt"
        if device == torch.device("cpu") or not torch.cuda.is_available()
        else "ts_cuda.pt"
    )
    return filename


def load_provided_model(name, device=DEVICE):
    bin_data = pkgutil.get_data(__package__, f"data/{name}/{_get_filename(device)}")
    buffer = io.BytesIO(bin_data)

    return torch.jit.load(buffer)


def load_model(store_directory, device=DEVICE):
    full_path = Path(store_directory) / _get_filename(device)

    return torch.jit.load(str(full_path))
