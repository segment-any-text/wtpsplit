import io
import logging
from pathlib import Path
import pkgutil
import torch
from torch import nn
import numpy as np
from .defaults import DEVICE


def text_to_id(char):
    x = ord(char)
    return x + 2 if x <= 127 else 1


def id_to_text(x):
    return chr(x - 2) if (x - 2) <= 127 and x > 1 else "X"


def store_model(model, store_directory):
    store_directory = Path(store_directory)
    store_directory.mkdir(exist_ok=True, parents=True)

    sample = torch.zeros([1, 100])
    # model is trained with fp16, so it can be safely quantized to 16 bit
    # CPU model is quantized to 8 bit, with minimal loss in accuracy
    # according to tests in train.ipynb
    quantized_model = torch.quantization.quantize_dynamic(
        model.float().cpu(), {nn.LSTM, nn.Linear}, dtype=torch.qint8
    )
    traced = torch.jit.trace(quantized_model, sample)
    traced.save(str(store_directory / "ts_cpu.pt"))

    if torch.cuda.is_available():
        traced = torch.jit.trace(model.half().cuda(), sample.cuda())
        traced.save(str(store_directory / "ts_cuda.pt"))
    else:
        logging.warn(
            "CUDA is not available. CUDA version of model could not be stored."
        )

    import tensorflowjs as tfjs  # noqa: F401

    tfjs.converters.save_keras_model(
        model.get_keras_equivalent(),
        str(store_directory / "tfjs_model"),
        quantization_dtype=np.uint8,
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
