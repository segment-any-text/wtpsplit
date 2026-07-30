"""Device and compiled-backend configuration tests."""

from types import SimpleNamespace

import pytest
import torch

from wtpsplit import SaT
from wtpsplit import _configure_pytorch_model
from wtpsplit.extract import PyTorchWrapper


def make_wrapper():
    module = torch.nn.Linear(4, 2)
    module.config = SimpleNamespace()
    return PyTorchWrapper(module)


def test_configure_pytorch_model_moves_to_requested_device():
    wrapper = make_wrapper()
    _configure_pytorch_model(wrapper, device="cpu")
    assert wrapper.model.weight.device.type == "cpu"


@pytest.mark.parametrize(
    ("options", "expected"),
    [({}, {}), ({"backend": "eager", "dynamic": True}, {"backend": "eager", "dynamic": True})],
)
def test_configure_pytorch_model_passes_compile_options(monkeypatch, options, expected):
    wrapper = make_wrapper()
    captured = {}

    def fake_compile(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    _configure_pytorch_model(wrapper, compile=options)

    assert captured == {
        "model": wrapper.model,
        "kwargs": expected,
    }


def test_onnx_rejects_pytorch_device_and_compile_options():
    with pytest.raises(ValueError, match="ort_providers"):
        SaT("unused", ort_providers=["CPUExecutionProvider"], device="cpu")
    with pytest.raises(ValueError, match="PyTorch"):
        SaT("unused", ort_providers=["CPUExecutionProvider"], compile=True)


def test_sat_compiled_backend_runs_end_to_end():
    sat = SaT(
        "segment-any-text/sat-3l-sm",
        hub_prefix=None,
        device="cpu",
        compile={"backend": "eager"},
    )
    result = sat.split("First sentence. Second sentence.")
    assert "".join(result) == "First sentence. Second sentence."


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable on this runner")
def test_mps_device_move():
    wrapper = make_wrapper()
    _configure_pytorch_model(wrapper, device="mps")
    assert wrapper.model.weight.device.type == "mps"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable on this runner")
def test_sat_mps_inference_end_to_end():
    sat = SaT("segment-any-text/sat-3l-sm", hub_prefix=None, device="mps")
    result = sat.split("First sentence. Second sentence.")
    assert "".join(result) == "First sentence. Second sentence."
