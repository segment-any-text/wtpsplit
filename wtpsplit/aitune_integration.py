"""Optional NVIDIA AITune integration for :meth:`PyTorchWrapper.optimize`."""

from __future__ import annotations

from typing import Any


def _require_aitune():
    try:
        import aitune.torch as ait  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "backend='aitune' requires NVIDIA AITune. Install with:\n"
            "  pip install wtpsplit[aitune] --extra-index-url https://pypi.nvidia.com"
        ) from e
    return ait


def _build_calibration_dataset(model, device) -> list[dict[str, Any]]:
    """Synthetic calibration samples covering short and long chunk lengths."""
    import torch

    config = model.config
    model_type = getattr(config, "model_type", "") or ""
    lengths = (64, 128, 256, 512)
    samples: list[dict[str, Any]] = []

    for seq_len in lengths:
        if "xlm" in model_type:
            length = min(seq_len + 2, 514)
            samples.append(
                {
                    "input_ids": torch.randint(0, 50000, (length,), dtype=torch.long, device=device),
                    "attention_mask": torch.ones(length, dtype=torch.float32, device=device),
                }
            )
        else:
            num_hashes = getattr(config, "num_hash_functions", 8)
            num_buckets = getattr(config, "num_hash_buckets", 10000)
            samples.append(
                {
                    "hashed_ids": torch.randint(
                        0, num_buckets, (seq_len, num_hashes), dtype=torch.long, device=device
                    ),
                    "attention_mask": torch.ones(seq_len, dtype=torch.float32, device=device),
                }
            )

    return samples


def _resolve_strategy(strategy_name: str | None, backends: list | None):
    from aitune.torch.backend import TorchInductorBackend
    from aitune.torch.tune_strategy import FirstWinsStrategy, HighestThroughputStrategy, OneBackendStrategy

    if backends is not None:
        return FirstWinsStrategy(backends=backends)

    key = (strategy_name or "first_wins").lower().replace("-", "_")
    if key in ("inductor", "inductor_only", "torch_inductor"):
        return OneBackendStrategy(backend=TorchInductorBackend())

    backends = _default_aitune_backends()
    if key in ("highest_throughput", "best", "max_throughput"):
        from aitune.torch.backend import TorchEagerBackend

        return HighestThroughputStrategy(backends=backends + [TorchEagerBackend()])

    return FirstWinsStrategy(backends=backends)


def _default_aitune_backends() -> list:
    """Backends tried in order for ``first_wins`` (TensorRT first when available)."""
    from aitune.torch.backend import TorchInductorBackend

    backends = []
    for factory in (
        _try_backend("aitune.torch.backend", "TensorRTBackend"),
        _try_backend("aitune.torch.backend", "TorchTensorRTJitBackend"),
        _try_backend("aitune.torch.backend", "TorchInductorBackend"),
    ):
        if factory is not None:
            backends.append(factory())
    if not backends:
        backends.append(TorchInductorBackend())
    return backends


def _try_backend(module_name: str, class_name: str):
    try:
        import importlib

        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)
    except Exception:
        return None


def pop_aitune_kwargs(compile_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract AITune-specific keys from ``compile_kwargs`` (mutates the dict)."""
    keys = (
        "aitune_batch_sizes",
        "aitune_max_batches",
        "aitune_strategy",
        "aitune_backends",
        "aitune_dry_run",
        "aitune_calibration",
    )
    out = {}
    for key in keys:
        if key in compile_kwargs:
            out[key] = compile_kwargs.pop(key)
    return out


def apply_aitune(model, **aitune_kwargs):
    """Wrap ``model`` with AITune and run ahead-of-time tuning. Returns the wrapped module."""
    import torch

    ait = _require_aitune()
    from aitune.torch.dataloader import DynamicShapeDataset
    from aitune.torch.tune_strategy import TuneStrategy

    if not torch.cuda.is_available():
        raise RuntimeError(
            "AITune requires a CUDA GPU. Move the model with `.to('cuda')` before `optimize(backend='aitune')`."
        )

    device = getattr(model, "device", None)
    if device is None or device.type != "cuda":
        raise RuntimeError(
            "AITune requires the model on CUDA. Call `.to('cuda')` before `optimize(backend='aitune')`."
        )

    calibration = aitune_kwargs.pop("aitune_calibration", None)
    dataset = (
        calibration
        if calibration is not None
        else DynamicShapeDataset(_build_calibration_dataset(model, device))
    )
    batch_sizes = aitune_kwargs.pop("aitune_batch_sizes", None) or [1, 2]
    max_batches = aitune_kwargs.pop("aitune_max_batches", 4)
    strategy_name = aitune_kwargs.pop("aitune_strategy", None)
    backends = aitune_kwargs.pop("aitune_backends", None)
    dry_run = aitune_kwargs.pop("aitune_dry_run", False)

    strategy: TuneStrategy = _resolve_strategy(strategy_name, backends)
    name = f"{model.__class__.__module__}.{model.__class__.__qualname__}".replace("/", "_")
    wrapped = ait.Module(model, name=name, strategy=strategy)

    ait.tune(
        wrapped,
        dataset,
        batch_sizes=batch_sizes,
        max_num_batches_per_batch_size=max_batches,
        device=device,
        dry_run=dry_run,
    )
    if not dry_run:
        _install_metadata_fallback(wrapped)
    return wrapped


def _install_metadata_fallback(wrapped):
    """Allow AITune dynamic-shape metadata to match runtime samples.

    AITune records dynamic axes during tuning, but its runtime lookup can still
    require exact ``SampleMetadata`` dictionary membership. SaT/WtP chunk lengths
    vary naturally, so fall back to the compatible ``TensorSpec.matches`` check
    before reporting that no backend was found.
    """
    import types

    from aitune.torch.module.sample_metadata import SampleMetadata

    tuned = getattr(wrapped, "_self_wrapper", None)
    if tuned is None or not hasattr(tuned, "_backends"):
        return

    def metadata_matches(expected, actual):
        if expected == actual:
            return True
        if expected.llm_phase != actual.llm_phase:
            return False
        if expected.other_data != actual.other_data:
            return False

        expected_specs = {spec.name: spec for spec in expected.tensor_specs}
        actual_specs = {spec.name: spec for spec in actual.tensor_specs}
        if expected_specs.keys() != actual_specs.keys():
            return False

        for name, actual_spec in actual_specs.items():
            expected_spec = expected_specs[name]
            if expected_spec.dtype != actual_spec.dtype:
                return False
            if not expected_spec.matches(actual_spec):
                return False
        return True

    def safe_call_backend_with_dynamic_match(self, sample):
        args, kwargs = sample
        sample_metadata = SampleMetadata.from_inputs(args, kwargs, strict=self._config.strict_mode)
        backend = self._backends.get(sample_metadata)
        if backend is None:
            for candidate_metadata, candidate_backend in self._backends.items():
                if metadata_matches(candidate_metadata, sample_metadata):
                    backend = candidate_backend
                    break
        if backend is None:
            raise RuntimeError(self.ERROR_NO_BACKEND_FOUND.format(sample_metadata))
        return backend.infer(*args, **kwargs)

    tuned.safe_call_backend = types.MethodType(safe_call_backend_with_dynamic_match, tuned)
    tuned._backend_func = tuned.safe_call_backend
