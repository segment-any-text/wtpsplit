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
    """Synthetic unbatched samples covering the chunk lengths ``extract`` actually feeds.

    AITune stacks these on dim 0, so each tensor is one example: ``(seq,)`` or
    ``(seq, num_hashes)``. Lengths stay inside the position-embedding window and
    on the character-model downsampling grid.
    """
    import torch

    config = model.config
    model_type = getattr(config, "model_type", "") or ""
    rate = int(getattr(config, "downsampling_rate", 1) or 1)
    lengths = tuple(length for length in (64, 128, 256, 512) if length % rate == 0) or (rate,)
    uses_lang = getattr(config, "language_adapter", "off") in {"on", "shared"}
    samples: list[dict[str, Any]] = []

    for seq_len in lengths:
        if "xlm" in model_type:
            # CLS/SEP are added around the chunk; stay within the 512 position window.
            length = min(seq_len + 2, 512)
            sample = {
                "input_ids": torch.randint(0, 50000, (length,), dtype=torch.long, device=device),
                "attention_mask": torch.ones(length, dtype=torch.float32, device=device),
            }
        else:
            num_hashes = getattr(config, "num_hash_functions", 8)
            num_buckets = getattr(config, "num_hash_buckets", 10000)
            sample = {
                "hashed_ids": torch.randint(0, num_buckets, (seq_len, num_hashes), dtype=torch.long, device=device),
                "attention_mask": torch.ones(seq_len, dtype=torch.float32, device=device),
            }
        if uses_lang and "xlm" not in model_type:
            # Batched to shape (batch,) by AITune, matching ``extract``.
            sample["language_ids"] = torch.zeros((), dtype=torch.long, device=device)
        samples.append(sample)

    return samples


def _backend_cls(*names: str):
    for name in names:
        factory = _try_backend("aitune.torch.backend", name)
        if factory is not None:
            return factory
    raise ImportError(
        "Could not import an AITune TorchInductor backend "
        f"(tried {', '.join(names)}). Upgrade with: pip install -U 'aitune>=0.4'"
    )


def _strategy_cls(*names: str):
    for name in names:
        factory = _try_backend("aitune.torch.tune_strategy", name)
        if factory is not None:
            return factory
    raise ImportError(
        "Could not import an AITune tune strategy "
        f"(tried {', '.join(names)}). Upgrade with: pip install -U 'aitune>=0.4'"
    )


def _inductor_backend(*, mode: str | None, fullgraph: bool, dynamic: bool):
    """TorchInductor backend, using the current JIT class and falling back to older names."""
    backend_cls = _backend_cls("TorchInductorJitBackend", "TorchInductorBackend")
    config_cls = _try_backend("aitune.torch.backend", "TorchInductorJitBackendConfig")
    if config_cls is None:
        return backend_cls()
    config_kwargs: dict[str, Any] = {"fullgraph": fullgraph, "dynamic": dynamic}
    if mode is not None:
        config_kwargs["mode"] = mode
    try:
        return backend_cls(config=config_cls(**config_kwargs))
    except TypeError:
        return backend_cls()


def _resolve_strategy(
    strategy_name: str | None,
    backends: list | None,
    *,
    mode: str | None,
    fullgraph: bool,
    dynamic: bool,
):
    first_wins = _strategy_cls("FirstWinsStrategy")
    one_backend = _strategy_cls("OneBackendStrategy")

    if backends is not None:
        return first_wins(backends=backends)

    key = (strategy_name or "first_wins").lower().replace("-", "_")
    inductor = _inductor_backend(mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    if key in ("inductor", "inductor_only", "torch_inductor"):
        return one_backend(backend=inductor)

    available = _default_aitune_backends(inductor)
    if key in ("highest_throughput", "best", "max_throughput"):
        strategy_cls = _strategy_cls("MaxThroughputStrategy", "HighestThroughputStrategy")
        try:
            return strategy_cls(backends=available)
        except TypeError:
            return strategy_cls()

    return first_wins(backends=available)


def _default_aitune_backends(inductor_backend) -> list:
    """Backends tried in order for ``first_wins`` (TensorRT first when importable)."""
    backends = []
    for class_name in ("TensorRTBackend", "TorchTensorRTJitBackend"):
        factory = _try_backend("aitune.torch.backend", class_name)
        if factory is not None:
            backends.append(factory())
    backends.append(inductor_backend)
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


def apply_aitune(model, *, mode: str | None = None, fullgraph: bool = False, dynamic: bool = True, **aitune_kwargs):
    """Wrap ``model`` with AITune and run ahead-of-time tuning. Returns the wrapped module."""
    import logging

    import torch

    ait = _require_aitune()
    from aitune.torch.dataloader import DynamicShapeDataset

    logger = logging.getLogger(__name__)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "AITune requires a CUDA GPU. Move the model with `.to('cuda')` before `optimize(backend='aitune')`."
        )

    device = getattr(model, "device", None)
    if device is None or getattr(device, "type", None) != "cuda":
        raise RuntimeError(
            "AITune requires the model on CUDA. Call `.to('cuda')` before `optimize(backend='aitune')`."
        )

    model.eval()
    calibration = aitune_kwargs.pop("aitune_calibration", None)
    dataset = (
        calibration if calibration is not None else DynamicShapeDataset(_build_calibration_dataset(model, device))
    )
    batch_sizes = aitune_kwargs.pop("aitune_batch_sizes", None) or [1, 2]
    max_batches = aitune_kwargs.pop("aitune_max_batches", 4)
    strategy_name = aitune_kwargs.pop("aitune_strategy", None)
    backends = aitune_kwargs.pop("aitune_backends", None)
    dry_run = aitune_kwargs.pop("aitune_dry_run", False)
    if len(set(batch_sizes)) < 2:
        logger.warning(
            "AITune needs at least two batch sizes to mark the batch axis as dynamic; "
            "using [1, 2] so later split() batch sizes still match."
        )
        batch_sizes = [1, 2]

    strategy = _resolve_strategy(
        strategy_name, backends, mode=mode, fullgraph=fullgraph, dynamic=dynamic
    )
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
