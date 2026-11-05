from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from transformers.utils.hub import cached_file

from wtpsplit._host_kernels import (
    decode_subword_split,
    exact_float16_logit_threshold,
    extract_subword_logits_fast,
    project_subword_proba,
)
from wtpsplit._inference import (
    decode_probabilities,
    iter_probabilities,
    normalize_input_line_breaks,
    resolve_text_input,
    validate_input_line_breaks,
    validate_segmentation_options,
)
from wtpsplit.constants import (
    DEFAULT_SENTENCE_THRESHOLD,
    DEFAULT_STRIDE,
    calibrated_threshold_for_checkpoint,
    default_threshold_for_checkpoint,
)
from wtpsplit.extract import (
    CHAR_MODEL_TYPES,
    DEFAULT_TOKENIZERS,
    PyTorchWrapper,
    SaTORTWrapper,
    extract,
    max_content_block_size,
)
from wtpsplit.model_registry import register_sat_configs, register_sat_models
from wtpsplit.segmentation import Segmentation, boundary_confidences, sentence_spans
from wtpsplit.utils import Constants, sigmoid, token_to_char_probs

__version__ = "3.0.0"
__all__ = ["DEFAULT_STRIDE", "SaT", "Segmentation", "WtP", "__version__"]

warnings.simplefilter("default", DeprecationWarning)  # show by default
warnings.simplefilter("ignore", category=FutureWarning)  # for transformers


def _manual_lora_merge(model, lora_load_path):
    """Merge LoRA adapter weights directly into a model's parameters.

    Lightweight alternative to ``adapters`` library that only
    supports the ``merge_lora=True`` path (the default). Reads the
    adapter‐hub file format produced by ``wtpsplit/train/train_lora.py``.
    """
    import json
    import torch

    lora_dir = Path(lora_load_path)

    # --- read adapter config for LoRA hyper-parameters ---
    config_path = lora_dir / "adapter_config.json"
    with open(config_path) as f:
        adapter_cfg = json.load(f)
    try:
        config_section = adapter_cfg["config"]
        lora_r = config_section["r"]
        lora_alpha = config_section["alpha"]
        scaling = lora_alpha / lora_r
    except KeyError as e:
        raise ValueError(
            f"Invalid LoRA adapter configuration in '{config_path}': "
            f"missing required key {e!r}. Expected keys: 'config' with 'r' and 'alpha'."
        ) from e
    except TypeError as e:
        raise ValueError(
            f"Invalid LoRA adapter configuration in '{config_path}': "
            "unexpected structure; expected a JSON object with a 'config' mapping containing 'r' and 'alpha'."
        ) from e
    except ZeroDivisionError as e:
        raise ValueError(f"Invalid LoRA adapter configuration in '{config_path}': 'r' must be a non-zero value.") from e

    for metadata_key, model_value in [
        ("hidden_size", getattr(model.config, "hidden_size", None)),
        ("num_hidden_layers", getattr(model.config, "num_hidden_layers", None)),
        ("model_type", getattr(model.config, "model_type", None)),
    ]:
        adapter_value = adapter_cfg.get(metadata_key)
        if adapter_value is not None and adapter_value != model_value:
            raise ValueError(
                f"LoRA adapter {metadata_key}={adapter_value!r} does not match the base model value {model_value!r}."
            )

    # --- validate LoRA weight deltas before mutating the model ---
    adapter_weights = torch.load(lora_dir / "pytorch_adapter.bin", map_location="cpu", weights_only=True)

    # Group lora_A / lora_B pairs by their target module.
    # Key pattern: ``<module_path>.loras.<adapter_name>.lora_A``
    lora_pairs: dict = {}
    for key, tensor in adapter_weights.items():
        if ".loras." not in key:
            continue
        base_key = key.rsplit(".loras.", 1)[0]
        if key.endswith(".lora_A"):
            lora_pairs.setdefault(base_key, {})["A"] = tensor
        elif key.endswith(".lora_B"):
            lora_pairs.setdefault(base_key, {})["B"] = tensor

    if not lora_pairs:
        raise ValueError(f"No LoRA tensor pairs were found in '{lora_dir / 'pytorch_adapter.bin'}'.")

    model_params = dict(model.named_parameters())
    merge_operations = []
    for base_key, pair in lora_pairs.items():
        if "A" not in pair or "B" not in pair:
            raise ValueError(
                f"Incomplete LoRA pair for '{base_key}' in '{lora_dir / 'pytorch_adapter.bin'}': "
                "each module must have both lora_A and lora_B weights."
            )
        param_key = base_key + ".weight"
        if param_key not in model_params:
            raise KeyError(f"LoRA target parameter '{param_key}' not found in model.")
        param = model_params[param_key]
        if pair["A"].ndim != 2 or pair["B"].ndim != 2:
            raise ValueError(f"LoRA tensors for '{base_key}' must both be two-dimensional.")
        if pair["A"].shape[0] != lora_r or pair["B"].shape[1] != lora_r:
            raise ValueError(
                f"LoRA rank mismatch for '{base_key}': config r={lora_r}, "
                f"A shape={tuple(pair['A'].shape)}, B shape={tuple(pair['B'].shape)}."
            )
        delta = pair["B"] @ pair["A"]
        if delta.shape != param.shape:
            raise ValueError(
                f"LoRA tensor shape mismatch for '{param_key}': delta shape={tuple(delta.shape)}, "
                f"base shape={tuple(param.shape)}."
            )
        merge_operations.append((param, delta))

    configured_targets = config_section.get("target_modules")
    if configured_targets is not None and set(configured_targets) != set(lora_pairs):
        raise ValueError("LoRA `target_modules` metadata does not match the tensors in `pytorch_adapter.bin`.")

    # --- validate classification-head weights before mutating anything ---
    head_operations = []
    head_path = lora_dir / "pytorch_model_head.bin"
    head_config_path = lora_dir / "head_config.json"
    if head_path.exists():
        head_weights = torch.load(head_path, map_location="cpu", weights_only=True)
        if head_config_path.exists():
            with open(head_config_path) as f:
                head_config = json.load(f)
            expected_labels = head_config.get("num_labels")
            actual_labels = getattr(model.config, "num_labels", None)
            if expected_labels is not None and expected_labels != actual_labels:
                raise ValueError(
                    "The LoRA classification head size does not match the base model: "
                    f"adapter num_labels={expected_labels}, base num_labels={actual_labels}."
                )
        if not head_weights:
            raise ValueError(f"No classification-head tensors were found in '{head_path}'.")
        for key, tensor in head_weights.items():
            if key not in model_params:
                raise KeyError(f"LoRA classification-head parameter '{key}' not found in model.")
            if tensor.shape != model_params[key].shape:
                raise ValueError(
                    f"LoRA classification-head shape mismatch for '{key}': "
                    f"adapter shape={tuple(tensor.shape)}, base shape={tuple(model_params[key].shape)}."
                )
            head_operations.append((model_params[key], tensor))

    # --- all validation passed; mutate the model atomically ---
    for param, delta in merge_operations:
        with torch.no_grad():
            param.add_(scaling * delta.to(device=param.device, dtype=param.dtype))

    for param, tensor in head_operations:
        with torch.no_grad():
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))


def _configure_pytorch_model(model, *, device=None, compile=False):
    """Move and optionally compile a ``PyTorchWrapper`` model."""
    import torch

    if device is not None:
        model.to(device)

    if compile is not False:
        compile_kwargs = {} if compile is True else dict(compile)
        model.model = torch.compile(model.model, **compile_kwargs)
        model._compiled = True


def _resolve_sat_tokenizer_name(
    model_name_to_fetch: str,
    tokenizer_name_or_path,
    *,
    is_local: bool,
    from_pretrained_kwargs=None,
) -> str:
    """Pick a tokenizer for SaT when the caller did not pass one explicitly.

    Preference order:
    1. Explicit ``tokenizer_name_or_path``
    2. Tokenizer files shipped next to a local checkpoint
    3. The default tokenizer for the checkpoint's ``model_type``
    """
    if tokenizer_name_or_path is not None:
        return str(tokenizer_name_or_path)

    if is_local and (Path(model_name_to_fetch) / "tokenizer_config.json").exists():
        return model_name_to_fetch

    register_sat_configs()
    config = AutoConfig.from_pretrained(model_name_to_fetch, **(from_pretrained_kwargs or {}))
    default_tokenizer = DEFAULT_TOKENIZERS.get(config.model_type)
    if default_tokenizer is None:
        raise ValueError(
            f"No default tokenizer is known for model type {config.model_type!r}. "
            "Pass `tokenizer_name_or_path=` explicitly."
        )
    return default_tokenizer


VALID_INFERENCE_MODES = ("bitwise", "equivalent", "sparse")
DEFAULT_SPARSE_CHAR_GATE_THRESHOLD = -4.0
_UNSET = object()


class SaT:
    def __init__(
        self,
        model_name_or_model,
        tokenizer_name_or_path=None,
        from_pretrained_kwargs=None,
        ort_providers=None,
        ort_kwargs=None,
        domain: str = None,
        language: str = None,
        lora_path: str = None,  # local
        hub_prefix="segment-any-text",
        merge_lora: bool = True,
        device=None,
        compile: bool | dict = False,
        *,
        mode: Literal["bitwise", "equivalent", "sparse"] = "bitwise",
        style_or_domain: str = None,
        character_preprocessing: str | None = None,
        trim_padding: bool = True,
        trim_subword_padding: bool | None = None,
        pack_sequences: bool | None = None,
        fuse_depthwise_delta: bool | None = None,
        gate_logit_threshold: float | None = _UNSET,  # type: ignore[assignment]
        project_tokens_first: bool | None = None,
        quantization: str | None = None,
        precision: str | None = None,
        input_line_breaks: Literal["preserve", "space"] | None = None,
    ):
        """Load a SaT checkpoint.

        ``mode`` selects the numerical/execution preset:
        - ``"bitwise"`` (default): 0.0 ULP bit-for-bit identical to the reference execution path.
        - ``"equivalent"``: mathematically identical equations with unpadded GEMM tiling and
          linear operator fusion (``trim_subword_padding``, ``pack_sequences``,
          ``fuse_depthwise_delta``, ``project_tokens_first``, and ``character_preprocessing="exact"``).
        - ``"sparse"``: everything in ``"equivalent"`` plus coarse-gated sparse character head
          (``gate_logit_threshold=-4.0``) and ``FlexAttention`` when running on CUDA.

        ``input_line_breaks`` controls how the encoder sees input ``\\n``/``\\r``. ``None`` (default) uses
        the checkpoint's ``config.json`` value, which is ``"preserve"`` (text fed unchanged) unless the
        checkpoint opts in. ``"space"`` replaces each LF/CR by one space before encoding, one character
        for one character, so offsets and returned text are unchanged. Released SaT/SaT-SM models
        already treat line breaks as spaces and need no mapping. ``split_on_input_newlines`` is independent.
        """
        if mode not in VALID_INFERENCE_MODES:
            raise ValueError(f"mode must be one of {VALID_INFERENCE_MODES}, got {mode!r}")
        if input_line_breaks is not None:
            validate_input_line_breaks(input_line_breaks)
        if quantization not in (None, "int8", "combined_int8", "embedding_int8", "linear_int8"):
            raise ValueError("quantization must be None, 'int8', 'combined_int8', 'embedding_int8' or 'linear_int8'")
        if quantization is not None:
            import torch
            if ort_providers is not None or compile is not False:
                raise ValueError("INT8 is an experimental PyTorch CPU option; it cannot be combined with ONNX or compile")
            if device is not None and torch.device(device).type != "cpu":
                raise ValueError("INT8 quantization is CPU-only")
        self.quantization = quantization
        self.precision = precision
        self.precision_info = None
        precision_hardware = None
        if precision is not None:
            from wtpsplit.precision import validate_precision
            precision_hardware = validate_precision(
                precision, device=device, quantization=quantization, ort_providers=ort_providers,
                compile=compile, from_pretrained_kwargs=from_pretrained_kwargs,
            )
        resolved_char_prep = (
            character_preprocessing
            if character_preprocessing is not None
            else ("exact" if mode in ("equivalent", "sparse") else "reference")
        )
        if resolved_char_prep not in {"reference", "exact"}:
            raise ValueError("character_preprocessing must be 'reference' or 'exact'")
        self.character_preprocessing = resolved_char_prep
        self.trim_padding = trim_padding
        if style_or_domain is not None:
            if domain is not None:
                raise TypeError("Pass only one of `domain` or the deprecated `style_or_domain` alias.")
            warnings.warn(
                "`style_or_domain` is deprecated; use `domain` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            domain = style_or_domain

        if ort_providers is not None and device is not None:
            raise ValueError("`device` configures PyTorch inference; select ONNX devices with `ort_providers`.")
        if ort_providers is not None and compile is not False:
            raise ValueError("`compile` is available only for PyTorch inference, not ONNX Runtime.")

        if not isinstance(model_name_or_model, (str, Path)):
            raise TypeError(
                f"`model_name_or_model` must be a string or Path (Hugging Face ID or local directory path), "
                f"received object of type: {type(model_name_or_model)}. "
                "For offline ONNX use, please provide the path to the directory containing 'model_optimized.onnx' and 'config.json'."
            )

        self.model_name_or_model = model_name_or_model
        self.ort_providers = ort_providers
        self.ort_kwargs = ort_kwargs
        self.language = language  # Store for language-aware prior defaults
        self._compiled = compile is not False

        self.use_lora = False

        model_name = str(model_name_or_model)
        is_local = os.path.isdir(model_name)

        if not is_local and hub_prefix is not None:
            model_name_to_fetch = f"{hub_prefix}/{model_name}"
        else:
            model_name_to_fetch = model_name

        resolved_tokenizer = _resolve_sat_tokenizer_name(
            model_name_to_fetch,
            tokenizer_name_or_path,
            is_local=is_local,
            from_pretrained_kwargs=from_pretrained_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer)
        self.special_tokens = [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]

        if isinstance(model_name_or_model, (str, Path)):
            if is_local:
                model_path = Path(model_name)
                onnx_path = model_path / "model_optimized.onnx"
                if not onnx_path.exists():
                    onnx_path = None
            else:
                # no need to load if no ort_providers set
                if ort_providers is not None:
                    onnx_path = cached_file(
                        model_name_to_fetch, "model_optimized.onnx", **(from_pretrained_kwargs or {})
                    )
                else:
                    onnx_path = None

            if ort_providers is not None:
                if onnx_path is None:
                    raise ValueError(
                        "Could not find model_optimized.onnx. Omit `ort_providers` to run with PyTorch."
                    )

                try:
                    import onnxruntime as ort  # noqa
                except ModuleNotFoundError:
                    raise ValueError("Please install `onnxruntime` to use SaT with an ONNX model.")

                register_sat_configs()

                self.model = SaTORTWrapper(
                    AutoConfig.from_pretrained(model_name_to_fetch, **(from_pretrained_kwargs or {})),
                    ort.InferenceSession(str(onnx_path), providers=ort_providers, **(ort_kwargs or {})),
                )
                if lora_path:
                    raise ValueError(
                        "If using ONNX with LoRA, execute `scripts/export_to_onnx_sat.py` with `use_lora=True`."
                        "Reference the chosen `output_dir` here for `model_name_or_model`. and set `lora_path=None`."
                    )
            else:
                # to register models for AutoConfig
                try:
                    import torch  # noqa
                except ModuleNotFoundError:
                    raise ValueError("Please install `torch` to use SaT with a PyTorch model.")

                register_sat_models()

                # Check if LoRA adapter has a head with different num_labels than the base model.
                # This is needed because sm models have num_labels=1 but LoRA training uses num_labels=111+.
                # We check BEFORE loading the model so we can load with the correct num_labels.
                effective_kwargs = dict(from_pretrained_kwargs or {})
                if mode == "sparse" and device is not None and str(device).startswith("cuda"):
                    effective_kwargs.setdefault("attn_implementation", "flex_attention")
                if precision is not None:
                    # Merge adapters in FP32, then select serving precision below.
                    effective_kwargs['dtype'] = torch.float32
                if lora_path or (domain and language):
                    import json

                    head_config_path = None
                    adapter_num_labels = None

                    try:
                        if lora_path:
                            # Local adapter: read head_config.json directly
                            lora_dir = Path(lora_path)
                            head_config_path = lora_dir / "head_config.json"
                            if head_config_path.exists():
                                with open(head_config_path) as f:
                                    head_config = json.load(f)
                                adapter_num_labels = head_config.get("num_labels")
                        else:
                            # Hub adapter: fetch head_config.json first to check num_labels
                            try:
                                head_config_file = hf_hub_download(
                                    repo_id=model_name_to_fetch,
                                    subfolder=f"loras/{domain}/{language}",
                                    filename="head_config.json",
                                )
                                head_config_path = Path(head_config_file)
                                with open(head_config_path) as f:
                                    head_config = json.load(f)
                                adapter_num_labels = head_config.get("num_labels")
                            except Exception:
                                # If head_config.json doesn't exist or download fails,
                                # proceed without num_labels detection (will fail later if mismatch)
                                pass

                        if adapter_num_labels is not None:
                            # Get base model's num_labels
                            base_config = AutoConfig.from_pretrained(model_name_to_fetch)
                            base_num_labels = getattr(base_config, "num_labels", None)
                            if base_num_labels != adapter_num_labels:
                                # Warn if overriding user-provided values
                                user_num_labels = effective_kwargs.get("num_labels")
                                if user_num_labels is not None and user_num_labels != adapter_num_labels:
                                    warnings.warn(
                                        f"`num_labels` provided in `from_pretrained_kwargs` "
                                        f"({user_num_labels}) is being overridden to "
                                        f"{adapter_num_labels} to match the LoRA adapter head.",
                                        UserWarning,
                                    )
                                user_ignore = effective_kwargs.get("ignore_mismatched_sizes")
                                if user_ignore is not None and not user_ignore:
                                    warnings.warn(
                                        "`ignore_mismatched_sizes` provided in `from_pretrained_kwargs` "
                                        "is being overridden to True to allow loading a LoRA adapter "
                                        "with a different classification head size.",
                                        UserWarning,
                                    )
                                # Override to match adapter's head
                                effective_kwargs["num_labels"] = adapter_num_labels
                                effective_kwargs["ignore_mismatched_sizes"] = True
                    except (OSError, json.JSONDecodeError, ValueError) as e:
                        raise RuntimeError(
                            f"Failed to auto-detect 'num_labels' from LoRA head configuration"
                            f"{f' at {head_config_path}' if head_config_path else ''}: {e}"
                        ) from e

                self.model = PyTorchWrapper(
                    AutoModelForTokenClassification.from_pretrained(model_name_to_fetch, **effective_kwargs)
                )
            # LoRA LOADING
            if not lora_path:
                if (domain and not language) or (language and not domain):
                    raise ValueError("Please specify both language and domain!")
            if (domain and language) or lora_path:
                try:
                    # 1. Locate / download adapter files
                    if not lora_path:
                        adapter_file_path = None
                        for file in [
                            "adapter_config.json",
                            "head_config.json",
                            "pytorch_adapter.bin",
                            "pytorch_model_head.bin",
                        ]:
                            adapter_file_path = hf_hub_download(
                                repo_id=model_name_to_fetch,
                                subfolder=f"loras/{domain}/{language}",
                                filename=file,
                            )
                        lora_load_path = str(Path(adapter_file_path).parent)
                    else:
                        lora_load_path = str(lora_path)
                        lora_dir = Path(lora_load_path)
                        if not lora_dir.is_dir():
                            raise FileNotFoundError(f"`lora_path` must be a directory, but got: {lora_load_path}")

                        expected_files = [
                            "adapter_config.json",
                            "pytorch_adapter.bin",
                        ]
                        if True:  # keep in sync with load_adapter(with_head=True)
                            expected_files.extend(["head_config.json", "pytorch_model_head.bin"])

                        missing = [f for f in expected_files if not (lora_dir / f).exists()]
                        if missing:
                            raise FileNotFoundError(
                                "Could not load LoRA adapter from `lora_path` because required files are missing.\n"
                                f"- lora_path: {lora_load_path}\n"
                                f"- missing: {missing}\n"
                                "If you trained via `wtpsplit/train/train_lora.py`, the adapter is saved under:\n"
                                "  <output_dir>/<dataset_name>/<language_code>/\n"
                                "and that folder should contain the files listed above."
                            )

                    # AdapterHub currently requires transformers 4.x. wtpsplit 3
                    # supports transformers 5 only, so adapters are merged directly.
                    if not merge_lora:
                        raise RuntimeError(
                            "merge_lora=False is no longer supported in wtpsplit 3. "
                            "AdapterHub is incompatible with the required transformers 5 runtime. "
                            "Use merge_lora=True (the default)."
                        )
                    _manual_lora_merge(self.model.model, lora_load_path)

                    self.use_lora = True
                except Exception as e:  # noqa
                    if lora_path:
                        raise RuntimeError(
                            "Failed to load the local LoRA adapter provided via `lora_path`.\n"
                            f"- lora_path: {lora_path}\n"
                            "Tip: `lora_path` must point to the adapter folder containing "
                            "`adapter_config.json`, `pytorch_adapter.bin` (and if trained with head: "
                            "`head_config.json`, `pytorch_model_head.bin`).\n"
                            "Note: Adapters are model-variant specific (e.g. sat-12l-sm vs sat-12l)."
                        ) from e
                    raise RuntimeError(
                        "Failed to load the LoRA adapter from the Hugging Face Hub.\n"
                        f"- domain: {domain!r}\n"
                        f"- language: {language!r}\n"
                        "Troubleshooting tips:\n"
                        "- Ensure that an adapter with this (domain, language) combination "
                        "exists on the Hub.\n"
                        "- Check for typos and that both `domain` and `language` are "
                        "supported values.\n"
                        "- Verify that you have an active internet connection and, for private "
                        "repositories, are logged in.\n"
                        "- If you intended to load a local adapter instead, provide its directory "
                        "via `lora_path`."
                    ) from e

        self.input_line_breaks = validate_input_line_breaks(
            input_line_breaks
            if input_line_breaks is not None
            else getattr(self.__dict__["model"].config, "input_line_breaks", "preserve")
        )

        if ort_providers is None:
            _configure_pytorch_model(self.model, device=device, compile=compile)
            if precision is not None:
                from wtpsplit.precision import configure_precision
                self.model = configure_precision(self.model.model, precision, precision_hardware)
                self.precision_info = self.model.precision_info
            if quantization is not None:
                from wtpsplit.quantization import quantize_cpu
                q_mode = "combined_int8" if quantization == "int8" else quantization
                self.model = quantize_cpu(self.model.model, mode=q_mode)

        self.set_mode(
            mode,
            trim_subword_padding=trim_subword_padding,
            pack_sequences=pack_sequences,
            fuse_depthwise_delta=fuse_depthwise_delta,
            gate_logit_threshold=gate_logit_threshold,
            project_tokens_first=project_tokens_first,
            character_preprocessing=resolved_char_prep,
        )

    def set_mode(
        self,
        mode: Literal["bitwise", "equivalent", "sparse"] = "bitwise",
        *,
        trim_subword_padding: bool | None = None,
        pack_sequences: bool | None = None,
        fuse_depthwise_delta: bool | None = None,
        gate_logit_threshold: float | None = _UNSET,  # type: ignore[assignment]
        project_tokens_first: bool | None = None,
        character_preprocessing: str | None = None,
    ) -> "SaT":
        """Configure the numerical/execution preset (`"bitwise"`, `"equivalent"`, or `"sparse"`).

        Individual keyword arguments override the preset when explicitly passed.
        Returns ``self`` for method chaining.
        """
        if mode not in VALID_INFERENCE_MODES:
            raise ValueError(f"mode must be one of {VALID_INFERENCE_MODES}, got {mode!r}")
        self.mode = mode
        is_equiv_or_sparse = mode in ("equivalent", "sparse")
        self.trim_subword_padding = bool(
            is_equiv_or_sparse if trim_subword_padding is None else trim_subword_padding
        )
        self.pack_sequences = bool(
            is_equiv_or_sparse if pack_sequences is None else pack_sequences
        )
        self.fuse_depthwise_delta = bool(
            is_equiv_or_sparse if fuse_depthwise_delta is None else fuse_depthwise_delta
        )
        if character_preprocessing is not None:
            if character_preprocessing not in {"reference", "exact"}:
                raise ValueError("character_preprocessing must be 'reference' or 'exact'")
            self.character_preprocessing = character_preprocessing
        elif is_equiv_or_sparse:
            self.character_preprocessing = "exact"
        else:
            self.character_preprocessing = "reference"

        if gate_logit_threshold is not _UNSET:
            self.gate_logit_threshold = (
                None if gate_logit_threshold is None else float(gate_logit_threshold)
            )
        else:
            self.gate_logit_threshold = (
                DEFAULT_SPARSE_CHAR_GATE_THRESHOLD if mode == "sparse" else None
            )

        wrapper = self.__dict__.get("model")
        pt_model = getattr(wrapper, "model", None) if wrapper is not None else None
        char_head = getattr(pt_model, "character_head", None) if pt_model is not None else None
        if char_head is not None:
            if not hasattr(char_head, "_default_project_tokens_first"):
                char_head._default_project_tokens_first = bool(char_head.project_tokens_first)
            char_head.fuse_depthwise_delta = self.fuse_depthwise_delta
            char_head.gate_logit_threshold = self.gate_logit_threshold
            if project_tokens_first is not None:
                char_head.project_tokens_first = bool(project_tokens_first)
            elif is_equiv_or_sparse:
                char_head.project_tokens_first = True
            else:
                char_head.project_tokens_first = bool(char_head._default_project_tokens_first)
        return self

    def __getattr__(self, name):
        assert hasattr(self, "model")
        return getattr(self.model, name)

    def get_threshold(self, language: str | None = None, domain: str | None = None) -> float:
        """Return the best available development-fitted sentence threshold.

        Calibration is opt-in: :meth:`split` keeps the released checkpoint's
        backward-compatible default unless callers pass this value explicitly.
        """
        if self.use_lora:
            return 0.5
        config = getattr(self.__dict__.get("model"), "config", None)
        calibration_source = getattr(config, "sat_export_source_checkpoint", str(self.model_name_or_model))
        calibrated = calibrated_threshold_for_checkpoint(
            calibration_source,
            language=language if language is not None else self.language,
            domain=domain,
        )
        if calibrated is not None:
            return calibrated
        exported_threshold = getattr(config, "sat_export_default_threshold", None)
        if exported_threshold is not None:
            return float(exported_threshold)
        released = default_threshold_for_checkpoint(str(self.model_name_or_model))
        return released if released is not None else DEFAULT_SENTENCE_THRESHOLD

    def predict_proba(
        self,
        text_or_texts,
        stride=DEFAULT_STRIDE,
        block_size: int = 512,
        batch_size=32,
        pad_last_batch: bool = False,
        weighting: Literal["uniform", "hat"] = "uniform",
        remove_whitespace_before_inference: bool = False,
        outer_batch_size=1000,
        return_paragraph_probabilities=False,
        verbose: bool = False,
        lazy: bool = False,
    ):
        return resolve_text_input(
            text_or_texts,
            lambda texts: self._predict_proba(
                texts,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                weighting=weighting,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                return_paragraph_probabilities=return_paragraph_probabilities,
                verbose=verbose,
            ),
            lazy=lazy,
        )

    def _predict_proba(
        self,
        texts,
        stride: int,
        block_size: int,
        batch_size: int,
        pad_last_batch: bool,
        weighting: Literal["uniform", "hat"],
        remove_whitespace_before_inference: bool,
        outer_batch_size: int,
        return_paragraph_probabilities: bool,
        verbose: bool,
    ):
        # Length-preserving, so every downstream offset (probabilities, spans, decoding
        # against the caller's original text) is unaffected.
        line_breaks = self.__dict__.get("input_line_breaks", "preserve")
        if line_breaks != "preserve":
            texts = [normalize_input_line_breaks(text, line_breaks) for text in texts]

        use_char_head = bool(getattr(self.model.config, "use_character_head", False))
        is_subword_proba = not use_char_head and self.model.config.model_type not in CHAR_MODEL_TYPES
        can_fast_subword_proba = (
            is_subword_proba
            and getattr(self.model.config, "language_adapter", "off") != "on"
            and self.tokenizer is not None
            and not verbose
        )
        def extract_logits(input_texts):
            if can_fast_subword_proba:
                outer_batch_logits, encoded_batch = extract_subword_logits_fast(
                    input_texts,
                    self.model,
                    self.tokenizer,
                    stride=stride,
                    max_block_size=block_size,
                    batch_size=batch_size,
                    pad_last_batch=pad_last_batch,
                    weighting=weighting,
                    max_content_block=max_content_block_size(self.model.config),
                    trim_subword_padding=bool(getattr(self, "trim_subword_padding", False)),
                    pack_sequences=bool(getattr(self, "pack_sequences", False)),
                )
                return [
                    project_subword_proba(
                        len(text),
                        encoded_batch.token_arrays[i],
                        outer_batch_logits[i],
                        encoded_batch.offsets_or_encoding(i),
                        self.special_tokens,
                    )[:, None]
                    for i, text in enumerate(input_texts)
                ]
            outer_batch_logits, _, _, tokenizer_output = extract(
                input_texts,
                self.model,
                stride=stride,
                max_block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                weighting=weighting,
                verbose=verbose,
                tokenizer=self.tokenizer,
                character_preprocessing=getattr(self, "character_preprocessing", "reference"),
                trim_padding=getattr(self, "trim_padding", True),
            )
            if use_char_head:
                return [
                    logits[: len(text)]
                    for text, logits in zip(input_texts, outer_batch_logits)
                ]
            if is_subword_proba:
                return [
                    project_subword_proba(
                        len(text),
                        tokenizer_output["input_ids"][i],
                        outer_batch_logits[i],
                        tokenizer_output["offset_mapping"][i],
                        self.special_tokens,
                    )[:, None]
                    for i, text in enumerate(input_texts)
                ]
            return [
                token_to_char_probs(
                    text,
                    tokenizer_output["input_ids"][i],
                    outer_batch_logits[i],
                    self.special_tokens,
                    tokenizer_output["offset_mapping"][i],
                )
                for i, text in enumerate(input_texts)
            ]

        def probability_fn(logits):
            if is_subword_proba:
                probabilities = logits[:, 0]
                return probabilities, probabilities
            with np.errstate(over="ignore"):
                probabilities = sigmoid(logits[:, Constants.NEWLINE_INDEX])
            return probabilities, probabilities

        yield from iter_probabilities(
            texts,
            outer_batch_size=outer_batch_size,
            remove_whitespace_before_inference=remove_whitespace_before_inference,
            extract_logits=extract_logits,
            probability_fn=probability_fn,
            return_paragraph_probabilities=return_paragraph_probabilities,
            outputs_are_probabilities=is_subword_proba,
        )

    def segment(
        self,
        text_or_texts,
        threshold: float = None,  # ignored when max_length is set
        stride=DEFAULT_STRIDE,
        block_size: int = 512,
        batch_size=32,
        pad_last_batch: bool = False,
        weighting: Literal["uniform", "hat"] = "uniform",
        remove_whitespace_before_inference: bool = False,
        outer_batch_size=1000,
        paragraph_threshold: float = 0.5,
        strip_whitespace: bool = False,
        do_paragraph_segmentation: bool = False,
        split_on_input_newlines: bool = True,  # only applies when max_length is not set
        treat_newline_as_space=None,  # Deprecated
        verbose: bool = False,
        min_length: int = 1,
        max_length: int = None,  # when set, segments may contain newlines; use ''.join(segments)
        prior_type: str = "uniform",
        prior_kwargs: dict = None,
        algorithm: str = "viterbi",
        use_negative_evidence: bool = True,
        lazy: bool = False,
    ):
        """Return structured sentence text, spans, and boundary probabilities."""
        if treat_newline_as_space is not None:
            warnings.warn(
                "treat_newline_as_space is deprecated and will be removed in a future release. "
                "Use split_on_input_newlines with inverse bools instead.",
                DeprecationWarning,
            )
            split_on_input_newlines = not treat_newline_as_space

        validate_segmentation_options(
            threshold=threshold,
            min_length=min_length,
            max_length=max_length,
            prior_type=prior_type,
            algorithm=algorithm,
            split_on_input_newlines=split_on_input_newlines,
        )

        return resolve_text_input(
            text_or_texts,
            lambda texts: self._segment(
                texts,
                threshold=threshold,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                weighting=weighting,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                paragraph_threshold=paragraph_threshold,
                strip_whitespace=strip_whitespace,
                do_paragraph_segmentation=do_paragraph_segmentation,
                split_on_input_newlines=split_on_input_newlines,
                verbose=verbose,
                min_length=min_length,
                max_length=max_length,
                prior_type=prior_type,
                prior_kwargs=prior_kwargs,
                algorithm=algorithm,
                use_negative_evidence=use_negative_evidence,
            ),
            lazy=lazy,
        )

    def _resolve_sentence_threshold(self, threshold: float | None) -> float:
        if threshold is not None:
            return threshold
        exported_threshold = getattr(self.model.config, "sat_export_default_threshold", None)
        if exported_threshold is not None and not self.use_lora:
            return float(exported_threshold)
        model_str = self.model_name_or_model
        if not isinstance(model_str, str):
            warnings.warn(
                f"get_default_threshold received non-string argument: {type(model_str)}. Using base default."
            )
            return DEFAULT_SENTENCE_THRESHOLD
        if self.use_lora:
            return 0.5
        resolved = default_threshold_for_checkpoint(model_str)
        if resolved is None:
            warnings.warn(
                f"{model_str!r} is not a recognised sat-* checkpoint name, so the operating point "
                f"falls back to {DEFAULT_SENTENCE_THRESHOLD}. If this is a fine-tune or a renamed "
                "checkpoint, pass threshold= explicitly.",
                stacklevel=3,
            )
            return DEFAULT_SENTENCE_THRESHOLD
        return resolved

    def split(
        self,
        text_or_texts,
        threshold: float = None,
        stride=DEFAULT_STRIDE,
        block_size: int = 512,
        batch_size=32,
        pad_last_batch: bool = False,
        weighting: Literal["uniform", "hat"] = "uniform",
        remove_whitespace_before_inference: bool = False,
        outer_batch_size=1000,
        paragraph_threshold: float = 0.5,
        strip_whitespace: bool = False,
        do_paragraph_segmentation: bool = False,
        split_on_input_newlines: bool = True,
        treat_newline_as_space=None,
        verbose: bool = False,
        min_length: int = 1,
        max_length: int = None,
        prior_type: str = "uniform",
        prior_kwargs: dict = None,
        algorithm: str = "viterbi",
        use_negative_evidence: bool = True,
        lazy: bool = False,
    ):
        """Return sentence strings."""
        if treat_newline_as_space is not None:
            warnings.warn(
                "treat_newline_as_space is deprecated and will be removed in a future release. "
                "Use split_on_input_newlines with inverse bools instead.",
                DeprecationWarning,
            )
            split_on_input_newlines = not treat_newline_as_space

        validate_segmentation_options(
            threshold=threshold,
            min_length=min_length,
            max_length=max_length,
            prior_type=prior_type,
            algorithm=algorithm,
            split_on_input_newlines=split_on_input_newlines,
        )

        return resolve_text_input(
            text_or_texts,
            lambda texts: self._split(
                texts,
                threshold=threshold,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                weighting=weighting,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                paragraph_threshold=paragraph_threshold,
                strip_whitespace=strip_whitespace,
                do_paragraph_segmentation=do_paragraph_segmentation,
                split_on_input_newlines=split_on_input_newlines,
                verbose=verbose,
                min_length=min_length,
                max_length=max_length,
                prior_type=prior_type,
                prior_kwargs=prior_kwargs,
                algorithm=algorithm,
                use_negative_evidence=use_negative_evidence,
            ),
            lazy=lazy,
        )

    def _split(
        self,
        texts,
        threshold: float | None,
        stride: int,
        block_size: int,
        batch_size: int,
        pad_last_batch: bool,
        weighting: Literal["uniform", "hat"],
        paragraph_threshold: float,
        remove_whitespace_before_inference: bool,
        outer_batch_size: int,
        do_paragraph_segmentation: bool,
        split_on_input_newlines: bool,
        min_length: int,
        max_length: int | None,
        strip_whitespace: bool,
        verbose: bool,
        prior_type: str,
        prior_kwargs: dict | None,
        algorithm: str,
        use_negative_evidence: bool = True,
    ):
        sentence_threshold = self._resolve_sentence_threshold(threshold)

        can_fast_subword_split = (
            not do_paragraph_segmentation
            and max_length is None
            and min_length <= 1
            and not remove_whitespace_before_inference
            and not getattr(self.model.config, "use_character_head", False)
            and self.model.config.model_type not in CHAR_MODEL_TYPES
            and getattr(self.model.config, "language_adapter", "off") != "on"
            and self.tokenizer is not None
            and not verbose
        )

        if can_fast_subword_split:
            z_min = exact_float16_logit_threshold(sentence_threshold)
            line_breaks = self.__dict__.get("input_line_breaks", "preserve")
            max_content_block = max_content_block_size(self.model.config)
            n_outer = math.ceil(len(texts) / outer_batch_size)

            for outer_idx in range(n_outer):
                start = outer_idx * outer_batch_size
                end = min((outer_idx + 1) * outer_batch_size, len(texts))
                outer_orig = texts[start:end]
                outer_in = (
                    [normalize_input_line_breaks(t, line_breaks) for t in outer_orig]
                    if line_breaks != "preserve"
                    else outer_orig
                )
                nonempty_map = [i for i, t in enumerate(outer_in) if t.strip()]
                nonempty_texts = [outer_in[i] for i in nonempty_map]

                if nonempty_texts:
                    stitched_logits, encoded_batch = extract_subword_logits_fast(
                        nonempty_texts,
                        self.model,
                        self.tokenizer,
                        stride=stride,
                        max_block_size=block_size,
                        batch_size=batch_size,
                        pad_last_batch=pad_last_batch,
                        weighting=weighting,
                        max_content_block=max_content_block,
                        trim_subword_padding=bool(getattr(self, "trim_subword_padding", False)),
                        pack_sequences=bool(getattr(self, "pack_sequences", False)),
                    )
                else:
                    stitched_logits = []
                    encoded_batch = None

                nonempty_cursor = 0
                for doc_i, orig_text in enumerate(outer_orig):
                    if split_on_input_newlines is False:
                        warnings.warn(
                            "split_on_input_newlines=False disables forced splitting at input newlines. "
                            "This option does not replace newlines with spaces; the model may still "
                            "predict boundaries around them. Returned segments may contain input newlines.",
                            UserWarning,
                            stacklevel=2,
                        )
                    if doc_i in nonempty_map and encoded_batch is not None:
                        tok_logits = stitched_logits[nonempty_cursor]
                        off_or_enc = encoded_batch.offsets_or_encoding(nonempty_cursor)
                        nonempty_cursor += 1
                    else:
                        tok_logits = np.empty((0, 1), dtype=np.float16)
                        off_or_enc = []

                    yield decode_subword_split(
                        orig_text,
                        tok_logits,
                        off_or_enc,
                        z_min=z_min,
                        strip_whitespace=strip_whitespace,
                        split_on_input_newlines=bool(split_on_input_newlines),
                    )
            return

        for text, probs in zip(
            texts,
            self.predict_proba(
                texts,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                weighting=weighting,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                return_paragraph_probabilities=do_paragraph_segmentation,
                verbose=verbose,
                lazy=True,
            ),
        ):
            decoded = decode_probabilities(
                text,
                probs,
                sentence_threshold=sentence_threshold,
                paragraph_threshold=paragraph_threshold,
                strip_whitespace=strip_whitespace,
                do_paragraph_segmentation=do_paragraph_segmentation,
                split_on_input_newlines=split_on_input_newlines,
                min_length=min_length,
                max_length=max_length,
                prior_type=prior_type,
                prior_kwargs=prior_kwargs,
                algorithm=algorithm,
                use_negative_evidence=use_negative_evidence,
                language=self.language,
            )
            yield decoded.paragraphs if decoded.paragraphs is not None else decoded.sentences

    def _segment(
        self,
        texts,
        threshold: float | None,
        stride: int,
        block_size: int,
        batch_size: int,
        pad_last_batch: bool,
        weighting: Literal["uniform", "hat"],
        paragraph_threshold: float,
        remove_whitespace_before_inference: bool,
        outer_batch_size: int,
        do_paragraph_segmentation: bool,
        split_on_input_newlines: bool,
        min_length: int,
        max_length: int | None,
        strip_whitespace: bool,
        verbose: bool,
        prior_type: str,
        prior_kwargs: dict | None,
        algorithm: str,
        use_negative_evidence: bool = True,
    ):
        sentence_threshold = self._resolve_sentence_threshold(threshold)

        for text, probs in zip(
            texts,
            self.predict_proba(
                texts,
                stride=stride,
                block_size=block_size,
                batch_size=batch_size,
                pad_last_batch=pad_last_batch,
                weighting=weighting,
                remove_whitespace_before_inference=remove_whitespace_before_inference,
                outer_batch_size=outer_batch_size,
                return_paragraph_probabilities=do_paragraph_segmentation,
                verbose=verbose,
                lazy=True,
            ),
        ):
            decoded = decode_probabilities(
                text,
                probs,
                sentence_threshold=sentence_threshold,
                paragraph_threshold=paragraph_threshold,
                strip_whitespace=strip_whitespace,
                do_paragraph_segmentation=do_paragraph_segmentation,
                split_on_input_newlines=split_on_input_newlines,
                min_length=min_length,
                max_length=max_length,
                prior_type=prior_type,
                prior_kwargs=prior_kwargs,
                algorithm=algorithm,
                use_negative_evidence=use_negative_evidence,
                language=self.language,
            )

            if decoded.paragraphs is not None:
                sentence_probs, newline_probs = probs
                flat_sentences = decoded.sentences
                flat_spans = sentence_spans(text, flat_sentences)
                paragraph_spans = []
                span_offset = 0
                for paragraph in decoded.paragraphs:
                    paragraph_spans.append(flat_spans[span_offset : span_offset + len(paragraph)])
                    span_offset += len(paragraph)
                yield Segmentation(
                    text=text,
                    sentences=flat_sentences,
                    spans=flat_spans,
                    probabilities=sentence_probs,
                    confidences=boundary_confidences(sentence_probs, flat_spans),
                    paragraphs=decoded.paragraphs,
                    paragraph_spans=paragraph_spans,
                    paragraph_probabilities=newline_probs,
                )
            else:
                spans = sentence_spans(text, decoded.sentences)
                yield Segmentation(
                    text=text,
                    sentences=decoded.sentences,
                    spans=spans,
                    probabilities=probs,
                    confidences=boundary_confidences(probs, spans),
                )


def __getattr__(name: str):
    if name == "WtP":
        try:
            from wtpsplit.legacy import WtP
        except ModuleNotFoundError as error:
            if error.name in {"skops", "sklearn"}:
                raise ImportError(
                    "WtP is a legacy API and requires optional dependencies. "
                    "Install them with `pip install 'wtpsplit[legacy]'`."
                ) from error
            raise
        return WtP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
