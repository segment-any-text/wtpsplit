import math
import sys
import logging
from typing import Literal, Optional

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from wtpsplit.aitune_integration import apply_aitune, pop_aitune_kwargs
from wtpsplit.utils import Constants, hash_encode

logger = logging.getLogger(__name__)

_INDUCTOR_ALIASES = {"inductor", "torchinductor", "torch_inductor", "default"}
_AITUNE_ALIASES = {"aitune", "ai_tune"}


def normalize_optimize_backend(backend: Optional[str]) -> str:
    """Map public ``optimize(backend=...)`` names onto ``inductor`` or ``aitune``."""
    key = (backend or "inductor").lower().replace("-", "_")
    if key in _INDUCTOR_ALIASES:
        return "inductor"
    if key in _AITUNE_ALIASES:
        return "aitune"
    if key in {"none", "off", "eager"}:
        raise ValueError(
            f"backend={backend!r} is not an optimized backend. "
            "Omit optimize() for eager PyTorch, or pass backend='inductor' / 'aitune'."
        )
    return key


def _module_device(module):
    """Device of a plain, ``torch.compile``d, or AITune-wrapped module."""
    device = getattr(module, "device", None)
    if getattr(device, "type", None):
        return device
    try:
        return next(module.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Cannot infer model device: the module has no parameters.") from exc


def logits_from_model_output(output):
    """Read logits from a Hugging Face dict/output or from a compiled tuple."""
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("Model forward returned an empty sequence; expected logits.")
        return output[0]
    logits = getattr(output, "logits", None)
    if logits is not None:
        return logits
    try:
        return output["logits"]
    except Exception as exc:
        raise TypeError(
            "Model forward did not return logits. Expected a dict-like output or a tuple whose first item is logits."
        ) from exc


class BertCharORTWrapper:
    def __init__(self, config, ort_session):
        self.config = config
        self.ort_session = ort_session

    def __getattr__(self, name):
        assert hasattr(self, "ort_session")
        return getattr(self.ort_session, name)

    def __call__(self, hashed_ids, attention_mask):
        logits = self.ort_session.run(
            ["logits"],
            {
                "attention_mask": attention_mask.astype(np.float16),  # ORT expects fp16 mask
                "hashed_ids": hashed_ids,
            },
        )[0]

        return {"logits": logits}


class SaTORTWrapper:
    def __init__(self, config, ort_session):
        self.config = config
        self.ort_session = ort_session

    def __getattr__(self, name):
        assert hasattr(self, "ort_session")
        return getattr(self.ort_session, name)

    def __call__(self, input_ids, attention_mask):
        logits = self.ort_session.run(
            ["logits"],
            {
                "attention_mask": attention_mask.astype(np.float16),
                "input_ids": input_ids.astype(np.int64),
            },
        )[0]

        return {"logits": logits}


class PyTorchWrapper:
    def __init__(self, model):
        self.model = model
        self.config = model.config
        self._torch_compiled = False

    def __getattr__(self, name):
        assert hasattr(self, "model")
        return getattr(self.model, name)

    def optimize(
        self,
        *,
        backend: str = "inductor",
        mode: Optional[str] = None,
        fullgraph: bool = False,
        dynamic: bool = True,
        **compile_kwargs,
    ):
        """Compile the underlying Hugging Face model with :func:`torch.compile` (TorchInductor by default).

        Call after moving the model to the target device and changing dtype (e.g. ``half()``), so the
        compiled graph matches inference. A second call is ignored.

        Chunk length and the last batch size vary across ``split`` calls, so ``dynamic=True`` is the
        default. ``mode="reduce-overhead"`` enables CUDA graphs and fights those varying shapes; pass
        it only when every forward uses the same batch and sequence length. ``None`` lets Inductor
        pick its default mode.

        Args:
            backend: ``"inductor"`` for TorchInductor (aliases: ``"torchinductor"``, ``"torch_inductor"``),
                or ``"aitune"`` for NVIDIA AITune (CUDA only; requires ``pip install wtpsplit[aitune]``).
            mode: Compilation mode (``"default"``, ``"reduce-overhead"``, ``"max-autotune"``,
                ``"max-autotune-no-cudagraphs"``). ``None`` uses the backend default.
            fullgraph: Passed to :func:`torch.compile` and to AITune's Inductor backend.
            dynamic: If ``True`` (default), allow varying sequence lengths across chunks.
            **compile_kwargs: For ``inductor``, passed to :func:`torch.compile`. For ``aitune``, optional:
                ``aitune_strategy`` (``"first_wins"``, ``"inductor_only"``, ``"highest_throughput"``),
                ``aitune_batch_sizes``, ``aitune_max_batches``, ``aitune_calibration``, ``aitune_dry_run``.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("`torch` must be installed to use optimize().") from None

        if self._torch_compiled:
            logger.warning("optimize() was already applied; keeping the existing compiled model.")
            return self

        key = normalize_optimize_backend(backend)

        if key == "aitune":
            aitune_kwargs = pop_aitune_kwargs(compile_kwargs)
            if compile_kwargs:
                raise TypeError(f"Unexpected keyword arguments for backend='aitune': {sorted(compile_kwargs)}")
            self.model.eval()
            self.model = apply_aitune(
                self.model, mode=mode, fullgraph=fullgraph, dynamic=dynamic, **aitune_kwargs
            )
            self._torch_compiled = True
            return self

        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile requires PyTorch 2.0 or newer.")

        self.model.eval()
        compile_kwargs = dict(compile_kwargs)
        if mode is not None:
            compile_kwargs["mode"] = mode

        self.model = torch.compile(
            self.model,
            backend=key,
            fullgraph=fullgraph,
            dynamic=dynamic,
            **compile_kwargs,
        )
        self._torch_compiled = True
        return self

    def __call__(self, attention_mask, hashed_ids=None, language_ids=None, input_ids=None):
        try:
            import torch
        except ImportError:
            raise ImportError("`torch` must be installed to use PyTorch models!")

        # inference_mode: stricter than no_grad(); this wrapper is inference-only.
        with torch.inference_mode():
            device = _module_device(self.model)
            forward_kwargs = {
                "attention_mask": torch.from_numpy(attention_mask).to(device),
            }
            if input_ids is not None:
                forward_kwargs["input_ids"] = torch.from_numpy(input_ids).to(device)
            if hashed_ids is not None:
                forward_kwargs["hashed_ids"] = torch.from_numpy(hashed_ids).to(device)
            if language_ids is not None:
                forward_kwargs["language_ids"] = torch.from_numpy(language_ids).to(device)

            logits = logits_from_model_output(self.model(**forward_kwargs)).detach().cpu().numpy()

        return {"logits": logits}


def extract(
    batch_of_texts,
    model,
    stride,
    max_block_size,
    batch_size,
    lang_code=None,
    pad_last_batch=False,
    weighting: Literal["uniform", "hat"] = "uniform",
    verbose=False,
    tokenizer=None,
):
    """
    Computes logits for the given batch of texts by:
        1. slicing the texts into chunks of size `block_size`.
        2. passing every chunk through the model forward.
        3. stitching predictings back together by averaging chunk logits.

    ad 1.: text is sliced into partially overlapping chunks by moving forward by a `stride` parameter (think conv1d).
    """
    if "xlm" in model.config.model_type:
        use_subwords = True
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "facebookAI/xlm-roberta-base",
            )
        # tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
        tokens = tokenizer(batch_of_texts, return_offsets_mapping=True, verbose=False, add_special_tokens=False)
        # remove CLS and SEP tokens, they are added later anyhow
        # batch_of_texts = [text[1:-1] for text in tokens["input_ids"]]
        batch_of_texts = tokens["input_ids"]
        # offset_mapping = [offset[1:-1] for offset in tokens["offset_mapping"]]
        offset_mapping = tokens["offset_mapping"]
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = 0
        use_subwords = False

    text_lengths = [len(text) for text in batch_of_texts]
    # reduce block size if possible
    block_size = min(max_block_size, max(text_lengths))
    if use_subwords and block_size > 510:
        overflow_length = block_size - 510
        block_size -= overflow_length  # account for CLS and SEP tokens

    # make sure block_size is a multiple of downsampling rate
    downsampling_rate = getattr(model.config, "downsampling_rate", 1)
    block_size = math.ceil(block_size / downsampling_rate) * downsampling_rate

    # total number of forward passes
    num_chunks = sum(math.ceil(max(length - block_size, 0) / stride) + 1 for length in text_lengths)

    # preallocate a buffer for all input hashes & attention masks
    if not use_subwords:
        input_hashes = np.zeros((num_chunks, block_size, model.config.num_hash_functions), dtype=np.int64)
        attention_mask = np.zeros((num_chunks, block_size), dtype=np.float32)
    else:
        input_ids = np.zeros((num_chunks, block_size + 2), dtype=np.int64)
        attention_mask = np.zeros((num_chunks, block_size + 2), dtype=np.float32)

    # locs keep track of the location of every chunk with a 3-tuple (text_idx, char_start, char_end) that indexes
    # back into the batch_of_texts
    locs = np.zeros((num_chunks, 3), dtype=np.int32)

    if not use_subwords:
        # this is equivalent to (but faster than) np.array([ord(c) for c in "".join(batch_of_texts)])
        codec = "utf-32-le" if sys.byteorder == "little" else "utf-32-be"
        ordinals = np.frombuffer(bytearray("".join(batch_of_texts), encoding=codec), dtype=np.int32)
        # hash encode all ids
        flat_hashed_ids = hash_encode(
            ordinals, num_hashes=model.config.num_hash_functions, num_buckets=model.config.num_hash_buckets
        )
    # note that ordinals and flat_hashed_ids have the same length
    offset = 0
    current_chunk = 0

    # create chunks
    for i in range(len(batch_of_texts)):
        for j in range(0, text_lengths[i], stride):
            # for every chunk, assign input hashes, attention mask and loc
            start, end = j, j + block_size
            done = False

            if end >= text_lengths[i]:
                end = text_lengths[i]
                start = max(end - block_size, 0)
                done = True

            if not use_subwords:
                input_hashes[current_chunk, : end - start] = flat_hashed_ids[offset + start : offset + end]
                attention_mask[current_chunk, : end - start] = 1
            else:
                chunk = [cls_token_id] + batch_of_texts[i][start:end] + [sep_token_id]
                input_ids[current_chunk, : len(chunk)] = chunk
                attention_mask[current_chunk, : len(chunk)] = 1

            locs[current_chunk, :] = [i, start, end]
            current_chunk += 1

            if done:
                break

        offset += text_lengths[i]

    assert current_chunk == num_chunks
    n_batches = math.ceil(len(attention_mask) / batch_size)

    # containers for the final logits
    all_logits = [
        np.zeros(
            (length, model.config.num_labels),
            dtype=np.float16,
        )
        for length in text_lengths
    ]
    # container for the number of chunks that any character was part of (to average chunk predictions)
    all_counts = [np.zeros(length, dtype=np.float16) for length in text_lengths]

    uses_lang_adapters = getattr(model.config, "language_adapter", "off") == "on"
    if uses_lang_adapters:
        if lang_code is None:
            raise ValueError("Please specify a `lang_code` when using a model with language adapters.")

        if isinstance(model, BertCharORTWrapper):
            raise ValueError("Language adapters are not supported in ONNX models.")

        language_ids = np.array(
            [Constants.LANG_CODE_TO_INDEX[lang_code]] * batch_size,
            dtype=int,
        )
    else:
        language_ids = None

    # compute weights for the given weighting scheme
    if weighting == "uniform":
        weights = np.ones(block_size, dtype=np.float16)
    elif weighting == "hat":
        x = np.linspace(-(1 - 1 / block_size), 1 - 1 / block_size, block_size, dtype=np.float16)
        weights = 1 - np.abs(x)

    # forward passes through all chunks
    for batch_idx in tqdm(range(n_batches), disable=not verbose):
        start, end = batch_idx * batch_size, min(len(attention_mask), (batch_idx + 1) * batch_size)

        if not use_subwords:
            batch_input_hashes = input_hashes[start:end]
        else:
            batch_input_ids = input_ids[start:end]
        batch_attention_mask = attention_mask[start:end]

        if len(batch_attention_mask) < batch_size and pad_last_batch:
            n_missing = batch_size - len(batch_attention_mask)

            if not use_subwords:
                batch_input_hashes = np.pad(batch_input_hashes, ((0, n_missing), (0, 0), (0, 0)))
            else:
                # Pad with the specific pad_token_id for the tokenizer
                batch_input_ids = np.pad(batch_input_ids, ((0, n_missing), (0, 0)), constant_values=pad_token_id)
            batch_attention_mask = np.pad(batch_attention_mask, ((0, n_missing), (0, 0)))

        kwargs = {"language_ids": language_ids[: len(batch_attention_mask)]} if uses_lang_adapters else {}
        if use_subwords:
            kwargs["input_ids"] = batch_input_ids
        else:
            kwargs["hashed_ids"] = batch_input_hashes

        logits = model(
            attention_mask=batch_attention_mask,
            **kwargs,
        )["logits"]

        if use_subwords:
            logits = logits[:, 1:-1, :]  # remove CLS and SEP tokens

        for i in range(start, end):
            original_idx, start_char_idx, end_char_idx = locs[i]
            n = end_char_idx - start_char_idx
            all_logits[original_idx][start_char_idx:end_char_idx] += weights[:n, np.newaxis] * logits[i - start, :n]
            all_counts[original_idx][start_char_idx:end_char_idx] += weights[:n]

    # so far, logits are summed, so we average them here
    all_logits = [(logits / counts[:, None]).astype(np.float16) for logits, counts in zip(all_logits, all_counts)]

    return (
        all_logits,
        offset_mapping if use_subwords else None,
        tokenizer if use_subwords else None,
        tokens if use_subwords else None,
    )
