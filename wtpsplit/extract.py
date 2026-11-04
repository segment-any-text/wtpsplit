import math
import sys
import logging
from contextlib import nullcontext
from typing import Literal

import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from wtpsplit._host_kernels import pack_subword_windows, stitch_subword_logits
from wtpsplit.char_head import build_char_inputs
from wtpsplit.utils import Constants, hash_encode

logger = logging.getLogger(__name__)

# Character-level backbones (WtP). Everything else is subword-based, so new subword
# backbones work here without edits; previously this was an "xlm" substring test that
# silently excluded any non-XLM backbone.
CHAR_MODEL_TYPES = {"bert-char", "la-canine"}

# Only used when `extract` is called without a tokenizer; `SaT` and `WtP` always pass one.
DEFAULT_TOKENIZERS = {
    "xlm-token": "facebookAI/xlm-roberta-base",
    "xlm-roberta": "facebookAI/xlm-roberta-base",
}

# XLM-R reserves two position slots (its padding_idx offset), leaving 512 usable, minus
# CLS and SEP. Kept as a literal so the XLM-R path behaves exactly as before.
XLM_MAX_CONTENT_BLOCK_SIZE = 510


def outputs_character_logits(config) -> bool:
    """Whether model logits already align one-to-one with input characters."""
    return config.model_type in CHAR_MODEL_TYPES or getattr(config, "use_character_head", False)


def max_content_block_size(config) -> int:
    """Longest chunk of real tokens that fits, leaving room for CLS and SEP."""
    if config.model_type in ("xlm-token", "xlm-roberta"):
        return XLM_MAX_CONTENT_BLOCK_SIZE
    max_positions = getattr(config, "max_position_embeddings", None)
    if not max_positions:
        return XLM_MAX_CONTENT_BLOCK_SIZE
    return max_positions - 2


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
        self.input_types = {value.name: value.type for value in ort_session.get_inputs()}

    def __getattr__(self, name):
        assert hasattr(self, "ort_session")
        return getattr(self.ort_session, name)

    def __call__(self, input_ids, attention_mask, **character_inputs):
        # Existing released graphs use FP16 masks; new FP32 exports do not.
        # Character features can arrive as CPU tensors or NumPy arrays.
        values = dict(input_ids=input_ids, attention_mask=attention_mask, **character_inputs)
        dtypes = {"tensor(int64)": np.int64, "tensor(int32)": np.int32,
                  "tensor(float16)": np.float16, "tensor(float)": np.float32,
                  "tensor(bool)": np.bool_}
        feeds = {}
        for name, dtype in self.input_types.items():
            if name not in values or values[name] is None:
                raise ValueError(f"Missing ONNX input: {name}")
            if dtype not in dtypes:
                raise ValueError(f"Unsupported ONNX input type for {name}: {dtype}")
            value = values[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            feeds[name] = np.asarray(value, dtype=dtypes[dtype])
        logits = self.ort_session.run(["logits"], feeds)[0]
        return {"logits": logits}


class PyTorchWrapper:
    def __init__(self, model, *, autocast_device=None, autocast_dtype=None, autocast_enabled=True, output_float32=False, pinned_inputs=None):
        self.model = model
        self.config = model.config
        self.autocast_device = autocast_device
        self.autocast_dtype = autocast_dtype
        self.autocast_enabled = autocast_enabled
        self.output_float32 = output_float32
        self.pinned_inputs = pinned_inputs

    def __getattr__(self, name):
        assert hasattr(self, "model")
        return getattr(self.model, name)

    def __call__(
        self,
        attention_mask,
        hashed_ids=None,
        language_ids=None,
        input_ids=None,
        char_to_token=None,
        char_is_token_final=None,
        char_position_in_token=None,
        char_hashes=None,
        char_mask=None,
    ):
        try:
            import torch
        except ImportError:
            raise ImportError("`torch` must be installed to use PyTorch models!")

        def as_tensor(value, name):
            if value is None:
                return None
            if self.pinned_inputs is not None:
                return self.pinned_inputs.tensor(name, value, self.model.device)
            if torch.is_tensor(value):
                return value.to(self.model.device)
            return torch.from_numpy(value).to(self.model.device)

        model_kwargs = {"attention_mask": as_tensor(attention_mask, "attention_mask")}
        optional_inputs = {
            "input_ids": input_ids,
            "hashed_ids": hashed_ids,
            "language_ids": language_ids,
            "char_to_token": char_to_token,
            "char_is_token_final": char_is_token_final,
            "char_position_in_token": char_position_in_token,
            "char_hashes": char_hashes,
            "char_mask": char_mask,
        }
        model_kwargs.update(
            {
                name: as_tensor(value, name)
                for name, value in optional_inputs.items()
                if value is not None
            }
        )

        context = (
            torch.autocast(self.autocast_device, dtype=self.autocast_dtype, enabled=self.autocast_enabled)
            if self.autocast_device is not None else nullcontext()
        )
        with torch.no_grad(), context:
            logits = self.model(**model_kwargs)["logits"]
        if self.output_float32:
            logits = logits.float()
        logits = logits.cpu().numpy()

        return {"logits": logits}


def _window_weights(length: int, weighting: Literal["uniform", "hat"]) -> np.ndarray:
    if weighting == "uniform" or length <= 1:
        return np.ones(length, dtype=np.float32)
    if weighting == "hat":
        x = np.linspace(-(1 - 1 / length), 1 - 1 / length, length, dtype=np.float32)
        return 1 - np.abs(x)
    raise ValueError(f"Unknown weighting scheme: {weighting!r}")


def extract_character_head(
    texts,
    model,
    *,
    stride,
    max_block_size,
    batch_size,
    pad_last_batch=False,
    weighting: Literal["uniform", "hat"] = "uniform",
    verbose=False,
    tokenizer,
    trim_padding=True,
):
    """Extract and stitch character logits from overlapping token windows."""
    import torch

    if stride < 1:
        raise ValueError(f"`stride` must be at least 1, got {stride}.")

    encoded = tokenizer(texts, return_offsets_mapping=True, verbose=False, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    offset_mappings = encoded["offset_mapping"]
    content_block_size = min(max_block_size, max_content_block_size(model.config))
    if content_block_size < 1:
        raise ValueError(f"`max_block_size` leaves no room for content tokens: {max_block_size}.")

    windows = []
    for text_index, (text, ids, offsets) in enumerate(zip(texts, token_ids, offset_mappings)):
        if not ids:
            continue
        for token_start in range(0, len(ids), stride):
            token_end = token_start + content_block_size
            done = False
            if token_end >= len(ids):
                token_end = len(ids)
                token_start = max(token_end - content_block_size, 0)
                done = True

            char_start = 0 if token_start == 0 else offsets[token_start][0]
            char_end = len(text) if token_end == len(ids) else offsets[token_end - 1][1]
            relative_offsets = [
                (max(start - char_start, 0), min(end - char_start, char_end - char_start))
                for start, end in offsets[token_start:token_end]
            ]
            windows.append(
                {
                    "text_index": text_index,
                    "char_start": char_start,
                    "char_end": char_end,
                    "text": text[char_start:char_end],
                    "offsets": relative_offsets,
                    "input_ids": ids[token_start:token_end],
                }
            )
            if done:
                break

    all_logits = [np.zeros((len(text), model.config.num_labels), dtype=np.float32) for text in texts]
    all_counts = [np.zeros(len(text), dtype=np.float32) for text in texts]
    num_batches = math.ceil(len(windows) / batch_size)

    for batch_index in tqdm(range(num_batches), disable=not verbose):
        start = batch_index * batch_size
        real_windows = windows[start : start + batch_size]
        current_batch_size = len(real_windows)
        if not real_windows:
            continue

        token_width = content_block_size + 2
        if trim_padding:
            token_width = max(len(window["input_ids"]) + 2 for window in real_windows)
        batch_input_ids = np.full(
            (current_batch_size, token_width),
            tokenizer.pad_token_id,
            dtype=np.int64,
        )
        batch_attention_mask = np.zeros((current_batch_size, token_width), dtype=np.float32)
        for row, window in enumerate(real_windows):
            ids = [tokenizer.cls_token_id, *window["input_ids"], tokenizer.sep_token_id]
            batch_input_ids[row, : len(ids)] = ids
            batch_attention_mask[row, : len(ids)] = 1

        char_inputs = build_char_inputs(
            [window["text"] for window in real_windows],
            [window["offsets"] for window in real_windows],
        )
        char_to_token = char_inputs.char_to_token + 1

        if current_batch_size < batch_size and pad_last_batch:
            missing = batch_size - current_batch_size
            batch_input_ids = np.pad(
                batch_input_ids,
                ((0, missing), (0, 0)),
                constant_values=tokenizer.pad_token_id,
            )
            batch_attention_mask = np.pad(batch_attention_mask, ((0, missing), (0, 0)))

            def pad_rows(tensor):
                return torch.nn.functional.pad(tensor, (0,) * (2 * (tensor.ndim - 1)) + (0, missing))

            char_to_token = pad_rows(char_to_token)
            char_is_token_final = pad_rows(char_inputs.is_token_final)
            char_position_in_token = pad_rows(char_inputs.position_in_token)
            char_hashes = pad_rows(char_inputs.hashes)
            char_mask = pad_rows(char_inputs.mask)
        else:
            char_is_token_final = char_inputs.is_token_final
            char_position_in_token = char_inputs.position_in_token
            char_hashes = char_inputs.hashes
            char_mask = char_inputs.mask

        logits = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            char_to_token=char_to_token,
            char_is_token_final=char_is_token_final,
            char_position_in_token=char_position_in_token,
            char_hashes=char_hashes,
            char_mask=char_mask,
        )["logits"][:current_batch_size]

        for row, window in enumerate(real_windows):
            text_index = window["text_index"]
            char_start, char_end = window["char_start"], window["char_end"]
            length = char_end - char_start
            weights = _window_weights(length, weighting)
            all_logits[text_index][char_start:char_end] += logits[row, :length] * weights[:, None]
            all_counts[text_index][char_start:char_end] += weights

    for index, text in enumerate(texts):
        if not text:
            continue
        covered = all_counts[index] > 0
        all_logits[index][covered] /= all_counts[index][covered, None]
        all_logits[index][~covered] = -12.0

    return (
        [logits.astype(np.float16) for logits in all_logits],
        None,
        tokenizer,
        encoded,
    )


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
    character_preprocessing="reference",
    trim_padding=True,
):
    """
    Computes logits for the given batch of texts by:
        1. slicing the texts into chunks of size `block_size`.
        2. passing every chunk through the model forward.
        3. stitching predictings back together by averaging chunk logits.

    ad 1.: text is sliced into partially overlapping chunks by moving forward by a `stride` parameter (think conv1d).
    """
    if getattr(model.config, "use_character_head", False):
        if tokenizer is None:
            default_tokenizer = DEFAULT_TOKENIZERS.get(model.config.model_type)
            if default_tokenizer is None:
                raise ValueError(
                    f"No default tokenizer is known for model type {model.config.model_type!r}. "
                    "Pass `tokenizer=` explicitly."
                )
            tokenizer = AutoTokenizer.from_pretrained(default_tokenizer)
        if character_preprocessing == "exact":
            from wtpsplit._char_preprocessing import extract_character_head_exact
            character_extractor = extract_character_head_exact
        elif character_preprocessing == "reference":
            character_extractor = extract_character_head
        else:
            raise ValueError(f"Unknown character preprocessing: {character_preprocessing!r}")
        return character_extractor(
            batch_of_texts,
            model,
            stride=stride,
            max_block_size=max_block_size,
            batch_size=batch_size,
            pad_last_batch=pad_last_batch,
            weighting=weighting,
            verbose=verbose,
            tokenizer=tokenizer,
            trim_padding=trim_padding,
        )

    if model.config.model_type not in CHAR_MODEL_TYPES:
        use_subwords = True
        if tokenizer is None:
            default_tokenizer = DEFAULT_TOKENIZERS.get(model.config.model_type)
            if default_tokenizer is None:
                raise ValueError(
                    f"No default tokenizer is known for model type {model.config.model_type!r}. "
                    "Pass `tokenizer=` explicitly."
                )
            tokenizer = AutoTokenizer.from_pretrained(default_tokenizer)
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
    downsampling_rate = getattr(model.config, "downsampling_rate", 1)

    if use_subwords:
        eff_max_block = min(max_block_size, max_content_block_size(model.config))
        input_ids, attention_mask, locs, block_size = pack_subword_windows(
            batch_of_texts,
            block_size=eff_max_block,
            stride=stride,
            downsampling_rate=downsampling_rate,
            cls_id=cls_token_id,
            sep_id=sep_token_id,
        )
        num_chunks = len(locs)
    else:
        # reduce block size if possible
        block_size = min(max_block_size, max(text_lengths))
        block_size = math.ceil(block_size / downsampling_rate) * downsampling_rate
        num_chunks = sum(math.ceil(max(length - block_size, 0) / stride) + 1 for length in text_lengths)
        input_hashes = np.zeros((num_chunks, block_size, model.config.num_hash_functions), dtype=np.int64)
        attention_mask = np.zeros((num_chunks, block_size), dtype=np.float32)
        locs = np.zeros((num_chunks, 3), dtype=np.int32)

        # this is equivalent to (but faster than) np.array([ord(c) for c in "".join(batch_of_texts)])
        codec = "utf-32-le" if sys.byteorder == "little" else "utf-32-be"
        ordinals = np.frombuffer(bytearray("".join(batch_of_texts), encoding=codec), dtype=np.int32)
        # hash encode all ids
        flat_hashed_ids = hash_encode(
            ordinals, num_hashes=model.config.num_hash_functions, num_buckets=model.config.num_hash_buckets
        )
        offset = 0
        current_chunk = 0
        for i in range(len(batch_of_texts)):
            for j in range(0, text_lengths[i], stride):
                start, end = j, j + block_size
                done = False
                if end >= text_lengths[i]:
                    end = text_lengths[i]
                    start = max(end - block_size, 0)
                    done = True
                input_hashes[current_chunk, : end - start] = flat_hashed_ids[offset + start : offset + end]
                attention_mask[current_chunk, : end - start] = 1
                locs[current_chunk, :] = [i, start, end]
                current_chunk += 1
                if done:
                    break
            offset += text_lengths[i]
        assert current_chunk == num_chunks

    n_batches = math.ceil(len(attention_mask) / batch_size)

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

    if use_subwords:
        raw_batches = []
        for batch_idx in tqdm(range(n_batches), disable=not verbose):
            start, end = batch_idx * batch_size, min(len(attention_mask), (batch_idx + 1) * batch_size)
            batch_input_ids = input_ids[start:end]
            batch_attention_mask = attention_mask[start:end]
            if len(batch_attention_mask) < batch_size and pad_last_batch:
                n_missing = batch_size - len(batch_attention_mask)
                batch_input_ids = np.pad(batch_input_ids, ((0, n_missing), (0, 0)), constant_values=pad_token_id)
                batch_attention_mask = np.pad(batch_attention_mask, ((0, n_missing), (0, 0)))
            kwargs = {"language_ids": language_ids[: len(batch_attention_mask)]} if uses_lang_adapters else {}
            kwargs["input_ids"] = batch_input_ids
            logits = model(attention_mask=batch_attention_mask, **kwargs)["logits"]
            raw_batches.append(logits[: end - start])

        all_logits = stitch_subword_logits(
            raw_batches,
            locs,
            text_lengths,
            block_size=block_size,
            num_labels=model.config.num_labels,
            weighting=weighting,
            strip_cls_sep=True,
        )
        return all_logits, offset_mapping, tokenizer, tokens

    # containers for the final logits (character-level WtP path)
    all_logits = [
        np.zeros(
            (length, model.config.num_labels),
            dtype=np.float16,
        )
        for length in text_lengths
    ]
    all_counts = [np.zeros(length, dtype=np.float16) for length in text_lengths]

    if weighting == "uniform":
        weights = np.ones(block_size, dtype=np.float16)
    elif weighting == "hat":
        x = np.linspace(-(1 - 1 / block_size), 1 - 1 / block_size, block_size, dtype=np.float16)
        weights = 1 - np.abs(x)

    for batch_idx in tqdm(range(n_batches), disable=not verbose):
        start, end = batch_idx * batch_size, min(len(attention_mask), (batch_idx + 1) * batch_size)
        batch_input_hashes = input_hashes[start:end]
        batch_attention_mask = attention_mask[start:end]

        if len(batch_attention_mask) < batch_size and pad_last_batch:
            n_missing = batch_size - len(batch_attention_mask)
            batch_input_hashes = np.pad(batch_input_hashes, ((0, n_missing), (0, 0), (0, 0)))
            batch_attention_mask = np.pad(batch_attention_mask, ((0, n_missing), (0, 0)))

        kwargs = {"language_ids": language_ids[: len(batch_attention_mask)]} if uses_lang_adapters else {}
        kwargs["hashed_ids"] = batch_input_hashes

        logits = model(
            attention_mask=batch_attention_mask,
            **kwargs,
        )["logits"]

        for i in range(start, end):
            original_idx, start_char_idx, end_char_idx = locs[i]
            n = end_char_idx - start_char_idx
            all_logits[original_idx][start_char_idx:end_char_idx] += weights[:n, np.newaxis] * logits[i - start, :n]
            all_counts[original_idx][start_char_idx:end_char_idx] += weights[:n]

    all_logits = [(logits / counts[:, None]).astype(np.float16) for logits, counts in zip(all_logits, all_counts)]

    return (
        all_logits,
        None,
        None,
        None,
    )
