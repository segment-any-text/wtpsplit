import math
import sys
import logging
from contextlib import nullcontext
from typing import Any, Literal

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
    "modernbert-token": "jhu-clsp/mmBERT-base",
    "modernbert": "jhu-clsp/mmBERT-base",
}

# XLM-R reserves two position slots (its padding_idx offset), leaving 512 usable, minus
# CLS and SEP. Kept as a literal so the XLM-R path behaves exactly as before.
XLM_MAX_CONTENT_BLOCK_SIZE = 510
MAX_DEFERRED_D2H_BYTES = 64 * 1024 * 1024
MAX_VARLEN_SDPA_TOKENS = 2048


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
    def __init__(self, config: Any, ort_session: Any) -> None:
        self.config = config
        self.ort_session = ort_session
        inputs = ort_session.get_inputs()
        self.input_types = {value.name: value.type for value in inputs}
        self.input_ranks = {
            value.name: (len(value.shape) if getattr(value, "shape", None) is not None else None)
            for value in inputs
        }

    def __getattr__(self, name: str) -> Any:
        assert hasattr(self, "ort_session")
        return getattr(self.ort_session, name)

    def __call__(self, input_ids: Any, attention_mask: Any, **character_inputs: Any) -> dict[str, Any]:
        # Existing released graphs use FP16 masks; new FP32 exports do not.
        # Character features can arrive as CPU tensors or NumPy arrays.
        values = dict(input_ids=input_ids, attention_mask=attention_mask, **character_inputs)
        dtypes = {
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(float16)": np.float16,
            "tensor(float)": np.float32,
            "tensor(bool)": np.bool_,
        }
        feeds = {}
        for name, dtype in self.input_types.items():
            if name not in values or values[name] is None:
                raise ValueError(f"Missing ONNX input: {name}")
            if dtype not in dtypes:
                raise ValueError(f"Unsupported ONNX input type for {name}: {dtype}")
            target_dtype = dtypes[dtype]
            value = values[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray) and value.dtype == target_dtype and value.flags.c_contiguous:
                feeds[name] = value
            else:
                feeds[name] = np.asarray(value, dtype=target_dtype)
        logits = self.ort_session.run(["logits"], feeds)[0]
        return {"logits": logits}


class PyTorchWrapper:
    def __init__(
        self,
        model: Any,
        *,
        autocast_device: str | None = None,
        autocast_dtype: Any = None,
        autocast_enabled: bool = True,
        output_float32: bool = False,
        pinned_inputs: Any = None,
    ) -> None:
        self.model = model
        self.config = model.config
        self.autocast_device = autocast_device
        self.autocast_dtype = autocast_dtype
        self.autocast_enabled = autocast_enabled
        self.output_float32 = output_float32
        self.pinned_inputs = pinned_inputs

    def __getattr__(self, name: str) -> Any:
        assert hasattr(self, "model")
        return getattr(self.model, name)

    def _as_device_tensor(self, value: Any, name: str) -> Any:
        import torch

        if value is None:
            return None
        if self.pinned_inputs is not None:
            return self.pinned_inputs.tensor(name, value, self.model.device)
        if torch.is_tensor(value):
            return value.to(self.model.device)
        return torch.from_numpy(value).to(self.model.device)

    def encode_tokens(
        self,
        attention_mask: Any,
        input_ids: Any = None,
        language_ids: Any = None,
    ) -> tuple[Any, Any]:
        """Run Stage 1 subword backbone and token classifier on device without `char_*` inputs."""
        try:
            import torch
        except ImportError:
            raise ImportError("`torch` must be installed to use PyTorch models!")

        attn_all_ones = (
            attention_mask is not None
            and isinstance(attention_mask, np.ndarray)
            and attention_mask.ndim == 2
            and attention_mask.size > 0
            and bool(attention_mask[:, -1].all())
            and bool((attention_mask == 1).all())
        )
        attn_tensor = self._as_device_tensor(attention_mask, "attention_mask")
        if attn_tensor is not None and attn_all_ones:
            attn_tensor._sat_all_ones = True
        model_kwargs: dict[str, Any] = {"attention_mask": attn_tensor}
        if input_ids is not None:
            model_kwargs["input_ids"] = self._as_device_tensor(input_ids, "input_ids")
        if language_ids is not None:
            model_kwargs["language_ids"] = self._as_device_tensor(language_ids, "language_ids")

        context = (
            torch.autocast(self.autocast_device, dtype=self.autocast_dtype, enabled=self.autocast_enabled)
            if self.autocast_device is not None
            else nullcontext()
        )
        with torch.no_grad(), context:
            token_hidden, token_logits = self.model.encode_tokens(**model_kwargs)
        return token_hidden, token_logits

    def forward_varlen(
        self,
        batch_input_ids: np.ndarray,
        batch_win_lens: np.ndarray,
    ) -> Any:
        """Pack non-pad tokens across ``batch_input_ids`` into ``[1, N_valid]``, run ``encode_tokens_varlen``, and unpack content logits."""
        import torch

        num_rows = len(batch_win_lens)
        max_content = int(batch_win_lens.max()) if num_rows > 0 else 0
        num_labels = int(self.model.config.num_labels)
        device = self.model.device
        out_dtype = torch.float32 if self.output_float32 else getattr(self.model, "dtype", torch.float32)
        if num_rows == 0 or max_content == 0:
            return torch.zeros((num_rows, max_content, num_labels), device=device, dtype=out_dtype)

        pad_idx = int(getattr(getattr(getattr(self.model, "roberta", None), "embeddings", None), "padding_idx", 1))
        attn_impl = getattr(self.model.config, "_attn_implementation", None)
        max_pack_tokens = None if attn_impl == "flex_attention" else MAX_VARLEN_SDPA_TOKENS

        context = (
            torch.autocast(self.autocast_device, dtype=self.autocast_dtype, enabled=self.autocast_enabled)
            if self.autocast_device is not None
            else nullcontext()
        )
        out: Any = None

        def _build_doc_runs(chunk_offsets: list[tuple[int, int, int]]) -> list[tuple[int, int, int, int]]:
            runs: list[tuple[int, int, int, int]] = []
            for _r_idx, s_off, e_off in chunk_offsets:
                seq_len = (e_off - s_off) + 2
                t_start = s_off - 1
                t_end = e_off + 1
                if runs and runs[-1][2] == seq_len and runs[-1][1] == t_start:
                    prev_s, _prev_e, prev_len, prev_cnt = runs[-1]
                    runs[-1] = (prev_s, t_end, prev_len, prev_cnt + 1)
                else:
                    runs.append((t_start, t_end, seq_len, 1))
            return runs

        def _run_pack_chunk(
            ids_parts: list[np.ndarray],
            pos_parts: list[np.ndarray],
            doc_parts: list[np.ndarray],
            chunk_offsets: list[tuple[int, int, int]],
        ) -> None:
            nonlocal out
            packed_ids_t = torch.from_numpy(np.concatenate(ids_parts, axis=0)).to(
                device=device, dtype=torch.long
            ).unsqueeze(0)
            packed_pos_t = torch.from_numpy(np.concatenate(pos_parts, axis=0)).to(
                device=device, dtype=torch.long
            ).unsqueeze(0)
            doc_ids_t = torch.from_numpy(np.concatenate(doc_parts, axis=0)).to(
                device=device, dtype=torch.int32
            )
            doc_runs = _build_doc_runs(chunk_offsets)
            with torch.no_grad(), context:
                try:
                    _, packed_logits = self.model.encode_tokens_varlen(
                        packed_ids_t,
                        packed_pos_t,
                        doc_ids_t,
                        doc_runs=doc_runs,
                    )
                except TypeError:
                    _, packed_logits = self.model.encode_tokens_varlen(
                        packed_ids_t,
                        packed_pos_t,
                        doc_ids_t,
                    )
            if self.output_float32:
                packed_logits = packed_logits.float()
            packed_2d = packed_logits[0]
            if out is None:
                out = torch.zeros((num_rows, max_content, num_labels), device=device, dtype=packed_2d.dtype)
            for r_idx, s_off, e_off in chunk_offsets:
                n_c = e_off - s_off
                if n_c > 0:
                    out[r_idx, :n_c] = packed_2d[s_off:e_off]

        packed_ids_parts: list[np.ndarray] = []
        packed_pos_parts: list[np.ndarray] = []
        doc_id_parts: list[np.ndarray] = []
        offsets: list[tuple[int, int, int]] = []
        cursor = 0
        for row_idx in range(num_rows):
            n_content = int(batch_win_lens[row_idx])
            seq_len = n_content + 2
            if max_pack_tokens is not None and cursor > 0 and (cursor + seq_len) > max_pack_tokens:
                _run_pack_chunk(packed_ids_parts, packed_pos_parts, doc_id_parts, offsets)
                packed_ids_parts.clear()
                packed_pos_parts.clear()
                doc_id_parts.clear()
                offsets.clear()
                cursor = 0
            row_ids = batch_input_ids[row_idx, :seq_len]
            packed_ids_parts.append(row_ids)
            mask = (row_ids != pad_idx).astype(np.int64)
            row_pos = np.cumsum(mask, dtype=np.int64) * mask + pad_idx
            packed_pos_parts.append(row_pos)
            doc_id_parts.append(np.full(seq_len, row_idx, dtype=np.int32))
            offsets.append((row_idx, cursor + 1, cursor + 1 + n_content))
            cursor += seq_len

        if cursor > 0:
            _run_pack_chunk(packed_ids_parts, packed_pos_parts, doc_id_parts, offsets)
        if out is None:
            return torch.zeros((num_rows, max_content, num_labels), device=device, dtype=out_dtype)
        if getattr(self, "_compiled", False):
            out = out.clone()
        return out

    def encode_tokens_varlen_batch(
        self,
        batch_input_ids: np.ndarray,
        batch_win_lens: np.ndarray,
    ) -> tuple[Any, Any]:
        """Pack non-pad tokens across ``batch_input_ids`` into ``[1, N_valid]``, run ``encode_tokens_varlen``, and unpack ``(token_hidden, token_logits)``."""
        import torch

        num_rows = len(batch_win_lens)
        max_seq = (int(batch_win_lens.max()) + 2) if num_rows > 0 else 0
        hidden_size = int(self.model.config.hidden_size)
        num_labels = int(self.model.config.num_labels)
        device = self.model.device
        dtype = getattr(self.model, "dtype", torch.float32)
        if num_rows == 0 or max_seq == 0:
            return (
                torch.zeros((num_rows, max_seq, hidden_size), device=device, dtype=dtype),
                torch.zeros((num_rows, max_seq, num_labels), device=device, dtype=dtype),
            )

        pad_idx = int(getattr(getattr(getattr(self.model, "roberta", None), "embeddings", None), "padding_idx", 1))
        packed_ids_parts: list[np.ndarray] = []
        packed_pos_parts: list[np.ndarray] = []
        doc_id_parts: list[np.ndarray] = []
        runs: list[tuple[int, int, int, int]] = []
        row_spans: list[tuple[int, int, int]] = []
        cursor = 0
        for row_idx in range(num_rows):
            seq_len = int(batch_win_lens[row_idx]) + 2
            row_ids = batch_input_ids[row_idx, :seq_len]
            packed_ids_parts.append(row_ids)
            mask = (row_ids != pad_idx).astype(np.int64)
            row_pos = np.cumsum(mask, dtype=np.int64) * mask + pad_idx
            packed_pos_parts.append(row_pos)
            doc_id_parts.append(np.full(seq_len, row_idx, dtype=np.int32))
            t_start = cursor
            t_end = cursor + seq_len
            row_spans.append((row_idx, t_start, t_end))
            if runs and runs[-1][2] == seq_len and runs[-1][1] == t_start:
                prev_s, _prev_e, prev_len, prev_cnt = runs[-1]
                runs[-1] = (prev_s, t_end, prev_len, prev_cnt + 1)
            else:
                runs.append((t_start, t_end, seq_len, 1))
            cursor = t_end

        packed_ids_t = torch.from_numpy(np.concatenate(packed_ids_parts, axis=0)).to(
            device=device, dtype=torch.long
        ).unsqueeze(0)
        packed_pos_t = torch.from_numpy(np.concatenate(packed_pos_parts, axis=0)).to(
            device=device, dtype=torch.long
        ).unsqueeze(0)
        doc_ids_t = torch.from_numpy(np.concatenate(doc_id_parts, axis=0)).to(
            device=device, dtype=torch.int32
        )

        context = (
            torch.autocast(self.autocast_device, dtype=self.autocast_dtype, enabled=self.autocast_enabled)
            if self.autocast_device is not None
            else nullcontext()
        )
        with torch.no_grad(), context:
            packed_hid, packed_log = self.model.encode_tokens_varlen(
                packed_ids_t,
                packed_pos_t,
                doc_ids_t,
                doc_runs=runs,
            )
        hid_2d = packed_hid[0]
        log_2d = packed_log[0]
        out_hid = torch.zeros((num_rows, max_seq, hidden_size), device=device, dtype=hid_2d.dtype)
        out_log = torch.zeros((num_rows, max_seq, num_labels), device=device, dtype=log_2d.dtype)
        for r_idx, t_s, t_e in row_spans:
            seq_l = t_e - t_s
            out_hid[r_idx, :seq_l] = hid_2d[t_s:t_e]
            out_log[r_idx, :seq_l] = log_2d[t_s:t_e]
        return out_hid, out_log

    def run_character_head(
        self,
        token_hidden: Any,
        token_logits: Any,
        char_inputs: Any,
        *,
        gate_logit_threshold: float | None = None,
    ) -> Any:
        """Run Stage 2 `CharacterResolutionHead` on device (with optional token-gate bypass)."""
        import torch
        from wtpsplit.char_head import SUPPRESSION_BIAS, CharInputs

        char_head = self.model.character_head
        eff_gate = (
            gate_logit_threshold
            if gate_logit_threshold is not None
            else getattr(char_head, "gate_logit_threshold", None)
        )
        context = (
            torch.autocast(self.autocast_device, dtype=self.autocast_dtype, enabled=self.autocast_enabled)
            if self.autocast_device is not None
            else nullcontext()
        )
        with torch.no_grad(), context:
            if eff_gate is not None and not char_head.training:
                tok_pass = token_logits.max(dim=-1).values > eff_gate
                if not bool(tok_pass.any()):
                    c2t = self._as_device_tensor(char_inputs.char_to_token, "char_to_token")
                    fin = self._as_device_tensor(char_inputs.is_token_final, "char_is_token_final")
                    msk = self._as_device_tensor(char_inputs.mask, "char_mask")
                    if hasattr(char_head, "_gather_base_logits_with_suppression"):
                        logits = char_head._gather_base_logits_with_suppression(token_logits, c2t, fin, msk)
                    else:
                        index = c2t.unsqueeze(-1)
                        gathered_logits = token_logits.gather(1, index.expand(-1, -1, token_logits.size(-1)))
                        logits = gathered_logits + char_head.suppression(fin)
                        logits = logits.masked_fill(msk.unsqueeze(-1) == 0, -SUPPRESSION_BIAS)
                    if self.output_float32:
                        logits = logits.float()
                    if getattr(self, "_compiled", False):
                        logits = logits.clone()
                    return logits

            dev_inputs = CharInputs(
                char_to_token=self._as_device_tensor(char_inputs.char_to_token, "char_to_token"),
                is_token_final=self._as_device_tensor(char_inputs.is_token_final, "char_is_token_final"),
                position_in_token=self._as_device_tensor(char_inputs.position_in_token, "char_position_in_token"),
                hashes=self._as_device_tensor(char_inputs.hashes, "char_hashes"),
                mask=self._as_device_tensor(char_inputs.mask, "char_mask"),
            )
            logits = char_head(
                token_hidden,
                token_logits,
                dev_inputs,
                gate_logit_threshold=eff_gate,
            )
        if self.output_float32:
            logits = logits.float()
        if getattr(self, "_compiled", False):
            logits = logits.clone()
        return logits

    def forward_tensor(
        self,
        attention_mask: Any,
        hashed_ids: Any = None,
        language_ids: Any = None,
        input_ids: Any = None,
        char_to_token: Any = None,
        char_is_token_final: Any = None,
        char_position_in_token: Any = None,
        char_hashes: Any = None,
        char_mask: Any = None,
        *,
        strip_cls_sep: bool = False,
    ) -> Any:
        try:
            import torch
        except ImportError:
            raise ImportError("`torch` must be installed to use PyTorch models!")

        attn_all_ones = (
            attention_mask is not None
            and isinstance(attention_mask, np.ndarray)
            and attention_mask.ndim == 2
            and attention_mask.size > 0
            and bool(attention_mask[:, -1].all())
            and bool((attention_mask == 1).all())
        )
        attn_tensor = self._as_device_tensor(attention_mask, "attention_mask")
        if attn_tensor is not None and attn_all_ones:
            attn_tensor._sat_all_ones = True
        model_kwargs = {"attention_mask": attn_tensor}
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
                name: self._as_device_tensor(value, name)
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
        if strip_cls_sep:
            logits = logits[:, 1:-1, :]
        if self.output_float32:
            logits = logits.float()
        if getattr(self, "_compiled", False) or strip_cls_sep:
            logits = logits.clone()
        return logits

    def __call__(
        self,
        attention_mask: Any,
        hashed_ids: Any = None,
        language_ids: Any = None,
        input_ids: Any = None,
        char_to_token: Any = None,
        char_is_token_final: Any = None,
        char_position_in_token: Any = None,
        char_hashes: Any = None,
        char_mask: Any = None,
    ) -> dict[str, np.ndarray]:
        logits = (
            self.forward_tensor(
                attention_mask,
                hashed_ids=hashed_ids,
                language_ids=language_ids,
                input_ids=input_ids,
                char_to_token=char_to_token,
                char_is_token_final=char_is_token_final,
                char_position_in_token=char_position_in_token,
                char_hashes=char_hashes,
                char_mask=char_mask,
            )
            .cpu()
            .numpy()
        )

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
    trim_subword_padding: bool = False,
    pack_sequences: bool = False,
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
            if extract_character_head_exact.__module__ == "wtpsplit._char_preprocessing":
                return extract_character_head_exact(
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
                    trim_subword_padding=trim_subword_padding,
                    pack_sequences=pack_sequences,
                )
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
        win_lens = None
        if trim_subword_padding and len(locs) > 0:
            win_lens = locs[:, 2] - locs[:, 1]
            if len(locs) > batch_size and not bool(np.all(win_lens[:-1] <= win_lens[1:])):
                order = np.argsort(win_lens, kind="stable")
                input_ids = input_ids[order]
                attention_mask = attention_mask[order]
                locs = locs[order]
                win_lens = win_lens[order]

        raw_batches = []
        can_defer_d2h = (
            type(model).__name__ == "PyTorchWrapper"
            and hasattr(model, "forward_tensor")
            and getattr(model, "pinned_inputs", None) is None
            and "__call__" not in getattr(model, "__dict__", {})
        )
        if can_defer_d2h:
            import torch

            tensor_batches = []
            accum_d2h_bytes = 0

            def _flush_extract_batches() -> None:
                nonlocal accum_d2h_bytes
                if not tensor_batches:
                    return
                if len(tensor_batches) == 1:
                    raw_batches.append(tensor_batches[0].cpu().numpy())
                else:
                    w0 = tensor_batches[0].shape[1]
                    if all(t.shape[1] == w0 for t in tensor_batches):
                        raw_batches.append(torch.cat(tensor_batches, dim=0).cpu().numpy())
                    else:
                        for t in tensor_batches:
                            raw_batches.append(t.cpu().numpy())
                tensor_batches.clear()
                accum_d2h_bytes = 0

            for batch_idx in tqdm(range(n_batches), disable=not verbose):
                start, end = batch_idx * batch_size, min(len(attention_mask), (batch_idx + 1) * batch_size)
                if win_lens is not None:
                    max_win_len = int(win_lens[start:end].max())
                    eff_win_block = (
                        math.ceil(max(max_win_len, 1) / downsampling_rate) * downsampling_rate
                        if downsampling_rate > 1
                        else max_win_len
                    )
                    batch_width = min(block_size, eff_win_block) + 2
                    batch_input_ids = input_ids[start:end, :batch_width]
                    batch_attention_mask = attention_mask[start:end, :batch_width]
                else:
                    batch_input_ids = input_ids[start:end]
                    batch_attention_mask = attention_mask[start:end]
                if len(batch_attention_mask) < batch_size and pad_last_batch:
                    n_missing = batch_size - len(batch_attention_mask)
                    batch_input_ids = np.pad(batch_input_ids, ((0, n_missing), (0, 0)), constant_values=pad_token_id)
                    batch_attention_mask = np.pad(batch_attention_mask, ((0, n_missing), (0, 0)))
                kwargs = {"language_ids": language_ids[: len(batch_attention_mask)]} if uses_lang_adapters else {}
                kwargs["input_ids"] = batch_input_ids
                logits_t = model.forward_tensor(
                    attention_mask=batch_attention_mask,
                    strip_cls_sep=True,
                    **kwargs,
                )[: end - start]
                tensor_batches.append(logits_t)
                accum_d2h_bytes += int(logits_t.numel() * logits_t.element_size())
                del logits_t
                if accum_d2h_bytes >= MAX_DEFERRED_D2H_BYTES:
                    _flush_extract_batches()
            _flush_extract_batches()
            if len(raw_batches) > 1 and all(
                b.ndim == 3 and b.shape[1:] == raw_batches[0].shape[1:] for b in raw_batches
            ):
                raw_batches = [np.concatenate(raw_batches, axis=0)]
            strip_cls_sep = False
        else:
            for batch_idx in tqdm(range(n_batches), disable=not verbose):
                start, end = batch_idx * batch_size, min(len(attention_mask), (batch_idx + 1) * batch_size)
                if win_lens is not None:
                    max_win_len = int(win_lens[start:end].max())
                    eff_win_block = (
                        math.ceil(max(max_win_len, 1) / downsampling_rate) * downsampling_rate
                        if downsampling_rate > 1
                        else max_win_len
                    )
                    batch_width = min(block_size, eff_win_block) + 2
                    batch_input_ids = input_ids[start:end, :batch_width]
                    batch_attention_mask = attention_mask[start:end, :batch_width]
                else:
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
            if len(raw_batches) > 1 and all(
                b.ndim == 3 and b.shape[1:] == raw_batches[0].shape[1:] for b in raw_batches
            ):
                raw_batches = [np.concatenate(raw_batches, axis=0)]
            strip_cls_sep = True

        all_logits = stitch_subword_logits(
            raw_batches,
            locs,
            text_lengths,
            block_size=block_size,
            num_labels=model.config.num_labels,
            weighting=weighting,
            strip_cls_sep=strip_cls_sep,
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
