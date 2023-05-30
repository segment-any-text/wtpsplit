import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from wtpsplit.utils import Constants


def extract(
    batch_of_tokens,
    model,
    stride,
    block_size,
    batch_size,
    lang_code=None,
    pad_last_batch=True,
    verbose=True,
):
    input_ids = []
    attention_masks = []
    position_ids = []
    locs = defaultdict(lambda: {})

    only_windowing = all(len(tokens) >= block_size for tokens in batch_of_tokens)
    downsampling_rate = getattr(model.config, "downsampling_rate", 1)

    current_ids = []
    current_position_ids = torch.zeros(block_size, dtype=torch.long)
    if only_windowing:
        current_attention_mask = torch.zeros((block_size), dtype=torch.float32)
        position_ids = None
    else:
        current_attention_mask = torch.zeros((block_size, block_size), dtype=torch.float32)
    for idx_in_batch, tokens in enumerate(batch_of_tokens):
        if len(tokens) >= block_size:
            # finish packing
            if len(current_ids) > 0:
                assert not only_windowing

                current_ids.extend([0] * (block_size - len(current_ids)))
                input_ids.append(torch.tensor(current_ids, dtype=torch.long))
                attention_masks.append(current_attention_mask)
                position_ids.append(current_position_ids)

                current_ids = []
                current_attention_mask = torch.zeros_like(current_attention_mask)
                current_position_ids = torch.zeros(block_size, dtype=torch.long)

            # start windowing
            i = 0
            while i + block_size < len(tokens):
                locs[len(input_ids)][(0, block_size)] = (
                    idx_in_batch,
                    i,
                    i + block_size,
                )
                input_ids.append(torch.tensor(tokens[i : i + block_size], dtype=torch.long))
                attention_masks.append(torch.ones_like(current_attention_mask))
                if position_ids is not None:
                    position_ids.append(torch.arange(block_size, dtype=torch.long))

                i += stride

            # last window
            locs[len(input_ids)][(0, block_size)] = (
                idx_in_batch,
                len(tokens) - block_size,
                len(tokens),
            )
            input_ids.append(torch.tensor(tokens[-block_size:], dtype=torch.long))
            attention_masks.append(torch.ones_like(current_attention_mask))
            if position_ids is not None:
                position_ids.append(torch.arange(block_size, dtype=torch.long))
        else:
            assert not only_windowing

            padding = downsampling_rate - (len(tokens) % downsampling_rate)
            if padding == downsampling_rate:
                padding = 0

            padded_tokens = tokens + [0] * padding

            if len(current_ids) + len(padded_tokens) <= block_size:
                locs[len(input_ids)][(len(current_ids), len(current_ids) + len(tokens))] = (
                    idx_in_batch,
                    0,
                    len(tokens),
                )
                current_attention_mask[
                    len(current_ids) : len(current_ids) + len(tokens),
                    len(current_ids) : len(current_ids) + len(tokens),
                ] = 1
                current_position_ids[len(current_ids) : len(current_ids) + len(tokens)] = torch.arange(len(tokens))
                current_ids.extend(padded_tokens)
            else:
                current_ids.extend([0] * (block_size - len(current_ids)))
                input_ids.append(torch.tensor(current_ids, dtype=torch.long))
                attention_masks.append(current_attention_mask)
                position_ids.append(current_position_ids)

                current_ids = padded_tokens
                locs[len(input_ids)][(0, len(tokens))] = (idx_in_batch, 0, len(tokens))
                current_attention_mask = torch.zeros((block_size, block_size), dtype=torch.float32)
                current_attention_mask[
                    : len(tokens),
                    : len(tokens),
                ] = 1
                current_position_ids = torch.zeros(block_size, dtype=torch.long)
                current_position_ids[: len(tokens)] = torch.arange(len(tokens))

    if len(current_ids) > 0:
        assert not only_windowing

        current_ids.extend([0] * (block_size - len(current_ids)))
        input_ids.append(torch.tensor(current_ids, dtype=torch.long))
        attention_masks.append(current_attention_mask)
        position_ids.append(current_position_ids)

    input_ids = torch.stack(input_ids, 0)
    attention_masks = torch.stack(attention_masks, 0)
    if position_ids is not None:
        position_ids = torch.stack(position_ids, 0)
    n_batches = math.ceil(len(input_ids) / batch_size)

    all_logits = [
        torch.zeros(
            (
                len(tokens),
                model.config.num_labels,
            ),
            dtype=torch.float16,
        )
        for tokens in batch_of_tokens
    ]
    all_counts = [torch.zeros((len(tokens)), dtype=torch.int16) for tokens in batch_of_tokens]

    uses_lang_adapters = getattr(model.config, "language_adapter", "off") == "on"
    if uses_lang_adapters:
        if lang_code is None:
            raise ValueError("Please specify a `lang_code` when using a model with language adapters.")

        language_ids = torch.tensor(
            [Constants.LANG_CODE_TO_INDEX[lang_code]] * batch_size,
            dtype=torch.long,
            device=model.device,
        )
    else:
        language_ids = None

    for batch_idx in tqdm(range(n_batches), disable=not verbose):
        start, end = batch_idx * batch_size, min(len(input_ids), (batch_idx + 1) * batch_size)

        batch_input_ids = input_ids[start:end]
        batch_attention_mask = attention_masks[start:end]
        batch_position_ids = position_ids[start:end] if position_ids is not None else None

        if len(batch_input_ids) < batch_size and pad_last_batch:
            n_missing = batch_size - len(batch_input_ids)

            batch_input_ids = F.pad(batch_input_ids, (0, 0, 0, n_missing))
            if only_windowing:  # 1d attention mask
                batch_attention_mask = F.pad(batch_attention_mask, (0, 0, 0, n_missing))
            else:
                batch_attention_mask = F.pad(batch_attention_mask, (0, 0, 0, 0, 0, n_missing))
            batch_position_ids = (
                F.pad(batch_position_ids, (0, 0, 0, n_missing)) if batch_position_ids is not None else None
            )

        # no_grad bc inference_mode does not work on TPUs
        with torch.no_grad():
            kwargs = {"language_ids": language_ids[: len(batch_input_ids)]} if uses_lang_adapters else {}

            out = model(
                input_ids=batch_input_ids.to(model.device),
                attention_mask=batch_attention_mask.to(model.device),
                position_ids=batch_position_ids.to(model.device) if batch_position_ids is not None else None,
                **kwargs,
            )
            logits = out["logits"].cpu()

        for i in range(start, end):
            for key, value in locs[i].items():
                all_logits[value[0]][value[1] : value[2]] += logits[i - start, key[0] : key[1]]
                all_counts[value[0]][value[1] : value[2]] += 1

    all_logits = [logits / counts.unsqueeze(-1) for logits, counts in zip(all_logits, all_counts)]

    return all_logits
