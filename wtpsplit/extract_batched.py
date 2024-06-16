import math
import sys
import logging

import numpy as np
from transformers import AutoTokenizer
from tokenizers import AddedToken

from wtpsplit.utils import Constants, hash_encode
from wtpsplit.extract import ORTWrapper

logger = logging.getLogger(__name__)


def extract_batched(
    batch_of_texts,
    model,
    block_size,
    batch_size,
    lang_code=None,
    pad_last_batch=False,
    verbose=False,
):
    """
    Like extract.py, but does not split the input into chunks of block_size.
    Instead, it processes the input in batches. So each input batch must be smaller than block_size.
    """
    if "xlm" in model.config.model_type:
        use_subwords = True
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
        tokens = tokenizer(
            batch_of_texts,
            return_offsets_mapping=True,
            verbose=False,
            padding="max_length",  # pad to max length (TPUs need fixed length inputs)
            return_tensors="np",
            truncation=True,
            max_length=block_size,
        )
        attention_mask = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        offset_mapping = [offset[1:-1] for offset in tokens.pop("offset_mapping")]
        pad_token_id = tokenizer.pad_token_id
        text_lengths = [len(text) for text in input_ids]
    else:
        pad_token_id = 0
        use_subwords = False
        input_ids = batch_of_texts.copy()
        text_lengths = [len(text) for text in input_ids]
        offset_mapping = text_lengths

    # when using ChLMs, it can be that the input is too long --> simply truncate
    if max(text_lengths) > block_size:
        # truncate
        longer_than_block_size = [i for i, length in enumerate(text_lengths) if length > block_size]
        if verbose:
            logger.info(f"Truncating {len(longer_than_block_size)} texts longer than block_size={block_size}")
        for i in longer_than_block_size:
            input_ids[i] = input_ids[i][:block_size]

    # make sure block_size is a multiple of downsampling rate
    downsampling_rate = getattr(model.config, "downsampling_rate", 1)
    block_size = math.ceil(block_size / downsampling_rate) * downsampling_rate

    if not use_subwords:
        codec = "utf-32-le" if sys.byteorder == "little" else "utf-32-be"
        hashed_ids = []
        attention_mask = np.ones((len(input_ids), block_size), dtype=int)
        for i, text in enumerate(input_ids):
            ord = np.frombuffer(bytearray(text, encoding=codec), dtype=np.int32)
            # pad
            if len(ord) < block_size:
                # mask out padding
                attention_mask[i, len(ord) :] = 0
                # pad to max length, i.e., block size (no truncation due to TPUs)
                ord = np.pad(ord, (0, block_size - len(ord)))
            hashed_ids.append(hash_encode(ord, model.config.num_hash_functions, model.config.num_hash_buckets))
        hashed_ids = np.array(hashed_ids)

    uses_lang_adapters = getattr(model.config, "language_adapter", "off") == "on"
    if uses_lang_adapters:
        if lang_code is None:
            raise ValueError("Please specify a `lang_code` when using a model with language adapters.")

        if isinstance(model, ORTWrapper):
            raise ValueError("Language adapters are not supported in ONNX models.")

        language_ids = np.array(
            [Constants.LANG_CODE_TO_INDEX[lang_code]] * batch_size,
            dtype=int,
        )
    else:
        language_ids = None

    if len(attention_mask) < batch_size and pad_last_batch:
        n_missing = batch_size - len(attention_mask)

        if not use_subwords:
            hashed_ids = np.pad(hashed_ids, ((0, n_missing), (0, 0), (0, 0)))
        else:
            # Pad with the specific pad_token_id for the tokenizer
            input_ids = np.pad(input_ids, ((0, n_missing), (0, 0)), constant_values=pad_token_id)
        attention_mask = np.pad(attention_mask, ((0, n_missing), (0, 0)))
    else:
        n_missing = 0

    kwargs = {"language_ids": language_ids} if uses_lang_adapters else {}

    if use_subwords and model.config.model_type == "xlm-roberta":
        # TODO: generalize
        import torch
        with torch.no_grad():
            logits = model.model(
                input_ids=torch.from_numpy(input_ids).to(model.model.device),
                attention_mask=torch.from_numpy(attention_mask).to(model.model.device),
                **kwargs,
            )["logits"].cpu().numpy()
    else:
        logits = model(
            input_ids=input_ids if use_subwords else None,
            hashed_ids=None if use_subwords else hashed_ids,
            attention_mask=attention_mask,
            **kwargs,
        )["logits"]
    if use_subwords:
        logits = logits[:, 1:-1, :]  # remove CLS and SEP tokens

    return (
        logits[: batch_size - n_missing],
        offset_mapping,
        tokenizer if use_subwords else None,
    )
