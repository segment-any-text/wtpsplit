"""Hub-free tests for chunking and overlap reconstruction."""

from types import SimpleNamespace

import numpy as np
import pytest

from wtpsplit import SaT
from wtpsplit.extract import extract
from wtpsplit.extract_batched import extract_batched


class ConstantCharModel:
    def __init__(self):
        self.config = SimpleNamespace(
            model_type="bert-char",
            num_hash_functions=2,
            num_hash_buckets=128,
            num_labels=1,
            downsampling_rate=1,
            language_adapter="off",
        )
        self.batch_shapes = []

    def __call__(self, hashed_ids, attention_mask):
        self.batch_shapes.append(hashed_ids.shape)
        return {"logits": np.ones((*attention_mask.shape, 1), dtype=np.float32)}


class CharacterHeadTokenizer:
    cls_token_id = 2
    sep_token_id = 3
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        input_ids = []
        offset_mapping = []
        for text in texts:
            offsets = [(index, index + 1) for index, char in enumerate(text) if not char.isspace()]
            input_ids.append([10 + index for index in range(len(offsets))])
            offset_mapping.append(offsets)
        return {"input_ids": input_ids, "offset_mapping": offset_mapping}


class ConstantCharacterHeadModel:
    def __init__(self):
        self.config = SimpleNamespace(
            model_type="modernbert-token",
            use_character_head=True,
            max_position_embeddings=6,
            num_labels=1,
        )
        self.batch_shapes = []

    def __call__(
        self,
        input_ids,
        attention_mask,
        char_to_token,
        char_is_token_final,
        char_position_in_token,
        char_hashes,
        char_mask,
    ):
        self.batch_shapes.append((input_ids.shape, char_mask.shape))
        assert char_to_token.shape == char_mask.shape
        assert char_is_token_final.shape == char_mask.shape
        assert char_position_in_token.shape == char_mask.shape
        assert char_hashes.shape[:2] == char_mask.shape
        return {"logits": np.ones((*char_mask.shape, 1), dtype=np.float32)}


@pytest.mark.parametrize("weighting", ["uniform", "hat"])
def test_overlapping_chunks_reconstruct_constant_logits(weighting):
    model = ConstantCharModel()
    texts = ["abcdefghij", "xyzpqrs"]

    logits, offsets, tokenizer, tokens = extract(
        texts,
        model,
        stride=3,
        max_block_size=4,
        batch_size=2,
        pad_last_batch=True,
        weighting=weighting,
    )

    assert [array.shape for array in logits] == [(10, 1), (7, 1)]
    assert all(np.allclose(array, 1.0) for array in logits)
    assert len(model.batch_shapes) == 3
    assert model.batch_shapes[-1][0] == 2  # padded final batch
    assert offsets is tokenizer is tokens is None


@pytest.mark.parametrize("weighting", ["uniform", "hat"])
def test_character_head_windows_stitch_in_character_space(weighting):
    model = ConstantCharacterHeadModel()
    tokenizer = CharacterHeadTokenizer()
    texts = ["ab cd ef gh", "xy z"]

    logits, offsets, returned_tokenizer, tokens = extract(
        texts,
        model,
        stride=2,
        max_block_size=4,
        batch_size=2,
        pad_last_batch=True,
        weighting=weighting,
        tokenizer=tokenizer,
    )

    assert [array.shape for array in logits] == [(11, 1), (4, 1)]
    assert all(np.allclose(array, 1.0) for array in logits)
    assert model.batch_shapes[-1][0][0] == 2
    assert model.batch_shapes[-1][1][0] == 2
    assert offsets is None
    assert returned_tokenizer is tokenizer
    assert tokens["input_ids"][0]


def test_public_predict_proba_uses_character_logits_directly():
    sat = object.__new__(SaT)
    sat.model = ConstantCharacterHeadModel()
    sat.tokenizer = CharacterHeadTokenizer()
    sat.special_tokens = []

    text = "ab cd ef gh"
    probabilities = sat.predict_proba(
        text,
        stride=2,
        block_size=4,
        batch_size=2,
        pad_last_batch=True,
        weighting="hat",
    )

    assert probabilities.shape == (len(text),)
    assert np.allclose(probabilities, 1 / (1 + np.exp(-1)))


def test_legacy_batched_extractor_routes_character_head(monkeypatch):
    model = ConstantCharacterHeadModel()
    tokenizer = CharacterHeadTokenizer()
    monkeypatch.setattr(
        "wtpsplit.extract_batched.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    texts = ["ab cd ef", "xy z"]

    logits, lengths, returned_tokenizer = extract_batched(
        texts,
        model,
        block_size=4,
        batch_size=2,
        pad_last_batch=True,
    )

    assert [array.shape for array in logits] == [(8, 1), (4, 1)]
    assert all(np.allclose(array, 1.0) for array in logits)
    assert lengths == [len(text) for text in texts]
    assert returned_tokenizer is tokenizer
