"""Tests for the character-resolution boundary head.

Two properties carry the design:

1. **Every character is reachable**, so the tokenization ceiling disappears. This is what
   caps Tibetan at 0.237 recall today, and what makes mmBERT cost Mandarin and Japanese
   ~10 points of achievable recall.
2. **Initialisation preserves current behaviour**, so an existing SaT checkpoint can be
   warm-started instead of retrained, and a fine-tune that learns nothing is no worse than
   the model it started from.
"""

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from wtpsplit.char_head import (
    SUPPRESSION_BIAS,
    CharacterDataCollator,
    CharacterResolutionHead,
    build_char_inputs,
    char_boundary_label_batch,
    char_boundary_labels,
)
from wtpsplit.evaluation.diagnostics.boundary_ceiling import boundary_candidates
from wtpsplit.utils import token_to_char_probs

HIDDEN = 32


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("facebookAI/xlm-roberta-base")


@pytest.fixture(scope="module", params=["facebookAI/xlm-roberta-base", "jhu-clsp/mmBERT-base"])
def backbone_tokenizer(request):
    return AutoTokenizer.from_pretrained(request.param)


def encode(tokenizer, texts):
    encoding = tokenizer(texts, return_offsets_mapping=True, add_special_tokens=False)
    return encoding["input_ids"], encoding["offset_mapping"]


def make_head(num_labels=1):
    torch.manual_seed(0)
    return CharacterResolutionHead(HIDDEN, num_labels=num_labels)


# --------------------------------------------------------------------------------------
# Character indexing
# --------------------------------------------------------------------------------------


def test_every_character_maps_to_a_token(tokenizer):
    texts = ["This is a test. And another.", "Kurz."]
    _, offsets = encode(tokenizer, texts)
    inputs = build_char_inputs(texts, offsets)

    for i, text in enumerate(texts):
        assert inputs.mask[i, : len(text)].all()
        assert not inputs.mask[i, len(text) :].any()
        # every token contributes exactly one final character
        assert inputs.is_token_final[i].sum() <= len(offsets[i])


def test_token_final_flags_match_the_offsets(tokenizer):
    text = "This is a test. And another."
    _, offsets = encode(tokenizer, [text])
    inputs = build_char_inputs([text], offsets)

    expected = {end - 1 for start, end in offsets[0] if end > start}
    actual = set(torch.where(inputs.is_token_final[0] == 1)[0].tolist())

    assert actual == expected


@pytest.mark.parametrize(
    "text",
    [
        "This is a test. And another.",
        "中文句子。另一个句子。",
        "བོད་སྐད་ཀྱི་ཡི་གེ།",  # Tibetan: XLM-R emits multi-character <unk> spans here
        "วันนี้อากาศดี เราไปสวนกัน",
    ],
)
def test_all_characters_are_reachable(tokenizer, text):
    """The point of the head: the reachable set is the whole string, not token ends."""
    _, offsets = encode(tokenizer, [text])
    inputs = build_char_inputs([text], offsets)
    head = make_head()

    token_hidden = torch.randn(1, max(len(offsets[0]), 1), HIDDEN)
    token_logits = torch.randn(1, max(len(offsets[0]), 1), 1)
    head.reset_to_identity()
    # Relax the suppression so non-final characters are no longer held down; this is what
    # training is free to learn.
    with torch.no_grad():
        head.suppression.weight.zero_()

    logits = head(token_hidden, token_logits, inputs)[0, : len(text), 0]

    assert torch.isfinite(logits).all(), "every character must carry a usable logit"
    # Contrast with the current mapping, where only token-final positions are finite.
    tokens = tokenizer.convert_ids_to_tokens(encode(tokenizer, [text])[0][0])
    current = token_to_char_probs(text, tokens, np.random.randn(len(tokens), 1), set(), offsets[0])
    assert np.isneginf(current[:, 0]).any(), "sanity: the current mapping does leave gaps"
    assert not np.isneginf(current[:, 0]).all()


def test_ceiling_becomes_one_where_it_currently_binds(tokenizer):
    """Tibetan is the extreme case: measured ceiling 0.237 under XLM-R."""
    sentences = ["བོད་སྐད་ཀྱི་ཡི་གེ།", "འདི་གཉིས་པ་ཡིན།"]
    separator = " "
    text = separator.join(sentences)
    _, offsets = encode(tokenizer, [text])

    token_ends = {end - 1 for start, end in offsets[0] if end > start}
    candidates = boundary_candidates(sentences, separator)
    reachable_now = sum(1 for c in candidates if c & token_ends)

    # With the head, the reachable set is every position the mask covers.
    inputs = build_char_inputs([text], offsets)
    reachable_positions = set(torch.where(inputs.mask[0] == 1)[0].tolist())
    reachable_with_head = sum(1 for c in candidates if c & reachable_positions)

    assert reachable_with_head == len(candidates), "the head must make every boundary reachable"
    assert reachable_with_head >= reachable_now


# --------------------------------------------------------------------------------------
# Behaviour-preserving initialisation
# --------------------------------------------------------------------------------------


def test_initialisation_reproduces_the_last_character_mapping(tokenizer):
    text = "This is a test. And another."
    token_ids, offsets = encode(tokenizer, [text])
    inputs = build_char_inputs([text], offsets)

    head = make_head()
    token_hidden = torch.randn(1, len(token_ids[0]), HIDDEN)
    token_logits = torch.randn(1, len(token_ids[0]), 1)

    logits = head(token_hidden, token_logits, inputs)[0, : len(text), 0]

    final_positions = torch.where(inputs.is_token_final[0, : len(text)] == 1)[0]
    other_positions = torch.where(inputs.is_token_final[0, : len(text)] == 0)[0]

    # Token-final characters carry exactly the token's own logit.
    expected = token_logits[0, inputs.char_to_token[0, final_positions], 0]
    assert torch.allclose(logits[final_positions], expected, atol=1e-5)

    # Everything else is suppressed by the full bias.
    suppressed = logits[other_positions] - token_logits[0, inputs.char_to_token[0, other_positions], 0]
    assert torch.allclose(suppressed, torch.full_like(suppressed, -SUPPRESSION_BIAS), atol=1e-5)


def test_suppression_survives_a_sigmoid(tokenizer):
    """Suppressed characters must be effectively zero, but finite for fp16 training."""
    text = "This is a test."
    token_ids, offsets = encode(tokenizer, [text])
    inputs = build_char_inputs([text], offsets)
    head = make_head()

    token_logits = torch.zeros(1, len(token_ids[0]), 1)
    logits = head(torch.zeros(1, len(token_ids[0]), HIDDEN), token_logits, inputs)

    assert torch.isfinite(logits).all(), "no -inf: it would produce NaNs in mixed precision"
    suppressed = torch.sigmoid(logits[0, inputs.is_token_final[0] == 0, 0])
    assert (suppressed < 1e-4).all()


def test_head_is_trainable_away_from_identity(tokenizer):
    """A gradient step must be able to move probability onto a non-final character."""
    text = "This is a test."
    token_ids, offsets = encode(tokenizer, [text])
    inputs = build_char_inputs([text], offsets)
    head = make_head()

    token_hidden = torch.randn(1, len(token_ids[0]), HIDDEN)
    token_logits = torch.zeros(1, len(token_ids[0]), 1)
    target_position = int(torch.where(inputs.is_token_final[0] == 0)[0][0])

    optimiser = torch.optim.Adam(head.parameters(), lr=0.1)
    before = head(token_hidden, token_logits, inputs)[0, target_position, 0].item()
    for _ in range(50):
        optimiser.zero_grad()
        logits = head(token_hidden, token_logits, inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[0, target_position, 0], torch.tensor(1.0)
        )
        loss.backward()
        optimiser.step()
    after = head(token_hidden, token_logits, inputs)[0, target_position, 0].item()

    assert after > before + 1.0, f"expected the head to learn; moved {before:.3f} -> {after:.3f}"


# --------------------------------------------------------------------------------------
# Shape and cost
# --------------------------------------------------------------------------------------


def test_batched_shapes_and_padding(tokenizer):
    texts = ["A short one.", "A considerably longer sentence, with more characters in it."]
    token_ids, offsets = encode(tokenizer, texts)
    inputs = build_char_inputs(texts, offsets)
    width = max(len(t) for t in texts)
    num_tokens = max(len(t) for t in token_ids)

    head = make_head(num_labels=2)
    logits = head(torch.randn(2, num_tokens, HIDDEN), torch.randn(2, num_tokens, 2), inputs)

    assert logits.shape == (2, width, 2)
    # Padded positions are suppressed rather than left as arbitrary values.
    assert (logits[0, len(texts[0]) :] == -SUPPRESSION_BIAS).all()


# --------------------------------------------------------------------------------------
# Training targets
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentences,separator",
    [
        (["This is a test.", "This is another test.", "And a third."], " "),
        (["中文句子。", "另一个句子。"], ""),
        (["Kurz.", "Ok."], " "),
        (["วันนี้อากาศดี", "เราไปสวนกัน"], " "),
    ],
)
def test_char_labels_reconstruct_the_gold_split(sentences, separator):
    """The strongest available check: decoding at the labels must return the gold sentences.

    If this holds, the targets are correct by construction rather than by inspection.
    """
    from wtpsplit.utils import indices_to_sentences

    text = separator.join(sentences)
    labels = char_boundary_labels(sentences, separator)

    assert len(labels) == len(text)
    reconstructed = indices_to_sentences(text, np.where(labels)[0])

    assert "".join(reconstructed) == text
    assert [s.strip() for s in reconstructed] == [s.strip() for s in sentences]


def test_char_labels_leave_the_final_boundary_unlabelled():
    """`indices_to_sentences` always appends the tail, so the last boundary is free."""
    sentences = ["One.", "Two.", "Three."]
    labels = char_boundary_labels(sentences, " ")

    assert labels.sum() == len(sentences) - 1
    assert labels[-1] == 0


def test_char_label_batch_pads_with_ignore_index():
    documents = [["A short one.", "Then more."], ["Tiny.", "Bit."]]
    batch = char_boundary_label_batch(documents, [" ", " "], ignore_index=-100)

    widths = [len(" ".join(d)) for d in documents]
    assert batch.shape == (2, max(widths))
    for i, width in enumerate(widths):
        assert (batch[i, width:] == -100).all()
        assert (batch[i, :width] != -100).all()


def test_training_collator_aligns_both_tokenizers(backbone_tokenizer):
    texts = ["Short.", "A longer sentence. Then another."]
    encoded = backbone_tokenizer(texts, return_offsets_mapping=True, add_special_tokens=False)
    block_size = max(len(ids) for ids in encoded["input_ids"]) + 2
    features = []
    for text, token_ids, offsets in zip(texts, encoded["input_ids"], encoded["offset_mapping"]):
        input_ids = [
            backbone_tokenizer.cls_token_id,
            *token_ids,
            backbone_tokenizer.sep_token_id,
        ]
        padding = block_size - len(input_ids)
        features.append(
            {
                "input_ids": input_ids + [backbone_tokenizer.pad_token_id] * padding,
                "attention_mask": [1] * len(input_ids) + [0] * padding,
                "labels": char_boundary_labels([text], "").tolist(),
                "text": text,
                "offset_mapping": offsets,
            }
        )

    batch = CharacterDataCollator()(features)

    assert batch["labels"].shape == batch["char_mask"].shape
    assert batch["char_hashes"].shape[:2] == batch["labels"].shape
    assert (batch["labels"][0, len(texts[0]) :] == -100).all()
    assert batch["char_to_token"].min() >= 1
    for index, token_ids in enumerate(encoded["input_ids"]):
        assert batch["char_to_token"][index, : len(texts[index])].max() <= len(token_ids)


def test_labels_align_with_head_output_shape(tokenizer):
    """Targets and logits must line up position-for-position, or training silently misaligns."""
    sentences = ["This is a test.", "This is another test."]
    separator = " "
    text = separator.join(sentences)
    token_ids, offsets = encode(tokenizer, [text])

    inputs = build_char_inputs([text], offsets)
    head = make_head()
    logits = head(torch.randn(1, len(token_ids[0]), HIDDEN), torch.randn(1, len(token_ids[0]), 1), inputs)
    labels = char_boundary_label_batch([sentences], [separator], width=logits.shape[1])

    assert logits.shape[:2] == labels.shape
    # And the loss is computable over the unpadded region.
    keep = labels[0] != -100
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[0, keep, 0], labels[0, keep].float()
    )
    assert torch.isfinite(loss)


def test_manual_depthwise_matches_nn_conv1d():
    """The hand-rolled depthwise path must be numerically identical to `nn.Conv1d`.

    It exists purely for speed: PyTorch's depthwise kernel measured 1277 ms of a 1298 ms
    head on CPU, against 21 ms for every other component. If this ever diverges, the
    speedup has silently changed the model.
    """
    torch.manual_seed(0)
    head = CharacterResolutionHead(768)
    hidden = torch.randn(4, 300, head.context.in_channels)

    with torch.no_grad():
        reference = head.context(hidden.transpose(1, 2)).transpose(1, 2)
        actual = head._depthwise(hidden)

    assert torch.allclose(reference, actual, atol=1e-5)


def test_parameter_count_is_small_relative_to_a_backbone():
    """The head must not meaningfully change model size."""
    head = CharacterResolutionHead(768, num_labels=1)
    params = sum(p.numel() for p in head.parameters())

    # One XLM-R encoder layer is roughly 7M parameters; the head should be well under it.
    assert params < 2_000_000, f"head has {params:,} parameters"
