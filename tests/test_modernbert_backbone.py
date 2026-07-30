"""Tests for the ModernBERT-backed SaT 2 backbone.

The important property is behavioural rather than structural: with limited lookahead,
a token must not be able to influence the prediction at an earlier position beyond the
configured budget. Inspecting the mask tensor would only prove the mask was built; these
tests prove it is actually honoured end to end.
"""

from pathlib import Path

import pytest
import torch

from wtpsplit.configs import SubwordModernBertConfig
from wtpsplit.models_modernbert import (
    SubwordModernBertForTokenClassification,
    lookahead_mask_function,
    resolve_effective_lookahead,
)

SEQ_LEN = 64


def make_config(
    lookahead=None,
    num_layers=2,
    attn_implementation="sdpa",
    num_labels=2,
    use_character_head=False,
    character_head_init="identity",
):
    return SubwordModernBertConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        max_position_embeddings=SEQ_LEN * 2,
        # Wide enough that the sliding window is never the binding constraint, so the
        # influence-radius tests measure lookahead rather than ModernBERT's local window.
        # NB: `global_attn_every_n_layers=1` trips a rope_parameters validation bug in
        # transformers 5.14, so it is left at the default here.
        local_attention=SEQ_LEN * 2,
        num_labels=num_labels,
        use_character_head=use_character_head,
        character_head_init=character_head_init,
        lookahead=lookahead,
        pad_token_id=0,
        attn_implementation=attn_implementation,
    )


def build_model(**kwargs):
    torch.manual_seed(0)
    model = SubwordModernBertForTokenClassification(make_config(**kwargs))
    model.eval()
    return model


def test_effective_lookahead_divides_across_layers():
    """Matches SubwordXLMRobertaModel: `lookahead` is a total budget, split per layer."""
    assert resolve_effective_lookahead(make_config(lookahead=48, num_layers=12)) == 4
    assert resolve_effective_lookahead(make_config(lookahead=None)) is None


def test_effective_lookahead_rejects_indivisible_budget():
    with pytest.raises(ValueError, match="divisible"):
        resolve_effective_lookahead(make_config(lookahead=7, num_layers=2))


def test_lookahead_split_layers_is_rejected_not_ignored():
    config = make_config(lookahead=4, num_layers=2)
    config.lookahead_split_layers = 1
    with pytest.raises(ValueError, match="lookahead_split_layers"):
        resolve_effective_lookahead(config)


def test_mask_function_matches_tril_semantics():
    """The predicate must reproduce `torch.tril(ones, diagonal=lookahead)` exactly."""
    lookahead = 3
    length = 12
    fn = lookahead_mask_function(lookahead)
    q = torch.arange(length).unsqueeze(1).expand(length, length)
    kv = torch.arange(length).unsqueeze(0).expand(length, length)

    predicate = fn(0, 0, q, kv)
    reference = torch.tril(torch.ones(length, length), diagonal=lookahead).bool()

    assert torch.equal(predicate, reference)


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
def test_forward_runs_and_shapes_are_right(attn_implementation):
    model = build_model(lookahead=4, num_layers=2, attn_implementation=attn_implementation)
    input_ids = torch.randint(1, 256, (2, SEQ_LEN))
    attention_mask = torch.ones_like(input_ids)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    assert logits.shape == (2, SEQ_LEN, 2)
    assert torch.isfinite(logits).all()


def test_single_logit_classifier_uses_masked_binary_loss():
    model = SubwordModernBertForTokenClassification(make_config(num_labels=1))
    input_ids = torch.randint(1, 256, (2, SEQ_LEN))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 2, input_ids.shape)
    labels[0, -3:] = -100

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    keep = labels != -100
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs.logits.squeeze(-1)[keep], labels[keep].float()
    )

    assert torch.allclose(outputs.loss, expected)
    outputs.loss.backward()
    assert model.classifier.weight.grad is not None


def test_character_head_forward_uses_character_labels_and_backpropagates():
    model = SubwordModernBertForTokenClassification(
        make_config(num_labels=1, use_character_head=True)
    )
    input_ids = torch.randint(1, 256, (2, SEQ_LEN))
    attention_mask = torch.ones_like(input_ids)
    width = 83
    char_to_token = torch.randint(1, SEQ_LEN - 1, (2, width))
    char_is_token_final = torch.randint(0, 2, (2, width))
    char_position_in_token = torch.randint(0, 16, (2, width))
    char_hashes = torch.randint(0, 8192, (2, width, 8))
    char_mask = torch.ones(2, width, dtype=torch.long)
    char_mask[0, -7:] = 0
    labels = torch.randint(0, 2, (2, width))
    labels[0, -7:] = -100

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        char_to_token=char_to_token,
        char_is_token_final=char_is_token_final,
        char_position_in_token=char_position_in_token,
        char_hashes=char_hashes,
        char_mask=char_mask,
    )

    assert outputs.logits.shape == (2, width, 1)
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    assert model.character_head.delta.weight.grad is not None
    assert model.model.embeddings.tok_embeddings.weight.grad is not None


def test_character_head_configuration_round_trips(tmp_path):
    model = SubwordModernBertForTokenClassification(
        make_config(num_labels=1, use_character_head=True)
    )
    model.save_pretrained(tmp_path)

    reloaded = SubwordModernBertForTokenClassification.from_pretrained(tmp_path)

    assert reloaded.config.use_character_head is True
    assert reloaded.character_head is not None


def test_loading_character_head_onto_token_checkpoint_preserves_identity_init(tmp_path):
    model = SubwordModernBertForTokenClassification(make_config(num_labels=1))
    model.save_pretrained(tmp_path)

    reloaded = SubwordModernBertForTokenClassification.from_pretrained(
        tmp_path,
        use_character_head=True,
    )
    head = reloaded.character_head

    assert head is not None
    assert torch.equal(head.suppression.weight[:, 0], torch.tensor([-12.0, 0.0]))
    assert torch.count_nonzero(head.delta.weight) == 0
    assert torch.count_nonzero(head.delta.bias) == 0


def test_raw_character_head_can_request_random_init(tmp_path):
    model = SubwordModernBertForTokenClassification(make_config(num_labels=1))
    model.save_pretrained(tmp_path)

    reloaded = SubwordModernBertForTokenClassification.from_pretrained(
        tmp_path,
        use_character_head=True,
        character_head_init="random",
    )
    head = reloaded.character_head

    assert head is not None
    assert not torch.equal(head.suppression.weight[:, 0], torch.tensor([-12.0, 0.0]))
    assert torch.count_nonzero(head.delta.weight) > 0


def _influence_radius(model, seq_len=SEQ_LEN, probe_position=8, tol=1e-5):
    """Largest offset d such that perturbing token `probe_position + d` changes output there."""
    torch.manual_seed(1)
    input_ids = torch.randint(1, 256, (1, seq_len))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        baseline = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, probe_position]

    radius = -1
    for offset in range(1, seq_len - probe_position):
        perturbed = input_ids.clone()
        position = probe_position + offset
        perturbed[0, position] = (perturbed[0, position] + 137) % 256
        with torch.no_grad():
            changed = model(input_ids=perturbed, attention_mask=attention_mask).logits[0, probe_position]
        if (changed - baseline).abs().max() > tol:
            radius = offset
    return radius


def test_limited_lookahead_bounds_future_influence():
    """A token beyond the total budget must not reach back to an earlier position.

    Each layer sees `lookahead // num_layers` tokens ahead, so stacking `num_layers`
    layers gives a receptive field of at most `lookahead` positions forward.
    """
    total_lookahead = 8
    num_layers = 2
    model = build_model(lookahead=total_lookahead, num_layers=num_layers)

    radius = _influence_radius(model)

    assert radius > 0, "expected some forward influence within the budget"
    assert radius <= total_lookahead, f"influence reached {radius} tokens ahead, budget was {total_lookahead}"


def test_without_lookahead_attention_is_fully_bidirectional():
    """Control: the same probe must see far-future tokens when lookahead is disabled."""
    model = build_model(lookahead=None, num_layers=2)

    radius = _influence_radius(model)

    assert radius > 8, f"expected unrestricted attention to reach far ahead, got {radius}"


def test_lookahead_masks_are_built_only_when_configured():
    """With lookahead off, the mask path stays exactly the stock ModernBERT one."""
    model = build_model(lookahead=None, num_layers=2)
    assert model.effective_lookahead is None

    captured = {}
    original_forward = model.model.forward

    def spy(*args, **kwargs):
        captured["attention_mask"] = kwargs.get("attention_mask")
        return original_forward(*args, **kwargs)

    model.model.forward = spy
    input_ids = torch.randint(1, 256, (1, SEQ_LEN))
    model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    assert not isinstance(captured["attention_mask"], dict)


def test_padding_is_still_respected_alongside_lookahead():
    """Padded positions must not influence real ones once the lookahead mask composes in."""
    model = build_model(lookahead=8, num_layers=2)
    input_ids = torch.randint(1, 256, (1, SEQ_LEN))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, SEQ_LEN // 2 :] = 0
    input_ids[0, SEQ_LEN // 2 :] = 0

    with torch.no_grad():
        baseline = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, : SEQ_LEN // 2]

    noisy = input_ids.clone()
    noisy[0, SEQ_LEN // 2 :] = torch.randint(1, 256, (SEQ_LEN // 2,))
    with torch.no_grad():
        changed = model(input_ids=noisy, attention_mask=attention_mask).logits[0, : SEQ_LEN // 2]

    assert torch.allclose(baseline, changed, atol=1e-5)


def _flex_attention_runnable():
    """Probe the real kernel.

    Compiling a trivial function is not a sufficient check: dynamo handles that without
    Inductor codegen, whereas FlexAttention needs a lowered kernel, which has no CPU
    backend on macOS.
    """
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        shape = (1, 1, 128, 16)
        query = key = value = torch.zeros(shape)
        block_mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: kv_idx <= q_idx, B=None, H=None, Q_LEN=128, KV_LEN=128
        )
        flex_attention(query, key, value, block_mask=block_mask)
        return True
    except Exception:  # noqa: BLE001 - any failure here means the kernel is unavailable
        return False


def test_flex_attention_produces_a_block_mask():
    """The whole point of 1.1: lookahead must survive as a BlockMask, not a dense tensor.

    A dense `[batch, 1, seq, seq]` mask is what the XLM-R backbone builds and what
    FlashAttention cannot consume. Under `flex_attention` the same predicate has to come
    back as a `BlockMask` so it can be fused.
    """
    from torch.nn.attention.flex_attention import BlockMask

    from wtpsplit.models_modernbert import build_lookahead_attention_masks

    config = make_config(lookahead=8, num_layers=2, attn_implementation="flex_attention")
    masks = build_lookahead_attention_masks(
        config=config,
        attention_mask=torch.ones(1, SEQ_LEN, dtype=torch.long),
        batch_size=1,
        seq_len=SEQ_LEN,
        dtype=torch.float32,
        device=torch.device("cpu"),
        lookahead=4,
    )

    assert isinstance(masks["full_attention"], BlockMask)
    assert isinstance(masks["sliding_attention"], BlockMask)


@pytest.mark.skipif(
    not _flex_attention_runnable(),
    reason="FlexAttention requires a compiled kernel; TorchInductor has no CPU backend on macOS",
)
def test_flex_attention_matches_sdpa():
    """Same predicate, different kernel: outputs must agree."""
    input_ids = torch.randint(1, 256, (1, 128))
    attention_mask = torch.ones_like(input_ids)

    outputs = {}
    for implementation in ("sdpa", "flex_attention"):
        torch.manual_seed(0)
        model = SubwordModernBertForTokenClassification(
            make_config(lookahead=8, num_layers=2, attn_implementation=implementation)
        )
        model.eval()
        with torch.no_grad():
            outputs[implementation] = model(input_ids=input_ids, attention_mask=attention_mask).logits

    assert torch.allclose(outputs["sdpa"], outputs["flex_attention"], atol=1e-4)


def test_config_registers_with_auto_classes():
    from transformers import AutoConfig, AutoModelForTokenClassification

    import wtpsplit.models  # noqa: F401  (populates the registries)

    config = make_config(lookahead=4)
    assert AutoConfig.for_model("modernbert-token", vocab_size=256) is not None
    model = AutoModelForTokenClassification.from_config(config)
    assert isinstance(model, SubwordModernBertForTokenClassification)


def test_layer_types_resync_on_trim():
    """SaT ships a ladder of trimmed models, so overriding the layer count must work.

    ModernBertConfig derives `layer_types` only when unset, so a checkpoint trimmed via
    `from_pretrained(..., num_hidden_layers=n)` would otherwise keep the original list
    and fail validation on save.
    """
    config = make_config(num_layers=22)
    assert len(config.layer_types) == 22

    config.num_hidden_layers = 3
    config.validate()

    assert len(config.layer_types) == 3
    assert config.layer_types[0] == "full_attention"


def test_backbone_resolver_picks_the_lookahead_aware_wrapper():
    """`AutoModelForTokenClassification` would hand back the stock class and drop lookahead."""
    from wtpsplit.configs import SubwordXLMConfig
    from wtpsplit.train.backbones import BACKBONE_BY_MODEL_TYPE

    assert BACKBONE_BY_MODEL_TYPE["modernbert"] == (
        SubwordModernBertConfig,
        SubwordModernBertForTokenClassification,
    )
    assert BACKBONE_BY_MODEL_TYPE["xlm-roberta"][0] is SubwordXLMConfig


def test_extract_treats_modernbert_as_a_subword_model():
    """The old `"xlm" in model_type` test would have routed this down the char path."""
    from wtpsplit.extract import CHAR_MODEL_TYPES, max_content_block_size

    assert "modernbert-token" not in CHAR_MODEL_TYPES
    assert "bert-char" in CHAR_MODEL_TYPES

    # XLM-R keeps its historical 510 cap; ModernBERT unlocks its full context window.
    assert max_content_block_size(SubwordXLMConfigStub()) == 510
    assert max_content_block_size(make_config()) == make_config().max_position_embeddings - 2


class SubwordXLMConfigStub:
    model_type = "xlm-token"
    max_position_embeddings = 514


# --------------------------------------------------------------------------------------
# Equivalence guards. Generalising the backbone touched shared code paths, so these pin
# the pre-existing XLM-R and char-model behaviour against the heuristics they replaced.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type",
    ["xlm-token", "xlm-roberta", "bert-char", "la-canine"],
)
def test_subword_routing_unchanged_for_existing_backbones(model_type):
    """Replaced `"xlm" in model_type`; must agree on every backbone that already existed."""
    from wtpsplit.extract import CHAR_MODEL_TYPES

    assert (model_type not in CHAR_MODEL_TYPES) == ("xlm" in model_type)


@pytest.mark.parametrize("block_size", [64, 256, 510, 512, 1024])
def test_block_size_clamp_unchanged_for_xlm(block_size):
    """Replaced an open-coded subtraction; must equal it for XLM-R at every size."""
    from wtpsplit.configs import SubwordXLMConfig
    from wtpsplit.extract import max_content_block_size

    previous = block_size - (block_size - 510) if block_size > 510 else block_size
    current = min(block_size, max_content_block_size(SubwordXLMConfig()))

    assert current == previous


def test_pad_token_lookup_unchanged_for_existing_backbones():
    """Replaced `1 if "xlm" in model_type else 0` with a config lookup."""
    from wtpsplit.configs import BertCharConfig, LACanineConfig, SubwordXLMConfig

    for config in (SubwordXLMConfig(), BertCharConfig(), LACanineConfig()):
        previous = 1 if "xlm" in config.model_type else 0
        assert config.pad_token_id == previous, config.model_type


def test_sat_can_load_and_run_a_modernbert_checkpoint(tmp_path):
    """Full inference path: save a tiny checkpoint, load it through SaT, segment text."""
    from transformers import AutoTokenizer

    from wtpsplit import SaT

    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    config = SubwordModernBertConfig.from_pretrained(
        "jhu-clsp/mmBERT-base", num_hidden_layers=2, num_labels=1, lookahead=8
    )
    torch.manual_seed(0)
    model = SubwordModernBertForTokenClassification(config)
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)

    sat = SaT(str(tmp_path))
    assert isinstance(sat.model.model, SubwordModernBertForTokenClassification)
    assert sat.model.model.effective_lookahead == 4
    # Tokenizer must come from the checkpoint dir, not the XLM-R default.
    assert Path(sat.tokenizer.name_or_path).resolve() == tmp_path.resolve()

    text = "This is a test This is another test."
    segments = sat.split(text)
    assert "".join(segments) == text, "segmentation must be lossless"
    assert sat.predict_proba(text).shape[0] == len(text), "probabilities are per character"

    batched = [list(x) for x in sat.split([text, "A b. C d."])]
    assert len(batched) == 2
