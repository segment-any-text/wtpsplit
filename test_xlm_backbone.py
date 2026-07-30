"""Hub-free behavioral tests for the production XLM-R SaT backbone."""

import torch

from wtpsplit._training_loss import masked_binary_token_loss
from wtpsplit.configs import SubwordXLMConfig
from wtpsplit.models import (
    SubwordXLMForTokenClassification,
    get_extended_attention_mask,
)

SEQ_LEN = 24


def test_balanced_binary_loss_uses_the_observed_class_ratio():
    logits = torch.tensor([[[0.2], [-0.3], [0.7], [0.1], [-0.5]]])
    labels = torch.tensor([[1, 0, 0, 0, -100]])

    actual = masked_binary_token_loss(logits, labels, balance_classes=True)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(-1)[0, :4],
        labels[0, :4].float(),
        pos_weight=torch.tensor(3.0),
    )

    assert torch.allclose(actual, expected)


def make_config(
    lookahead=None,
    num_layers=2,
    lookahead_split_layers=None,
    num_labels=2,
    use_character_head=False,
    character_head_init="identity",
):
    return SubwordXLMConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        max_position_embeddings=SEQ_LEN * 2,
        num_labels=num_labels,
        use_character_head=use_character_head,
        character_head_init=character_head_init,
        lookahead=lookahead,
        lookahead_split_layers=lookahead_split_layers,
        pad_token_id=1,
    )


def build_model(**kwargs):
    torch.manual_seed(0)
    model = SubwordXLMForTokenClassification(make_config(**kwargs))
    model.eval()
    return model


def _influence_radius(model, probe_position=4, tolerance=1e-5):
    torch.manual_seed(1)
    input_ids = torch.randint(2, 128, (1, SEQ_LEN))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        baseline = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, probe_position]

    radius = -1
    for offset in range(1, SEQ_LEN - probe_position):
        perturbed = input_ids.clone()
        position = probe_position + offset
        perturbed[0, position] = (perturbed[0, position] + 61) % 126 + 2
        with torch.no_grad():
            changed = model(input_ids=perturbed, attention_mask=attention_mask).logits[0, probe_position]
        if (changed - baseline).abs().max() > tolerance:
            radius = offset
    return radius


def test_limited_lookahead_bounds_future_influence():
    model = build_model(lookahead=6, num_layers=2)
    radius = _influence_radius(model)

    assert radius > 0
    assert radius <= 6


def test_without_lookahead_is_bidirectional():
    radius = _influence_radius(build_model(lookahead=None, num_layers=2))
    assert radius > 6


def test_padding_is_respected_with_lookahead():
    model = build_model(lookahead=6, num_layers=2)
    input_ids = torch.randint(2, 128, (1, SEQ_LEN))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, SEQ_LEN // 2 :] = 0

    with torch.no_grad():
        baseline = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, : SEQ_LEN // 2]
    noisy = input_ids.clone()
    noisy[:, SEQ_LEN // 2 :] = torch.randint(2, 128, (1, SEQ_LEN // 2))
    with torch.no_grad():
        changed = model(input_ids=noisy, attention_mask=attention_mask).logits[:, : SEQ_LEN // 2]

    assert torch.allclose(baseline, changed, atol=1e-5)


def test_compatibility_kwargs_are_filtered_without_changing_logits():
    model = build_model(lookahead=6, num_layers=2)
    input_ids = torch.randint(2, 128, (1, SEQ_LEN))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        expected = model(input_ids=input_ids, attention_mask=attention_mask).logits
        actual = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hashed_ids=torch.zeros(1, SEQ_LEN, 2, dtype=torch.long),
            language_ids=torch.zeros(1, dtype=torch.long),
        ).logits

    assert torch.equal(expected, actual)


def test_single_logit_classifier_uses_masked_binary_loss():
    model = SubwordXLMForTokenClassification(make_config(num_labels=1))
    input_ids = torch.randint(2, 128, (2, SEQ_LEN))
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
    model = SubwordXLMForTokenClassification(make_config(num_labels=1, use_character_head=True))
    input_ids = torch.randint(2, 128, (2, SEQ_LEN))
    attention_mask = torch.ones_like(input_ids)
    width = 37
    char_to_token = torch.randint(1, SEQ_LEN - 1, (2, width))
    char_is_token_final = torch.randint(0, 2, (2, width))
    char_position_in_token = torch.randint(0, 16, (2, width))
    char_hashes = torch.randint(0, 8192, (2, width, 8))
    char_mask = torch.ones(2, width, dtype=torch.long)
    char_mask[0, -5:] = 0
    labels = torch.randint(0, 2, (2, width))
    labels[0, -5:] = -100

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
    assert model.roberta.embeddings.word_embeddings.weight.grad is not None


def test_character_head_configuration_round_trips(tmp_path):
    model = SubwordXLMForTokenClassification(make_config(num_labels=1, use_character_head=True))
    model.save_pretrained(tmp_path)

    reloaded = SubwordXLMForTokenClassification.from_pretrained(tmp_path)

    assert reloaded.config.use_character_head is True
    assert reloaded.character_head is not None


def test_loading_character_head_onto_token_checkpoint_preserves_identity_init(tmp_path):
    model = SubwordXLMForTokenClassification(make_config(num_labels=1))
    model.save_pretrained(tmp_path)

    reloaded = SubwordXLMForTokenClassification.from_pretrained(
        tmp_path,
        use_character_head=True,
    )
    head = reloaded.character_head

    assert head is not None
    assert torch.equal(head.suppression.weight[:, 0], torch.tensor([-12.0, 0.0]))
    assert torch.count_nonzero(head.delta.weight) == 0
    assert torch.count_nonzero(head.delta.bias) == 0


def test_raw_character_head_can_request_random_init(tmp_path):
    model = SubwordXLMForTokenClassification(make_config(num_labels=1))
    model.save_pretrained(tmp_path)

    reloaded = SubwordXLMForTokenClassification.from_pretrained(
        tmp_path,
        use_character_head=True,
        character_head_init="random",
    )
    head = reloaded.character_head

    assert head is not None
    assert not torch.equal(head.suppression.weight[:, 0], torch.tensor([-12.0, 0.0]))
    assert torch.count_nonzero(head.delta.weight) > 0


def test_split_layers_switch_preserves_total_budget():
    model = build_model(lookahead=6, num_layers=2, lookahead_split_layers=1)
    assert model.roberta.effective_lookahead == 6
    assert _influence_radius(model) <= 6


def test_dense_mask_matches_tril_semantics():
    config = make_config(lookahead=3, num_layers=1)
    attention_mask = torch.ones(1, 8)
    mask = get_extended_attention_mask(
        config,
        attention_mask,
        input_shape=attention_mask.shape,
        lookahead=3,
        dtype=torch.float32,
    )
    visible = mask[0, 0] == 0
    expected = torch.tril(torch.ones(8, 8), diagonal=3).bool()
    assert torch.equal(visible, expected)
