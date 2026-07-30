"""Model-free tests for the narrow Transformers registration boundary."""

from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification

from wtpsplit.model_registry import (
    register_legacy_models,
    register_sat_configs,
    register_sat_models,
)


def test_sat_registration_is_explicit_and_idempotent():
    register_sat_configs()
    register_sat_models()
    register_sat_models()

    from wtpsplit.configs import SubwordModernBertConfig, SubwordXLMConfig
    from wtpsplit.models import SubwordXLMForTokenClassification, SubwordXLMRobertaModel
    from wtpsplit.models_modernbert import SubwordModernBertForTokenClassification

    assert AutoConfig.for_model("xlm-token").__class__ is SubwordXLMConfig
    assert AutoConfig.for_model("modernbert-token").__class__ is SubwordModernBertConfig
    xlm_config = SubwordXLMConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=32,
    )
    modern_config = SubwordModernBertConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
        local_attention=128,
        attn_implementation="sdpa",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    assert isinstance(AutoModel.from_config(xlm_config), SubwordXLMRobertaModel)
    assert isinstance(
        AutoModelForTokenClassification.from_config(xlm_config),
        SubwordXLMForTokenClassification,
    )
    assert isinstance(
        AutoModelForTokenClassification.from_config(modern_config),
        SubwordModernBertForTokenClassification,
    )


def test_legacy_registration_is_explicit_and_idempotent():
    register_legacy_models()
    register_legacy_models()

    from wtpsplit.legacy.configs import BertCharConfig, LACanineConfig
    from wtpsplit.legacy.models import (
        BertCharForTokenClassification,
        BertCharModel,
        LACanineForTokenClassification,
        LACanineModel,
    )

    assert AutoConfig.for_model("bert-char").__class__ is BertCharConfig
    assert AutoConfig.for_model("la-canine").__class__ is LACanineConfig
    bert_config = BertCharConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_hash_buckets=32,
        num_hash_functions=2,
    )
    canine_config = LACanineConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_hash_buckets=32,
        num_hash_functions=2,
        n_languages=2,
    )
    assert isinstance(AutoModel.from_config(bert_config), BertCharModel)
    assert isinstance(AutoModel.from_config(canine_config), LACanineModel)
    assert isinstance(
        AutoModelForTokenClassification.from_config(bert_config),
        BertCharForTokenClassification,
    )
    assert isinstance(
        AutoModelForTokenClassification.from_config(canine_config),
        LACanineForTokenClassification,
    )
