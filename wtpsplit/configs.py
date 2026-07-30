from transformers import ModernBertConfig, XLMRobertaConfig


class SubwordXLMConfig(XLMRobertaConfig):
    """Config for XLM-R. Used for token-level training, i.e., SaT models.

    Args:
        XLMRobertaConfig: Base class.
    """

    model_type = "xlm-token"
    mixture_name = "xlm-token"

    def __init__(
        self,
        lookahead=None,
        lookahead_split_layers=None,
        use_character_head=False,
        balance_character_loss=False,
        character_head_init="identity",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if character_head_init not in {"identity", "random"}:
            raise ValueError("character_head_init must be 'identity' or 'random'")
        self.mixture_name = "xlm-token"
        self.lookahead = lookahead
        self.lookahead_split_layers = lookahead_split_layers
        self.use_character_head = use_character_head
        self.balance_character_loss = balance_character_loss
        self.character_head_init = character_head_init


class SubwordModernBertConfig(ModernBertConfig):
    """Config for ModernBERT-family backbones such as mmBERT. Used for SaT 2 models.

    `lookahead` keeps the same meaning as in `SubwordXLMConfig`: a total budget of
    forward-visible tokens, divided across layers.

    Args:
        ModernBertConfig: Base class.
    """

    model_type = "modernbert-token"
    mixture_name = "modernbert-token"

    def __init__(
        self,
        lookahead=None,
        lookahead_split_layers=None,
        use_character_head=False,
        balance_character_loss=False,
        character_head_init="identity",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if character_head_init not in {"identity", "random"}:
            raise ValueError("character_head_init must be 'identity' or 'random'")
        self.mixture_name = "modernbert-token"
        self.lookahead = lookahead
        self.lookahead_split_layers = lookahead_split_layers
        self.use_character_head = use_character_head
        self.balance_character_loss = balance_character_loss
        self.character_head_init = character_head_init
        self._resync_layer_types()

    def validate(self):
        # `from_pretrained` applies keyword overrides *after* `__init__`, so a trim
        # requested there is invisible until validation time. Resync here as well.
        self._resync_layer_types()
        return super().validate()

    def _resync_layer_types(self):
        """Recompute `layer_types` when the layer count is overridden.

        ModernBertConfig only derives `layer_types` when it is unset, so trimming a
        pretrained checkpoint (`num_hidden_layers=3` against a 22-layer mmBERT) leaves a
        stale list behind and fails validation. SaT ships a ladder of trimmed models, so
        this has to work.
        """
        layer_types = getattr(self, "layer_types", None)
        if layer_types is not None and len(layer_types) == self.num_hidden_layers:
            return

        every_n = getattr(self, "global_attn_every_n_layers", None)
        if every_n is None and layer_types:
            # Infer the stride from the existing pattern rather than assuming the default.
            full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
            every_n = full[1] - full[0] if len(full) > 1 else 3
        if not every_n:
            every_n = 3

        self.layer_types = [
            "sliding_attention" if bool(i % every_n) else "full_attention" for i in range(self.num_hidden_layers)
        ]


_LEGACY_CONFIGS = {"BertCharConfig", "LACanineConfig"}


def __getattr__(name: str):
    if name in _LEGACY_CONFIGS:
        from wtpsplit.legacy import configs as legacy_configs

        return getattr(legacy_configs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from wtpsplit.model_registry import register_sat_configs  # noqa: E402

register_sat_configs()
