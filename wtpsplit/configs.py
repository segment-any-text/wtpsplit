from transformers import XLMRobertaConfig

from wtpsplit._inference import validate_input_line_breaks


class SubwordXLMConfig(XLMRobertaConfig):
    """Config for XLM-R. Used for token-level training, i.e., SaT models.

    Args:
        XLMRobertaConfig: Base class.
        input_line_breaks: ``"preserve"`` (default; released checkpoints) or ``"space"``.
            With ``"space"``, ``SaT`` maps each input LF/CR to a space before encoding.
    """

    model_type = "xlm-token"
    mixture_name = "xlm-token"

    def __init__(
        self,
        lookahead=None,
        lookahead_split_layers=None,
        use_character_head=False,
        balance_character_loss=False,
        balance_boundary_loss=None,
        character_head_init="identity",
        character_head_project_tokens_first=False,
        input_line_breaks="preserve",
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
        self.balance_boundary_loss = balance_boundary_loss
        self.character_head_init = character_head_init
        self.character_head_project_tokens_first = character_head_project_tokens_first
        self.input_line_breaks = validate_input_line_breaks(input_line_breaks)


_LEGACY_CONFIGS = {"BertCharConfig", "LACanineConfig"}


def __getattr__(name: str):
    if name in _LEGACY_CONFIGS:
        from wtpsplit.legacy import configs as legacy_configs

        return getattr(legacy_configs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from wtpsplit.model_registry import register_sat_configs  # noqa: E402

register_sat_configs()
