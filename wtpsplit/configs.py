from transformers import AutoConfig, BertConfig, CanineConfig, XLMRobertaConfig


class LACanineConfig(CanineConfig):
    model_type = "la-canine"

    def __init__(
        self,
        n_languages=None,
        ngram_order=1,
        bottleneck_factor=2,
        language_adapter="on",
        lookahead=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_languages = n_languages
        self.ngram_order = ngram_order
        self.language_adapter = language_adapter  # 'on', 'off', 'shared'
        self.bottleneck_factor = bottleneck_factor

        self.lookahead = lookahead
        self.lookahead_block_size = 1


class BertCharConfig(BertConfig):
    model_type = "bert-char"

    def __init__(
        self,
        num_hash_buckets=8192,
        num_hash_functions=8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_hash_buckets = num_hash_buckets
        self.num_hash_functions = num_hash_functions


class SubwordXLMConfig(XLMRobertaConfig):
    """Config for XLM-R and XLM-V models. Used for token-level training.

    Args:
        XLMRobertaConfig: Base class.
    """

    model_type = "xlm-token"
    mixture_name = "xlm-token"

    def __init__(
        self,
        lookahead=None,
        lookahead_split_layers=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mixture_name = "xlm-token"
        self.lookahead = lookahead
        self.lookahead_split_layers = lookahead_split_layers


AutoConfig.register("bert-char", BertCharConfig)
AutoConfig.register("la-canine", LACanineConfig)
AutoConfig.register("xlm-token", SubwordXLMConfig)
