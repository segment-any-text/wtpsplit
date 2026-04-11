"""Stable imports for custom ``Trainer`` subclasses on **transformers 4.x and 5.x**.

In v5, many symbols are no longer re-exported from ``transformers.trainer``; importing from the
same public submodules used internally works on both versions (``integrations.deepspeed``,
``pytorch_utils``, ``trainer_pt_utils``, etc.).
"""

from transformers.integrations.deepspeed import deepspeed_init
from transformers.modeling_utils import unwrap_model
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (
    DataLoader,
    EvalLoopOutput,
    IterableDatasetShard,
    TRAINING_ARGS_NAME,
    WEIGHTS_NAME,
    logger,
)
from transformers.trainer_pt_utils import (
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_numpify,
    nested_truncate,
)
from transformers.trainer_utils import denumpify_detensorize, has_length
from transformers.utils import is_sagemaker_mp_enabled

__all__ = [
    "ALL_LAYERNORM_LAYERS",
    "DataLoader",
    "EvalLoopOutput",
    "IterableDatasetShard",
    "TRAINING_ARGS_NAME",
    "WEIGHTS_NAME",
    "deepspeed_init",
    "denumpify_detensorize",
    "find_batch_size",
    "get_parameter_names",
    "has_length",
    "is_sagemaker_mp_enabled",
    "logger",
    "nested_concat",
    "nested_numpify",
    "nested_truncate",
    "unwrap_model",
]
