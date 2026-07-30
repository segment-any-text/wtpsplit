"""Public inference defaults."""

import json
from importlib.resources import files
import re

DEFAULT_STRIDE = 64

# Operating points for the released checkpoint families. These are the values SaT has
# used since 2.0
DEFAULT_SENTENCE_THRESHOLD = 0.025
SM_SENTENCE_THRESHOLD = 0.25
NO_LOOKAHEAD_SENTENCE_THRESHOLD = 0.01

# Released names look like `sat-3l`, `sat-12l-sm`, `sat-9l-no-limited-lookahead`.
_CHECKPOINT_NAME = re.compile(r"^sat-\d+l(?:-no-limited-lookahead)?(?:-sm)?$")


def default_threshold_for_checkpoint(model_name_or_path: str) -> float | None:
    """Shipped operating point for a released checkpoint, or ``None`` if unrecognised.

    Matching is anchored on the final path component. The previous implementation tested
    ``"sm" in model_str`` against the whole string, so any local directory that happened to
    contain those two letters -- ``/home/smith/model``, ``./transformers-cache/...`` --
    silently selected the ``-sm`` operating point, which is ten times the base default.
    """
    name = model_name_or_path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if not _CHECKPOINT_NAME.match(name):
        return None
    # `-sm` takes precedence when a checkpoint carries both suffixes, matching the
    # precedence of the original substring checks.
    if name.endswith("-sm"):
        return SM_SENTENCE_THRESHOLD
    if name.endswith("-no-limited-lookahead"):
        return NO_LOOKAHEAD_SENTENCE_THRESHOLD
    return DEFAULT_SENTENCE_THRESHOLD


def calibrated_threshold_for_checkpoint(
    model_name_or_path: str,
    language: str | None = None,
    domain: str | None = None,
) -> float | None:
    """Return an experimental development-fitted threshold when available.

    Version 1 contains BOUQuET-development calibration and currently has no
    domain-specific cells. Unknown languages use the checkpoint's fitted global
    operating point. Unknown checkpoints return ``None`` rather than silently
    inheriting another model's calibration.
    """
    del domain  # Reserved for exact-gold domain calibration in the same schema.
    name = str(model_name_or_path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    try:
        payload = json.loads(files("wtpsplit.data").joinpath("calibrated_thresholds.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ModuleNotFoundError, json.JSONDecodeError):
        return None
    checkpoint = payload.get("checkpoints", {}).get(name)
    if checkpoint is None:
        return None
    if language is not None and language in checkpoint["languages"]:
        return float(checkpoint["languages"][language])
    return float(checkpoint["global"])
