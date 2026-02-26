import json
import warnings
from pathlib import Path

import numpy as np

# Default for unknown languages
DEFAULT_SENTENCE_STATS = {"target_length": 70, "spread": 25}

# Sentence statistics loaded from JSON (computed from Universal Dependencies)
# See: scripts/compute_sentence_stats.py
#
# Note: Some wtpsplit-supported languages are not in UD and will use defaults:
#   bn, eo, fy, ig, km, kn, ku, mg, mn, ms, my, ne, pa, ps, sq, tg, uz, xh, yi, zu
_stats_path = Path(__file__).parent.parent / "data" / "sentence_stats.json"
if _stats_path.exists():
    with open(_stats_path, "r") as _f:
        LANG_SENTENCE_STATS = json.load(_f).get("stats", {})
else:
    LANG_SENTENCE_STATS = {}


def get_language_defaults(lang_code=None):
    """Get recommended target_length and spread for a given language."""
    if lang_code is None:
        return DEFAULT_SENTENCE_STATS.copy()
    if lang_code not in LANG_SENTENCE_STATS:
        warnings.warn(
            f"No sentence statistics for '{lang_code}', using defaults "
            f"(target_length={DEFAULT_SENTENCE_STATS['target_length']}, "
            f"spread={DEFAULT_SENTENCE_STATS['spread']}). "
            f"You can override with explicit prior_kwargs.",
            stacklevel=3,
        )
        return DEFAULT_SENTENCE_STATS.copy()
    return LANG_SENTENCE_STATS.get(lang_code, DEFAULT_SENTENCE_STATS).copy()


def create_prior_function(name, kwargs):
    if name == "uniform":
        max_length = kwargs.get("max_length")

        def prior(length):
            if max_length is not None and length > max_length:
                return 0.0
            return 1.0

        return prior

    elif name == "clipped_polynomial":
        # Quadratic falloff from target_length, clips to zero far from peak
        # Use language-aware defaults if lang_code provided and target_length not specified
        lang_defaults = get_language_defaults(kwargs.get("lang_code"))
        target_length = kwargs.get("target_length", lang_defaults["target_length"])
        # Convert spread (tolerance in chars) to falloff coefficient
        # Clips to zero at |length - target| = spread
        spread = kwargs.get("spread", lang_defaults["spread"])
        falloff = 1.0 / (spread**2)
        max_length = kwargs.get("max_length")

        def prior(length):
            if max_length is not None and length > max_length:
                return 0.0
            val = 1.0 - falloff * ((length - target_length) ** 2)
            return max(val, 0.0)

        return prior

    elif name == "gaussian":
        # Gaussian prior centered at target_length
        # Use language-aware defaults if lang_code provided and target_length not specified
        lang_defaults = get_language_defaults(kwargs.get("lang_code"))
        target_length = kwargs.get("target_length", lang_defaults["target_length"])
        spread = kwargs.get("spread", lang_defaults["spread"])
        max_length = kwargs.get("max_length")

        def prior(length):
            if max_length is not None and length > max_length:
                return 0.0
            return np.exp(-0.5 * ((length - target_length) / spread) ** 2)

        return prior

    elif name == "lognormal":
        # Log-normal prior - right-skewed distribution (more tolerant of longer segments)
        # Use language-aware defaults if lang_code provided
        lang_defaults = get_language_defaults(kwargs.get("lang_code"))
        target_length = kwargs.get("target_length", lang_defaults["target_length"])
        # spread is in characters (like gaussian/clipped_polynomial) for consistency
        spread = kwargs.get("spread", lang_defaults["spread"])
        max_length = kwargs.get("max_length")

        # Convert character-based spread to lognormal sigma
        # sigma ≈ spread / target_length gives values in sensible 0.3-0.5 range
        sigma = spread / target_length
        mu = np.log(target_length) + sigma**2

        def prior(length):
            if length <= 0:
                return 0.0
            if max_length is not None and length > max_length:
                return 0.0
            log_len = np.log(length)
            return np.exp(-0.5 * ((log_len - mu) / sigma) ** 2) / length

        return prior

    else:
        raise ValueError(f"Unknown prior: {name}")
