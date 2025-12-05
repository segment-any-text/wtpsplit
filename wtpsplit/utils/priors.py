import numpy as np

# Empirical sentence length statistics per language (approximate, based on typical corpora)
# Used as defaults when lang_code is provided but target_length is not specified
#
# TODO: Derive these values from actual corpus statistics (mC4, OPUS100, UD training data)
#       Current values are educated guesses based on linguistic similarity.
#       Languages needing verification (insufficient data for confident estimates):
#       - African: am, ha, ig, mg, xh, yo, zu
#       - Others: ceb, eo, eu, jv
#
LANG_SENTENCE_STATS = {
    # === East Asian (no spaces, shorter in chars) ===
    "zh": {"target_length": 45, "spread": 15},  # Chinese
    "ja": {"target_length": 50, "spread": 18},  # Japanese
    "ko": {"target_length": 55, "spread": 18},  # Korean
    # === Southeast Asian ===
    "th": {"target_length": 60, "spread": 20},  # Thai (no spaces)
    "vi": {"target_length": 65, "spread": 22},  # Vietnamese
    "id": {"target_length": 70, "spread": 25},  # Indonesian
    "ms": {"target_length": 70, "spread": 25},  # Malay (similar to Indonesian)
    "jv": {"target_length": 68, "spread": 24},  # Javanese
    "km": {"target_length": 55, "spread": 18},  # Khmer (no spaces)
    "my": {"target_length": 55, "spread": 18},  # Myanmar/Burmese (no spaces)
    "ceb": {"target_length": 68, "spread": 25},  # Cebuano (Austronesian)
    # === South Asian / Indic ===
    "hi": {"target_length": 70, "spread": 25},  # Hindi
    "bn": {"target_length": 68, "spread": 25},  # Bengali
    "ta": {"target_length": 70, "spread": 26},  # Tamil
    "te": {"target_length": 68, "spread": 25},  # Telugu
    "ur": {"target_length": 70, "spread": 25},  # Urdu (similar to Hindi)
    "mr": {"target_length": 70, "spread": 25},  # Marathi
    "gu": {"target_length": 68, "spread": 25},  # Gujarati
    "pa": {"target_length": 70, "spread": 25},  # Punjabi
    "kn": {"target_length": 70, "spread": 26},  # Kannada
    "ml": {"target_length": 72, "spread": 26},  # Malayalam
    "ne": {"target_length": 68, "spread": 25},  # Nepali
    "si": {"target_length": 68, "spread": 25},  # Sinhala
    # === Germanic ===
    "de": {"target_length": 90, "spread": 35},  # German (long-ish compounds)
    "nl": {"target_length": 80, "spread": 30},  # Dutch
    "en": {"target_length": 75, "spread": 28},  # English
    "sv": {"target_length": 80, "spread": 30},  # Swedish
    "da": {"target_length": 78, "spread": 28},  # Danish
    "no": {"target_length": 78, "spread": 28},  # Norwegian
    "af": {"target_length": 75, "spread": 28},  # Afrikaans (Dutch-derived, simpler)
    "fy": {"target_length": 78, "spread": 28},  # Frisian
    "is": {"target_length": 80, "spread": 30},  # Icelandic
    "yi": {"target_length": 78, "spread": 28},  # Yiddish (Germanic base)
    # === Romance ===
    "fr": {"target_length": 85, "spread": 32},  # French
    "es": {"target_length": 80, "spread": 30},  # Spanish
    "it": {"target_length": 82, "spread": 30},  # Italian
    "pt": {"target_length": 80, "spread": 30},  # Portuguese
    "ro": {"target_length": 78, "spread": 28},  # Romanian
    "ca": {"target_length": 78, "spread": 28},  # Catalan
    "gl": {"target_length": 80, "spread": 30},  # Galician (similar to Portuguese)
    # === Slavic ===
    "ru": {"target_length": 85, "spread": 32},  # Russian
    "pl": {"target_length": 82, "spread": 30},  # Polish
    "cs": {"target_length": 80, "spread": 30},  # Czech
    "uk": {"target_length": 82, "spread": 30},  # Ukrainian
    "be": {"target_length": 82, "spread": 30},  # Belarusian (similar to Russian/Ukrainian)
    "bg": {"target_length": 78, "spread": 28},  # Bulgarian
    "mk": {"target_length": 78, "spread": 28},  # Macedonian (similar to Bulgarian)
    "sr": {"target_length": 80, "spread": 30},  # Serbian
    "sk": {"target_length": 80, "spread": 30},  # Slovak (similar to Czech)
    "sl": {"target_length": 78, "spread": 28},  # Slovenian
    # === Baltic ===
    "lt": {"target_length": 80, "spread": 30},  # Lithuanian
    "lv": {"target_length": 78, "spread": 28},  # Latvian
    # === Celtic ===
    "cy": {"target_length": 75, "spread": 28},  # Welsh
    "ga": {"target_length": 75, "spread": 28},  # Irish
    "gd": {"target_length": 75, "spread": 28},  # Scottish Gaelic
    # === Other European ===
    "el": {"target_length": 80, "spread": 30},  # Greek
    "fi": {"target_length": 75, "spread": 28},  # Finnish
    "hu": {"target_length": 78, "spread": 28},  # Hungarian
    "et": {"target_length": 75, "spread": 28},  # Estonian (Finnic)
    "sq": {"target_length": 78, "spread": 28},  # Albanian
    "mt": {"target_length": 75, "spread": 28},  # Maltese
    "eu": {"target_length": 78, "spread": 28},  # Basque (isolate)
    "la": {"target_length": 90, "spread": 35},  # Latin (classical, long sentences)
    "eo": {"target_length": 75, "spread": 28},  # Esperanto (European-style)
    # === Turkic ===
    "tr": {"target_length": 72, "spread": 26},  # Turkish
    "az": {"target_length": 72, "spread": 26},  # Azerbaijani
    "kk": {"target_length": 70, "spread": 26},  # Kazakh
    "ky": {"target_length": 70, "spread": 26},  # Kyrgyz
    "uz": {"target_length": 72, "spread": 26},  # Uzbek
    # === Iranian/Persian ===
    "fa": {"target_length": 80, "spread": 30},  # Persian/Farsi
    "ku": {"target_length": 75, "spread": 28},  # Kurdish
    "ps": {"target_length": 75, "spread": 28},  # Pashto
    "tg": {"target_length": 80, "spread": 30},  # Tajik (Persian variant)
    # === Caucasian ===
    "hy": {"target_length": 78, "spread": 28},  # Armenian
    "ka": {"target_length": 75, "spread": 28},  # Georgian
    # === Semitic ===
    "ar": {"target_length": 85, "spread": 32},  # Arabic
    "he": {"target_length": 75, "spread": 28},  # Hebrew
    "am": {"target_length": 75, "spread": 28},  # Amharic (Ethiopian Semitic)
    # === Mongolian ===
    # TODO ?
    "mn": {"target_length": 70, "spread": 26},  # Mongolian
    # === African (estimates - need verification) ===
    # TODO
    "ha": {"target_length": 70, "spread": 25},  # Hausa
    "ig": {"target_length": 70, "spread": 25},  # Igbo
    "yo": {"target_length": 70, "spread": 25},  # Yoruba
    "xh": {"target_length": 70, "spread": 25},  # Xhosa (Bantu)
    "zu": {"target_length": 70, "spread": 25},  # Zulu (Bantu)
    "mg": {"target_length": 70, "spread": 25},  # Malagasy
}

# Default for unknown languages
DEFAULT_SENTENCE_STATS = {"target_length": 70, "spread": 25}


def get_language_defaults(lang_code=None):
    """Get recommended target_length and spread for a given language."""
    if lang_code is None:
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
        # Log-normal prior - better models natural sentence length distribution
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
