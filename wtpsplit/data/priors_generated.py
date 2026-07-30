"""
Empirical sentence length statistics per language.

Computed from Universal Dependencies treebanks.
Generated: 2026-01-21
Source: Universal Dependencies v2.14

Usage:
    from wtpsplit.utils.priors import LANG_SENTENCE_STATS, get_language_defaults

    # Get defaults for a language
    defaults = get_language_defaults("de")  # {"target_length": ..., "spread": ...}

Methodology:
    - target_length: median sentence length in characters (robust to outliers)
    - spread: IQR / 1.35 (robust estimate of standard deviation)

To regenerate:
    python scripts/compute_sentence_stats.py --use_hf --format python
"""

import numpy as np

LANG_SENTENCE_STATS = {
    "ab": {"target_length": 48, "spread": 22},  # n=263, median=48.0, p25-p75=[36-66]
    "af": {"target_length": 132, "spread": 70},  # n=1934, median=132.0, p25-p75=[91-186]
    "ajp": {"target_length": 27, "spread": 14},  # n=100, median=27.5, p25-p75=[18-37]
    "akk": {"target_length": 51, "spread": 36},  # n=1975, median=51.0, p25-p75=[32-81]
    "aln": {"target_length": 67, "spread": 35},  # n=966, median=67.0, p25-p75=[46-94]
    "am": {"target_length": 18, "spread": 10},  # n=1074, median=18.0, p25-p75=[13-24]
    "apu": {"target_length": 30, "spread": 12},  # n=161, median=30.0, p25-p75=[23-40]
    "aqz": {"target_length": 16, "spread": 10},  # n=343, median=16.0, p25-p75=[12-20]
    "ar": {"target_length": 67, "spread": 55},  # n=28402, median=67.0, p25-p75=[39-114]
    "arr": {"target_length": 16, "spread": 10},  # n=674, median=16.0, p25-p75=[10-21]
    "az": {"target_length": 29, "spread": 14},  # n=108, median=29.0, p25-p75=[19-39]
    "azz": {"target_length": 30, "spread": 28},  # n=1260, median=30.0, p25-p75=[15-54]
    "bar": {"target_length": 51, "spread": 40},  # n=1070, median=51.0, p25-p75=[32-86]
    "be": {"target_length": 58, "spread": 48},  # n=25231, median=58.0, p25-p75=[32-97]
    "bej": {"target_length": 58, "spread": 46},  # n=380, median=58.0, p25-p75=[33-96]
    "bg": {"target_length": 66, "spread": 46},  # n=11138, median=66.0, p25-p75=[39-102]
    "bho": {"target_length": 72, "spread": 42},  # n=357, median=72.0, p25-p75=[47-104]
    "bm": {"target_length": 38, "spread": 22},  # n=1026, median=38.0, p25-p75=[25-56]
    "bor": {"target_length": 29, "spread": 22},  # n=881, median=29.0, p25-p75=[19-49]
    "br": {"target_length": 42, "spread": 33},  # n=884, median=42.5, p25-p75=[25-70]
    "bxr": {"target_length": 55, "spread": 37},  # n=927, median=55.0, p25-p75=[36-87]
    "ca": {"target_length": 151, "spread": 85},  # n=16678, median=151.0, p25-p75=[98-213]
    "ceb": {"target_length": 27, "spread": 13},  # n=188, median=27.0, p25-p75=[20-38]
    "ckt": {"target_length": 34, "spread": 19},  # n=1004, median=34.0, p25-p75=[23-49]
    "cop": {"target_length": 70, "spread": 42},  # n=2203, median=70.0, p25-p75=[44-101]
    "cpg": {"target_length": 40, "spread": 28},  # n=350, median=40.0, p25-p75=[25-64]
    "cs": {"target_length": 86, "spread": 62},  # n=127794, median=86.0, p25-p75=[48-133]
    "cu": {"target_length": 39, "spread": 29},  # n=22628, median=39.0, p25-p75=[25-65]
    "cy": {"target_length": 82, "spread": 62},  # n=2551, median=82.0, p25-p75=[46-131]
    "da": {"target_length": 82, "spread": 61},  # n=5512, median=82.0, p25-p75=[47-130]
    "de": {"target_length": 111, "spread": 61},  # n=208438, median=111.0, p25-p75=[72-155]
    "egy": {"target_length": 29, "spread": 16},  # n=721, median=29.0, p25-p75=[20-42]
    "el": {"target_length": 94, "spread": 74},  # n=4328, median=94.0, p25-p75=[55-155]
    "eme": {"target_length": 19, "spread": 11},  # n=913, median=19.0, p25-p75=[12-27]
    "en": {"target_length": 63, "spread": 53},  # n=47644, median=63.0, p25-p75=[33-105]
    "es": {"target_length": 135, "spread": 87},  # n=35214, median=135.0, p25-p75=[84-202]
    "ess": {"target_length": 29, "spread": 11},  # n=309, median=29.0, p25-p75=[22-38]
    "et": {"target_length": 74, "spread": 55},  # n=38147, median=74.0, p25-p75=[41-116]
    "eu": {"target_length": 79, "spread": 45},  # n=8993, median=79.0, p25-p75=[52-113]
    "fa": {"target_length": 72, "spread": 48},  # n=35104, median=72.0, p25-p75=[46-111]
    "fi": {"target_length": 63, "spread": 47},  # n=36981, median=63.0, p25-p75=[36-100]
    "fo": {"target_length": 63, "spread": 48},  # n=2829, median=63.0, p25-p75=[41-107]
    "fr": {"target_length": 90, "spread": 65},  # n=29735, median=90.0, p25-p75=[52-141]
    "frm": {"target_length": 87, "spread": 66},  # n=512, median=87.0, p25-p75=[54-143]
    "fro": {"target_length": 37, "spread": 25},  # n=19765, median=37.0, p25-p75=[27-61]
    "ga": {"target_length": 111, "spread": 54},  # n=7699, median=111.0, p25-p75=[69-142]
    "gd": {"target_length": 66, "spread": 72},  # n=4741, median=66.0, p25-p75=[32-130]
    "gl": {"target_length": 160, "spread": 57},  # n=5993, median=160.0, p25-p75=[124-201]
    "got": {"target_length": 49, "spread": 37},  # n=5401, median=49.0, p25-p75=[28-79]
    "grc": {"target_length": 66, "spread": 48},  # n=32577, median=66.0, p25-p75=[39-105]
    "gsw": {"target_length": 73, "spread": 25},  # n=100, median=73.0, p25-p75=[54-88]
    "gu": {"target_length": 36, "spread": 26},  # n=187, median=36.0, p25-p75=[24-60]
    "gub": {"target_length": 33, "spread": 16},  # n=1182, median=33.0, p25-p75=[24-46]
    "gun": {"target_length": 20, "spread": 11},  # n=1144, median=20.0, p25-p75=[15-30]
    "gv": {"target_length": 26, "spread": 10},  # n=2336, median=26.0, p25-p75=[20-34]
    "ha": {"target_length": 29, "spread": 24},  # n=2318, median=29.0, p25-p75=[16-49]
    "hbo": {"target_length": 118, "spread": 45},  # n=1579, median=118.0, p25-p75=[91-152]
    "he": {"target_length": 86, "spread": 48},  # n=11182, median=86.0, p25-p75=[57-123]
    "hi": {"target_length": 93, "spread": 45},  # n=17649, median=93.0, p25-p75=[66-128]
    "hit": {"target_length": 67, "spread": 37},  # n=136, median=67.0, p25-p75=[42-93]
    "hr": {"target_length": 118, "spread": 68},  # n=9010, median=118.0, p25-p75=[77-169]
    "hsb": {"target_length": 86, "spread": 48},  # n=646, median=86.5, p25-p75=[59-124]
    "ht": {"target_length": 89, "spread": 47},  # n=144, median=89.5, p25-p75=[66-131]
    "hu": {"target_length": 138, "spread": 79},  # n=1800, median=138.0, p25-p75=[90-197]
    "hy": {"target_length": 91, "spread": 77},  # n=4800, median=91.0, p25-p75=[49-154]
    "hyw": {"target_length": 88, "spread": 69},  # n=6656, median=88.0, p25-p75=[49-143]
    "id": {"target_length": 117, "spread": 64},  # n=7628, median=117.0, p25-p75=[80-167]
    "is": {"target_length": 87, "spread": 64},  # n=53564, median=87.0, p25-p75=[53-140]
    "it": {"target_length": 108, "spread": 62},  # n=40273, median=108.0, p25-p75=[65-149]
    "ja": {"target_length": 18, "spread": 15},  # n=132418, median=18.0, p25-p75=[9-30]
    "jv": {"target_length": 69, "spread": 38},  # n=1000, median=69.0, p25-p75=[46-98]
    "ka": {"target_length": 85, "spread": 60},  # n=151, median=85.0, p25-p75=[56-138]
    "kk": {"target_length": 52, "spread": 28},  # n=1078, median=52.0, p25-p75=[33-71]
    "kmr": {"target_length": 57, "spread": 21},  # n=754, median=57.0, p25-p75=[47-76]
    "ko": {"target_length": 44, "spread": 22},  # n=34702, median=44.0, p25-p75=[30-61]
    "koi": {"target_length": 34, "spread": 20},  # n=128, median=34.0, p25-p75=[22-49]
    "kpv": {"target_length": 51, "spread": 36},  # n=877, median=51.0, p25-p75=[33-82]
    "krl": {"target_length": 73, "spread": 40},  # n=228, median=73.5, p25-p75=[53-107]
    "ky": {"target_length": 58, "spread": 25},  # n=926, median=58.0, p25-p75=[39-73]
    "la": {"target_length": 73, "spread": 58},  # n=59946, median=73.0, p25-p75=[44-123]
    "lij": {"target_length": 55, "spread": 49},  # n=316, median=55.0, p25-p75=[35-102]
    "lt": {"target_length": 102, "spread": 74},  # n=3905, median=102.0, p25-p75=[59-159]
    "lv": {"target_length": 84, "spread": 62},  # n=18870, median=84.0, p25-p75=[47-132]
    "lzh": {"target_length": 5, "spread": 10},  # n=86339, median=5.0, p25-p75=[4-7]
    "mdf": {"target_length": 49, "spread": 22},  # n=474, median=49.0, p25-p75=[35-65]
    "mk": {"target_length": 37, "spread": 14},  # n=155, median=37.0, p25-p75=[28-48]
    "ml": {"target_length": 69, "spread": 47},  # n=218, median=69.5, p25-p75=[44-108]
    "mr": {"target_length": 32, "spread": 17},  # n=466, median=32.0, p25-p75=[21-44]
    "mt": {"target_length": 98, "spread": 75},  # n=2074, median=98.0, p25-p75=[53-155]
    "myu": {"target_length": 25, "spread": 14},  # n=158, median=25.0, p25-p75=[17-37]
    "myv": {"target_length": 46, "spread": 31},  # n=2138, median=46.0, p25-p75=[28-71]
    "nds": {"target_length": 80, "spread": 59},  # n=1000, median=80.5, p25-p75=[45-126]
    "nhi": {"target_length": 47, "spread": 31},  # n=909, median=47.0, p25-p75=[29-71]
    "nl": {"target_length": 79, "spread": 62},  # n=30723, median=79.0, p25-p75=[42-127]
    "no": {"target_length": 77, "spread": 54},  # n=37619, median=77.0, p25-p75=[44-117]
    "olo": {"target_length": 70, "spread": 41},  # n=125, median=70.0, p25-p75=[44-100]
    "orv": {"target_length": 43, "spread": 38},  # n=36661, median=43.0, p25-p75=[26-78]
    "ota": {"target_length": 82, "spread": 69},  # n=599, median=82.0, p25-p75=[48-142]
    "pad": {"target_length": 26, "spread": 13},  # n=101, median=26.0, p25-p75=[20-38]
    "pcm": {"target_length": 46, "spread": 37},  # n=9241, median=46.0, p25-p75=[27-77]
    "pl": {"target_length": 55, "spread": 42},  # n=40398, median=55.0, p25-p75=[33-90]
    "pt": {"target_length": 74, "spread": 57},  # n=78141, median=74.0, p25-p75=[44-121]
    "qaf": {"target_length": 68, "spread": 39},  # n=1287, median=68.0, p25-p75=[44-97]
    "qfn": {"target_length": 47, "spread": 18},  # n=400, median=47.0, p25-p75=[34-59]
    "qpm": {"target_length": 61, "spread": 40},  # n=2250, median=61.0, p25-p75=[37-91]
    "qtd": {"target_length": 79, "spread": 45},  # n=2184, median=79.0, p25-p75=[53-115]
    "quc": {"target_length": 26, "spread": 10},  # n=1435, median=26.0, p25-p75=[20-34]
    "ro": {"target_length": 102, "spread": 60},  # n=40690, median=102.0, p25-p75=[66-148]
    "ru": {"target_length": 79, "spread": 60},  # n=116324, median=79.0, p25-p75=[44-126]
    "sa": {"target_length": 37, "spread": 27},  # n=27412, median=37.0, p25-p75=[24-61]
    "sah": {"target_length": 25, "spread": 10},  # n=299, median=25.0, p25-p75=[21-32]
    "say": {"target_length": 32, "spread": 25},  # n=1864, median=32.0, p25-p75=[17-52]
    "si": {"target_length": 44, "spread": 10},  # n=100, median=44.0, p25-p75=[40-52]
    "sjo": {"target_length": 62, "spread": 59},  # n=810, median=62.0, p25-p75=[40-121]
    "sk": {"target_length": 44, "spread": 31},  # n=10604, median=44.0, p25-p75=[26-69]
    "sl": {"target_length": 79, "spread": 64},  # n=19539, median=79.0, p25-p75=[41-128]
    "sme": {"target_length": 41, "spread": 29},  # n=3122, median=41.0, p25-p75=[28-68]
    "sms": {"target_length": 51, "spread": 30},  # n=250, median=51.0, p25-p75=[34-75]
    "sr": {"target_length": 120, "spread": 62},  # n=4384, median=120.0, p25-p75=[80-165]
    "sv": {"target_length": 83, "spread": 53},  # n=12269, median=83.0, p25-p75=[51-123]
    "swl": {"target_length": 66, "spread": 46},  # n=203, median=66.0, p25-p75=[36-98]
    "ta": {"target_length": 52, "spread": 58},  # n=1134, median=52.0, p25-p75=[31-110]
    "te": {"target_length": 25, "spread": 10},  # n=1328, median=25.0, p25-p75=[19-32]
    "th": {"target_length": 95, "spread": 40},  # n=1000, median=95.0, p25-p75=[70-124]
    "tl": {"target_length": 34, "spread": 21},  # n=222, median=34.5, p25-p75=[22-51]
    "tpn": {"target_length": 38, "spread": 23},  # n=581, median=38.0, p25-p75=[25-57]
    "tr": {"target_length": 49, "spread": 37},  # n=82319, median=49.0, p25-p75=[26-77]
    "tt": {"target_length": 93, "spread": 44},  # n=148, median=93.5, p25-p75=[60-120]
    "ug": {"target_length": 63, "spread": 35},  # n=3456, median=63.0, p25-p75=[42-90]
    "uk": {"target_length": 75, "spread": 59},  # n=7092, median=75.0, p25-p75=[42-123]
    "ur": {"target_length": 108, "spread": 59},  # n=5130, median=108.0, p25-p75=[74-155]
    "vep": {"target_length": 69, "spread": 36},  # n=103, median=69.0, p25-p75=[44-94]
    "vi": {"target_length": 80, "spread": 40},  # n=3423, median=80.0, p25-p75=[53-108]
    "wo": {"target_length": 78, "spread": 48},  # n=2107, median=78.0, p25-p75=[50-115]
    "xav": {"target_length": 42, "spread": 25},  # n=171, median=42.0, p25-p75=[26-60]
    "xcl": {"target_length": 73, "spread": 42},  # n=4146, median=73.0, p25-p75=[48-105]
    "xnr": {"target_length": 35, "spread": 14},  # n=288, median=35.0, p25-p75=[27-47]
    "xum": {"target_length": 26, "spread": 26},  # n=133, median=26.0, p25-p75=[16-52]
    "yo": {"target_length": 96, "spread": 45},  # n=318, median=96.0, p25-p75=[69-131]
    "yrl": {"target_length": 38, "spread": 30},  # n=1470, median=38.0, p25-p75=[23-64]
    "yue": {"target_length": 13, "spread": 14},  # n=1004, median=13.0, p25-p75=[7-26]
    "zh": {"target_length": 29, "spread": 19},  # n=14944, median=29.0, p25-p75=[17-43]
}

# Default for unknown languages (approximate global average)
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
