"""
Capture split and logit-level outputs for parity testing across branches/envs.
Includes edge cases from the test suite (empty strings, batched, strip_whitespace,
newlines, paragraphs, LoRA, threshold, long input).
Usage: python scripts/capture_outputs.py [--out FILE]
Writes JSON to FILE. Run from repo root.
"""

import json
import os
import sys

# Repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _tolist(a):
    if isinstance(a, np.ndarray):
        return a.tolist()
    if isinstance(a, (list, tuple)):
        return [_tolist(x) for x in a]
    return a


def main():
    import torch

    torch.manual_seed(42)
    np.random.seed(42)

    out_path = "outputs_capture.json"
    if "--out" in sys.argv:
        i = sys.argv.index("--out")
        out_path = sys.argv[i + 1]

    from wtpsplit import SaT, WtP

    results = {
        "transformers": None,
        "sat": {"model": "segment-any-text/sat-3l", "cases": {}},
        "sat_sm": {"model": "segment-any-text/sat-12l-sm", "cases": {}},
        "wtp": {"model": "benjamin/wtp-bert-mini", "cases": {}},
        "wtp_canine": {"model": "benjamin/wtp-canine-s-3l", "cases": {}},
    }

    try:
        import transformers

        results["transformers"] = transformers.__version__
    except Exception:
        pass

    # Long two-paragraph text (used in paragraph tests)
    LONG_PARAGRAPH_TEXT = " ".join(
        """
Text segmentation is the process of dividing written text into meaningful units, such as words, sentences, or topics. The term applies both to mental processes used by humans when reading text, and to artificial processes implemented in computers, which are the subject of natural language processing. The problem is non-trivial, because while some written languages have explicit word boundary markers, such as the word spaces of written English and the distinctive initial, medial and final letter shapes of Arabic, such signals are sometimes ambiguous and not present in all written languages.
Daniel Wroughton Craig CMG (born 2 March 1968) is an English actor who gained international fame by playing the fictional secret agent James Bond for five installments in the film series, from Casino Royale (2006) up to No Time to Die (2021).
""".strip().split()
    )

    # --- SaT ---
    try:
        sat = SaT("segment-any-text/sat-3l", hub_prefix=None)

        # basic
        t = "This is a test sentence. This is another test sentence."
        results["sat"]["cases"]["basic"] = {
            "splits": sat.split(t, threshold=0.025),
            "proba": _tolist(sat.predict_proba(t, stride=64, block_size=512)),
        }

        # strip_whitespace
        t = "This is a test sentence This is another test sentence.   "
        results["sat"]["cases"]["strip_whitespace"] = {
            "splits": sat.split(t, strip_whitespace=True, threshold=0.025),
        }

        # newline (split on newline)
        t = "Yes\nthis is a test sentence. This is another test sentence."
        results["sat"]["cases"]["newline"] = {"splits": sat.split(t)}

        # newline as space
        results["sat"]["cases"]["newline_as_space"] = {
            "splits": sat.split(t, treat_newline_as_space=True),
        }

        # noisy
        t = "this is a sentence :) this is another sentence lol"
        results["sat"]["cases"]["noisy"] = {"splits": sat.split(t)}

        # batched
        texts = ["Paragraph-A Paragraph-B", "Paragraph-C100 Paragraph-D"]
        results["sat"]["cases"]["batched"] = {
            "splits": list(sat.split(texts, threshold=0.025)),
        }

        # empty strings
        results["sat"]["cases"]["empty_strings"] = {
            "splits_list": [
                sat.split(""),
                sat.split("   "),
                sat.split("   \n"),
            ],
        }

        # paragraphs
        results["sat"]["cases"]["paragraphs"] = {
            "splits": sat.split(LONG_PARAGRAPH_TEXT, do_paragraph_segmentation=True),
        }

        # LoRA: same text, different adapter builds (ud, opus100, ersatz)
        LORA_TEXT = "'I couldn't help it,' said Five, in a sulky tone; 'Seven jogged my elbow.' | On which Seven looked up and said, 'That's right, Five! Always lay the blame (...)!'"
        for style in ("ud", "opus100", "ersatz"):
            sat_lora = SaT("segment-any-text/sat-3l", hub_prefix=None, style_or_domain=style, language="en")
            results["sat"]["cases"][f"lora_{style}"] = {
                "splits": sat_lora.split(LORA_TEXT),
                "proba": _tolist(sat_lora.predict_proba(LORA_TEXT, stride=64, block_size=512)),
            }

        # LoRA with merge_lora=True explicitly (default; may use manual merge on tf5)
        sat_ud_merge = SaT(
            "segment-any-text/sat-3l", hub_prefix=None, style_or_domain="ud", language="en", merge_lora=True
        )
        results["sat"]["cases"]["lora_ud_merge_true"] = {
            "splits": sat_ud_merge.split(LORA_TEXT),
        }

    except Exception as e:
        results["sat"]["error"] = str(e)

    # --- SaT-SM (small model): without and with adapter ---
    try:
        sat_sm = SaT("segment-any-text/sat-12l-sm", hub_prefix=None)
        t = "This is a test sentence. This is another test sentence."
        results["sat_sm"]["cases"]["basic_no_adapter"] = {
            "splits": sat_sm.split(t, threshold=0.25),
            "proba": _tolist(sat_sm.predict_proba(t, stride=64, block_size=512)),
        }
        t = "this is a sentence :) this is another sentence lol"
        results["sat_sm"]["cases"]["noisy_no_adapter"] = {"splits": sat_sm.split(t)}
        # SaT-SM with LoRA/adapter (same style API as full SaT if supported)
        try:
            sat_sm_ud = SaT("segment-any-text/sat-12l-sm", hub_prefix=None, style_or_domain="ud", language="en")
            t = "This is a test sentence. This is another test sentence."
            results["sat_sm"]["cases"]["basic_with_adapter_ud"] = {
                "splits": sat_sm_ud.split(t),
                "proba": _tolist(sat_sm_ud.predict_proba(t, stride=64, block_size=512)),
            }
        except Exception as e_sm_lora:
            results["sat_sm"]["cases"]["basic_with_adapter_ud"] = {"error": str(e_sm_lora)}
    except Exception as e:
        results["sat_sm"]["error"] = str(e)

    # --- WtP (bert-mini): without and with style adapter ---
    try:
        wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None, ignore_legacy_warning=True)

        t = "This is a test sentence This is another test sentence."
        results["wtp"]["cases"]["basic"] = {
            "splits": wtp.split(t, threshold=0.005),
            "proba": _tolist(
                wtp.predict_proba(t, lang_code="en", style=None, stride=64, block_size=512, weighting="uniform")
            ),
        }

        t = "This is a test sentence This is another test sentence.   "
        results["wtp"]["cases"]["strip_whitespace"] = {
            "splits": wtp.split(t, strip_whitespace=True, threshold=0.005),
        }

        prefix = "x" * 2000
        t = prefix + " This is a test sentence. This is another test sentence."
        results["wtp"]["cases"]["long"] = {"splits": wtp.split(t)}

        texts = ["Paragraph-A Paragraph-B", "Paragraph-C100 Paragraph-D"]
        results["wtp"]["cases"]["batched"] = {
            "splits": list(wtp.split(texts, threshold=0.005)),
        }

        results["wtp"]["cases"]["paragraphs"] = {
            "splits": wtp.split(LONG_PARAGRAPH_TEXT, do_paragraph_segmentation=True),
        }

        t = "This is a test sentence. This is another test sentence."
        results["wtp"]["cases"]["threshold_high"] = {
            "splits": wtp.split(t, threshold=1.0),
        }

        results["wtp"]["cases"]["threshold_low"] = {
            "splits": wtp.split(t, threshold=-1e-3),
            "splits_first3": wtp.split(t, threshold=-1e-3)[:3],
        }

        # WtP with style adapter (ud, opus100, ersatz)
        WTP_STYLE_TEXT = "'I couldn't help it,' said Five, in a sulky tone; 'Seven jogged my elbow.' | On which Seven looked up and said, 'That's right, Five! Always lay the blame (...)!'"
        for style in ("ud", "opus100", "ersatz"):
            results["wtp"]["cases"][f"style_{style}"] = {
                "splits": wtp.split(WTP_STYLE_TEXT, lang_code="en", style=style),
            }

    except Exception as e:
        results["wtp"]["error"] = str(e)

    # --- WtP Canine: without and with lang adapter ---
    try:
        wtp_c = WtP("benjamin/wtp-canine-s-3l", hub_prefix=None, ignore_legacy_warning=True)
        t = "This is a test sentence. This is another test sentence."
        results["wtp_canine"]["cases"]["basic_no_adapter"] = {
            "splits": wtp_c.split(t, lang_code="en"),
        }
        results["wtp_canine"]["cases"]["paragraphs_with_lang_en"] = {
            "splits": wtp_c.split(LONG_PARAGRAPH_TEXT, do_paragraph_segmentation=True, lang_code="en"),
        }
    except Exception as e:
        results["wtp_canine"]["error"] = str(e)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {out_path} (transformers={results['transformers']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
