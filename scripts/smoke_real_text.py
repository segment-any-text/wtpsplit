"""
Smoke test: run one larger multilingual/real-world text through each model (SaT, SaT-SM, WtP).
Checks that splits are non-empty and text is preserved (join(splits) == input modulo whitespace).
Usage: python scripts/smoke_real_text.py
Run from repo root.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Multilingual / real-world style text
SMOKE_TEXT = """
Text segmentation is the process of dividing written text into meaningful units, such as words, sentences, or topics. The term applies both to mental processes used by humans when reading text, and to artificial processes implemented in computers. Le problème est non trivial : certaines langues ont des délimiteurs explicites, d'autres non. 日本語では句読点の使い方が異なります。 Daniel Craig (born 2 March 1968) is an English actor who played James Bond from Casino Royale (2006) up to No Time to Die (2021).
""".strip()


def _norm(s):
    return " ".join(s.split())


def main():
    from wtpsplit import SaT, WtP

    tf_version = "unknown"
    try:
        import transformers

        tf_version = transformers.__version__
    except Exception:
        pass
    print(f"transformers={tf_version}  smoke text length={len(SMOKE_TEXT)} chars")
    errors = []

    # SaT
    try:
        sat = SaT("segment-any-text/sat-3l", hub_prefix=None)
        splits = sat.split(SMOKE_TEXT, threshold=0.025)
        assert splits, "SaT returned no splits"
        assert _norm("".join(splits)) == _norm(SMOKE_TEXT), "SaT text mismatch"
        print("  SaT: OK")
    except Exception as e:
        errors.append(f"SaT: {e}")
        print("  SaT: FAIL", e)

    # SaT-SM
    try:
        sat_sm = SaT("segment-any-text/sat-12l-sm", hub_prefix=None)
        splits = sat_sm.split(SMOKE_TEXT, threshold=0.25)
        assert splits, "SaT-SM returned no splits"
        assert _norm("".join(splits)) == _norm(SMOKE_TEXT), "SaT-SM text mismatch"
        print("  SaT-SM: OK")
    except Exception as e:
        errors.append(f"SaT-SM: {e}")
        print("  SaT-SM: FAIL", e)

    # WtP Bert
    try:
        wtp = WtP("benjamin/wtp-bert-mini", hub_prefix=None, ignore_legacy_warning=True)
        splits = wtp.split(SMOKE_TEXT, lang_code="en", threshold=0.005)
        assert splits, "WtP Bert returned no splits"
        assert _norm("".join(splits)) == _norm(SMOKE_TEXT), "WtP Bert text mismatch"
        print("  WtP Bert: OK")
    except Exception as e:
        errors.append(f"WtP Bert: {e}")
        print("  WtP Bert: FAIL", e)

    # WtP Canine
    try:
        wtp_c = WtP("benjamin/wtp-canine-s-3l", hub_prefix=None, ignore_legacy_warning=True)
        splits = wtp_c.split(SMOKE_TEXT, lang_code="en")
        assert splits, "WtP Canine returned no splits"
        assert _norm("".join(splits)) == _norm(SMOKE_TEXT), "WtP Canine text mismatch"
        print("  WtP Canine: OK")
    except Exception as e:
        errors.append(f"WtP Canine: {e}")
        print("  WtP Canine: FAIL", e)

    if errors:
        print("Smoke test FAILED:", errors)
        return 1
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
