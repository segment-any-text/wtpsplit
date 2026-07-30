"""Compatibility and dependency-boundary tests for the legacy WtP API."""

import subprocess
import sys

import pytest

from wtpsplit.utils import Constants


def test_language_info_preserves_dataframe_compatibility_surface():
    assert "en" in Constants.LANGINFO.index
    assert Constants.LANGINFO.loc["en", "no_whitespace"] is False
    assert Constants.LANGINFO.loc["zh", "no_whitespace"] is True
    assert dict(Constants.LANGINFO.iterrows())["en"]["ud"] == "UD_English-GUM"


def test_wtp_is_reexported_from_legacy_package():
    from wtpsplit import WtP

    assert WtP.__module__ == "wtpsplit.legacy.wtp"


def test_legacy_models_remain_import_compatible():
    from wtpsplit.configs import BertCharConfig, LACanineConfig
    from wtpsplit.models import BertCharForTokenClassification, LACanineForTokenClassification

    assert BertCharConfig.__module__ == "wtpsplit.legacy.configs"
    assert LACanineConfig.__module__ == "wtpsplit.legacy.configs"
    assert BertCharForTokenClassification.__module__ == "wtpsplit.legacy.models"
    assert LACanineForTokenClassification.__module__ == "wtpsplit.legacy.models"


def test_legacy_model_imports_do_not_require_skops():
    code = """
import importlib.util

real_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args: None if name == "skops" else real_find_spec(name, *args)

from wtpsplit.configs import BertCharConfig, LACanineConfig
from wtpsplit.legacy.configs import BertCharConfig as LegacyBertCharConfig
from wtpsplit.models import BertCharForTokenClassification, LACanineForTokenClassification

assert BertCharConfig is LegacyBertCharConfig
assert BertCharForTokenClassification.__module__ == "wtpsplit.legacy.models"
assert LACanineConfig.__module__ == "wtpsplit.legacy.configs"
assert LACanineForTokenClassification.__module__ == "wtpsplit.legacy.models"
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_wtp_uses_canonical_names_and_eager_batches():
    from wtpsplit import WtP

    wtp = WtP("wtp-bert-mini", ignore_legacy_warning=True)
    text = "This is a sentence. This is another sentence."
    canonical = wtp.split(text, language="en", domain="ud")

    with pytest.warns(DeprecationWarning):
        legacy = wtp.split(text, lang_code="en", style="ud")

    assert legacy == canonical
    assert isinstance(wtp.split([text], language="en"), list)
    assert not isinstance(wtp.split([text], language="en", lazy=True), list)


def test_core_import_does_not_require_legacy_dependencies():
    code = """
import importlib.util

real_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args: None if name == "skops" else real_find_spec(name, *args)
import wtpsplit
assert wtpsplit.SaT.__module__ == "wtpsplit"

try:
    from wtpsplit import WtP
except ImportError as error:
    assert "wtpsplit[legacy]" in str(error)
else:
    raise AssertionError("WtP unexpectedly imported without its legacy dependency")
"""
    subprocess.run([sys.executable, "-c", code], check=True)
