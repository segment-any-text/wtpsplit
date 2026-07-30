"""Checkpoint-name resolution for the default operating point.

The threshold used to be picked with `"sm" in model_str` against the whole string, so any
local path containing those two letters selected the `-sm` operating point -- ten times the
base default -- with no warning. These tests pin the resolution to the final path component.
"""

import pytest

from wtpsplit.constants import (
    DEFAULT_SENTENCE_THRESHOLD,
    NO_LOOKAHEAD_SENTENCE_THRESHOLD,
    SM_SENTENCE_THRESHOLD,
    default_threshold_for_checkpoint,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sat-3l", DEFAULT_SENTENCE_THRESHOLD),
        ("sat-12l", DEFAULT_SENTENCE_THRESHOLD),
        ("sat-3l-sm", SM_SENTENCE_THRESHOLD),
        ("sat-12l-sm", SM_SENTENCE_THRESHOLD),
        ("sat-9l-no-limited-lookahead", NO_LOOKAHEAD_SENTENCE_THRESHOLD),
        # `-sm` wins when both suffixes are present, matching the old precedence.
        ("sat-9l-no-limited-lookahead-sm", SM_SENTENCE_THRESHOLD),
    ],
)
def test_released_names_resolve(name, expected):
    assert default_threshold_for_checkpoint(name) == expected


@pytest.mark.parametrize("prefix", ["segment-any-text/", "./", "/opt/models/", "../cache/"])
def test_prefixes_do_not_change_resolution(prefix):
    assert default_threshold_for_checkpoint(f"{prefix}sat-3l-sm") == SM_SENTENCE_THRESHOLD
    assert default_threshold_for_checkpoint(f"{prefix}sat-3l") == DEFAULT_SENTENCE_THRESHOLD


@pytest.mark.parametrize(
    "path",
    [
        "/home/smith/model",  # contains "sm" in a directory name
        "/tmp/transformers-cache/my-model",  # contains "sm" in "transformers"
        "./sms/checkpoint",
        "my-finetune",
        "sat-3l-sm-finetuned",  # a derivative, not a released checkpoint
        "",
    ],
)
def test_unrecognised_names_return_none(path):
    """These must not silently inherit a released operating point."""
    assert default_threshold_for_checkpoint(path) is None


def test_the_specific_regression():
    """A path whose directory contains 'sm' must not pick the -sm threshold."""
    assert default_threshold_for_checkpoint("/home/smith/sat-3l") == DEFAULT_SENTENCE_THRESHOLD


def test_trailing_slashes_and_backslashes():
    assert default_threshold_for_checkpoint("/opt/sat-3l-sm/") == SM_SENTENCE_THRESHOLD
    assert default_threshold_for_checkpoint(r"C:\models\sat-3l-sm") == SM_SENTENCE_THRESHOLD
