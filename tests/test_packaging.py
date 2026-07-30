"""Guards on what actually ships in the wheel.

Editable installs mask packaging mistakes: the working tree is always importable,
so a broken ``wheel-exclude`` glob or missing package data only shows up for users
who install from PyPI. These tests build the real artifact and inspect it.
"""

import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DATA_FILES = [
    "wtpsplit/data/punctuation.txt",
    "wtpsplit/data/punctuation_xlmr.txt",
    "wtpsplit/data/punctuation_xlmr_unk.txt",
    "wtpsplit/data/punctuation.json",
    "wtpsplit/data/language_info.csv",
    "wtpsplit/data/sentence_stats.json",
]

RESEARCH_PREFIXES = [
    "wtpsplit/train/",
    "wtpsplit/evaluation/",
    "wtpsplit/data_acquisition/",
]


@pytest.fixture(scope="module")
def wheel_contents(tmp_path_factory):
    if shutil.which("uv") is None:
        pytest.skip("uv is required to build the wheel")

    out_dir = tmp_path_factory.mktemp("dist")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )

    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    with zipfile.ZipFile(wheels[0]) as zf:
        return set(zf.namelist())


@pytest.mark.parametrize("data_file", EXPECTED_DATA_FILES)
def test_package_data_is_shipped(wheel_contents, data_file):
    assert data_file in wheel_contents


def test_research_code_is_not_shipped(wheel_contents):
    # uv_build still emits bare directory entries for fully-excluded trees; only
    # actual files would make the modules importable.
    leaked = sorted(
        name
        for name in wheel_contents
        if not name.endswith("/") and any(name.startswith(prefix) for prefix in RESEARCH_PREFIXES)
    )
    assert not leaked, f"research-only modules leaked into the wheel: {leaked}"


def test_runtime_cache_is_not_shipped(wheel_contents):
    leaked = sorted(name for name in wheel_contents if name.startswith("wtpsplit/.cache/") and not name.endswith("/"))
    assert not leaked, f"runtime cache files leaked into the wheel: {leaked}"


@pytest.mark.parametrize("package", ["train", "evaluation", "data_acquisition"])
def test_research_packages_are_not_importable(wheel_contents, package):
    assert f"wtpsplit/{package}/__init__.py" not in wheel_contents


def test_research_environment_is_source_only():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["dependency-groups"]["research"]
    assert "research" not in pyproject["project"]["optional-dependencies"]


def test_core_modules_are_shipped(wheel_contents):
    for module in [
        "wtpsplit/__init__.py",
        "wtpsplit/__init__.pyi",
        "wtpsplit/adaptation.py",
        "wtpsplit/_inference.py",
        "wtpsplit/extract.py",
        "wtpsplit/model_registry.py",
        "wtpsplit/models.py",
        "wtpsplit/legacy/wtp.py",
    ]:
        assert module in wheel_contents


def test_py_typed_marker_is_shipped(wheel_contents):
    assert "wtpsplit/py.typed" in wheel_contents
