"""Execute README examples against signature-validating lightweight doubles."""

from __future__ import annotations

import doctest
import inspect
import os
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

from wtpsplit import DEFAULT_STRIDE, SaT as RealSaT, WtP as RealWtP

ROOT = Path(__file__).resolve().parents[1]
README_PATHS = [ROOT / "README.md", ROOT / "README_WTP.md"]
FENCE_PATTERN = re.compile(r"^```(?P<language>[^\n]*)\n(?P<code>.*?)^```\s*$", re.MULTILINE | re.DOTALL)


class FakeSaT:
    def __init__(self, *args, **kwargs):
        inspect.signature(RealSaT.__init__).bind(None, *args, **kwargs)

    def split(self, text_or_texts, *args, **kwargs):
        inspect.signature(RealSaT.split).bind(self, text_or_texts, *args, **kwargs)
        if isinstance(text_or_texts, str):
            sentences = [text_or_texts]
            return [sentences] if kwargs.get("do_paragraph_segmentation") else sentences
        return [[text] for text in text_or_texts]

    def segment(self, text_or_texts, *args, **kwargs):
        inspect.signature(RealSaT.segment).bind(self, text_or_texts, *args, **kwargs)
        text = text_or_texts if isinstance(text_or_texts, str) else text_or_texts[0]
        return types.SimpleNamespace(
            text=text,
            sentences=[text],
            spans=[(0, len(text))],
            probabilities=np.zeros(len(text)),
            confidences=[0.0],
        )

    def predict_proba(self, text_or_texts, *args, **kwargs):
        inspect.signature(RealSaT.predict_proba).bind(self, text_or_texts, *args, **kwargs)
        if isinstance(text_or_texts, str):
            return np.zeros(len(text_or_texts))
        return [np.zeros(len(text)) for text in text_or_texts]

    def adapt(self, sentences, *args, **kwargs):
        inspect.signature(RealSaT.adapt).bind(self, sentences, *args, **kwargs)
        self.adaptation_history = [0.1]
        return self

    def save_adapter(self, output_dir):
        inspect.signature(RealSaT.save_adapter).bind(self, output_dir)
        return Path(output_dir)

    def half(self):
        return self

    def to(self, *args, **kwargs):
        return self


class FakeWtP:
    def __init__(self, *args, **kwargs):
        inspect.signature(RealWtP.__init__).bind(None, *args, **kwargs)
        self.mixtures = {}

    def split(self, text_or_texts, *args, **kwargs):
        inspect.signature(RealWtP.split).bind(self, text_or_texts, *args, **kwargs)
        if isinstance(text_or_texts, str):
            sentences = [text_or_texts]
            return [sentences] if kwargs.get("do_paragraph_segmentation") else sentences
        return [[text] for text in text_or_texts]

    def predict_proba(self, text_or_texts, *args, **kwargs):
        inspect.signature(RealWtP.predict_proba).bind(self, text_or_texts, *args, **kwargs)
        if isinstance(text_or_texts, str):
            return np.zeros(len(text_or_texts))
        return [np.zeros(len(text)) for text in text_or_texts]

    def get_threshold(self, *args, **kwargs):
        inspect.signature(RealWtP.get_threshold).bind(self, *args, **kwargs)
        return 0.5

    def half(self):
        return self

    def to(self, *args, **kwargs):
        return self


class FakeAutoModelForTokenClassification:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return types.SimpleNamespace(args=args, kwargs=kwargs)


def install_test_modules() -> None:
    fake_wtpsplit = types.ModuleType("wtpsplit")
    fake_wtpsplit.__path__ = []
    fake_wtpsplit.DEFAULT_STRIDE = DEFAULT_STRIDE
    fake_wtpsplit.SaT = FakeSaT
    fake_wtpsplit.WtP = FakeWtP
    sys.modules["wtpsplit"] = fake_wtpsplit
    sys.modules["wtpsplit.models"] = types.ModuleType("wtpsplit.models")

    fake_legacy = types.ModuleType("wtpsplit.legacy")
    fake_legacy.__path__ = []
    sys.modules["wtpsplit.legacy"] = fake_legacy
    sys.modules["wtpsplit.legacy.models"] = types.ModuleType("wtpsplit.legacy.models")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForTokenClassification = FakeAutoModelForTokenClassification
    sys.modules["transformers"] = fake_transformers

    fake_skops = types.ModuleType("skops")
    fake_skops.__path__ = []
    fake_skops_io = types.ModuleType("skops.io")
    fake_skops_io.load = lambda *args, **kwargs: {}
    fake_skops.io = fake_skops_io
    sys.modules["skops"] = fake_skops
    sys.modules["skops.io"] = fake_skops_io


def infer_language(language: str, code: str) -> str:
    language = language.strip().lower()
    if language:
        return language
    first_command = next((line.strip() for line in code.splitlines() if line.strip()), "")
    if first_command.startswith(("python ", "python3 ", "uv ", "git ")):
        return "bash"
    return "text"


def make_shell_environment(directory: Path) -> dict[str, str]:
    fake_bin = directory / "bin"
    fake_bin.mkdir()
    no_op = "#!/bin/sh\nexit 0\n"
    for command in ["hf", "pip", "python", "python3", "uv"]:
        executable = fake_bin / command
        executable.write_text(no_op, encoding="utf-8")
        executable.chmod(0o755)

    fake_git = fake_bin / "git"
    fake_git.write_text(
        """#!/bin/sh
if [ "$1" = "clone" ]; then
    destination="$3"
    if [ -z "$destination" ]; then
        destination="$(basename "$2" .git)"
    fi
    mkdir -p "$destination"
fi
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)

    environment = os.environ.copy()
    for name in [name for name in environment if name.startswith("BASH_FUNC_")]:
        del environment[name]
    environment.pop("BASH_ENV", None)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    return environment


def execute_readme(path: Path, shell_environment: dict[str, str]) -> dict[str, int]:
    namespace = {
        "__name__": "__readme_example__",
        "SaT": FakeSaT,
        "WtP": FakeWtP,
        "text": "First sentence. Second sentence.",
        "segments": ["First sentence. Second sentence."],
        "sat": FakeSaT("sat-3l-sm"),
        "wtp": FakeWtP("wtp-bert-mini"),
    }
    counts = {"python": 0, "pycon": 0, "bash": 0, "text": 0}

    for index, match in enumerate(FENCE_PATTERN.finditer(path.read_text(encoding="utf-8")), start=1):
        code = match.group("code")
        language = infer_language(match.group("language"), code)
        label = f"{path.name} block {index}"
        if language in {"python", "py"}:
            exec(compile(code, f"{path}:{index}", "exec"), namespace)
            counts["python"] += 1
        elif language in {"pycon", "python-console"}:
            test = doctest.DocTestParser().get_doctest(code, namespace, label, str(path), 0)
            failures, _ = doctest.DocTestRunner().run(test)
            if failures:
                raise AssertionError(f"{label} failed doctest")
            counts["pycon"] += 1
        elif language in {"bash", "sh", "shell", "console"}:
            subprocess.run(["bash", "-eu"], input=code, text=True, check=True, env=shell_environment)
            counts["bash"] += 1
        else:
            if not code.strip():
                raise AssertionError(f"{label} is empty")
            counts["text"] += 1
    return counts


def main() -> None:
    install_test_modules()
    with tempfile.TemporaryDirectory() as temporary_directory:
        previous_cwd = Path.cwd()
        try:
            os.chdir(temporary_directory)
            shell_environment = make_shell_environment(Path(temporary_directory))
            for path in README_PATHS:
                print(f"{path.name}: {execute_readme(path, shell_environment)}")
        finally:
            os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
