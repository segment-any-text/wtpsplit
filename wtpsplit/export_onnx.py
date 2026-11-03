"""Export an XLM-R SaT checkpoint as a standalone FP32 ONNX directory.

Run ``python -m wtpsplit.export_onnx --help``. Export dependencies are optional;
serving the resulting directory needs only the ONNX Runtime extra.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from wtpsplit import SaT
from wtpsplit.char_head import build_char_inputs
from wtpsplit.constants import DEFAULT_SENTENCE_THRESHOLD, default_threshold_for_checkpoint
from wtpsplit.onnx import cpu_session_options

TOKEN_INPUTS = ["input_ids", "attention_mask"]
CHAR_INPUTS = ["char_to_token", "char_is_token_final", "char_position_in_token", "char_hashes", "char_mask"]


class _Export(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, char_to_token=None, char_is_token_final=None,
                char_position_in_token=None, char_hashes=None, char_mask=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            char_to_token=char_to_token, char_is_token_final=char_is_token_final,
            char_position_in_token=char_position_in_token, char_hashes=char_hashes,
            char_mask=char_mask,
        ).logits


def _inputs(tokenizer, texts, character_head):
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    values = (encoded["input_ids"], encoded["attention_mask"].float())
    if character_head:
        raw = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
        chars = build_char_inputs(texts, raw["offset_mapping"])
        values += (chars.char_to_token + 1, chars.is_token_final, chars.position_in_token, chars.hashes, chars.mask)
    return values


def export_onnx(model_name_or_path, output_dir, *, tokenizer_name_or_path=None, threads=4):
    """Load a fresh CPU model, export and check dynamic shapes, then publish locally.

    The output must not exist. This function never modifies the source checkpoint
    or an in-memory SaT instance. Supports XLM-R token and character heads;
    ModernBERT and experimental compression exports are not supported here.
    Numerical smoke checks are not multilingual quality validation.
    """
    import onnx
    import onnxruntime as ort
    import transformers

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    options = cpu_session_options(threads)
    sat = SaT(model_name_or_path, tokenizer_name_or_path=tokenizer_name_or_path,
              device="cpu", from_pretrained_kwargs={"attn_implementation": "eager"})
    model = sat.model.model.float().eval()
    if model.config.model_type != "xlm-token":
        raise ValueError("FP32 export currently supports ordinary XLM-R SaT token/character heads only")
    character_head = bool(getattr(model.config, "use_character_head", False))
    names = TOKEN_INPUTS + (CHAR_INPUTS if character_head else [])
    wrapper = _Export(model).eval()
    examples = [["This is a test. Another sentence.", "中文。另一句。"],
                ["Short."], ["Longer text. " * 19, "བོད་ hello. 😀", "hello world."]]
    axes = {name: {0: "batch", 1: "tokens" if name in TOKEN_INPUTS else "characters"} for name in names}
    axes["logits"] = {0: "batch", 1: "characters" if character_head else "tokens"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sat-onnx-", dir=output.parent) as staging:
        directory = Path(staging)
        graph = directory / "model_optimized.onnx"
        with torch.inference_mode():
            torch.onnx.export(wrapper, _inputs(sat.tokenizer, examples[0], character_head), str(graph),
                              input_names=names, output_names=["logits"], dynamic_axes=axes,
                              opset_version=17, dynamo=False)
        onnx.checker.check_model(str(graph))
        session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
        checks = []
        for texts in examples:
            values = _inputs(sat.tokenizer, texts, character_head)
            with torch.inference_mode():
                expected = wrapper(*values).numpy()
            actual = session.run(["logits"], dict(zip(names, (value.numpy() for value in values))))[0]
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-4)
            checks.append({"batch": len(texts), "tokens": values[0].shape[1],
                           "output_shape": list(actual.shape),
                           "max_abs_logit_error": float(np.max(np.abs(actual - expected)))})
        del session
        # Moving a released checkpoint into an arbitrary export directory must
        # not silently change split()'s default operating point.
        model.config.sat_export_default_threshold = getattr(
            model.config, "sat_export_default_threshold", None)
        if model.config.sat_export_default_threshold is None:
            resolved = default_threshold_for_checkpoint(str(model_name_or_path))
            model.config.sat_export_default_threshold = (
                DEFAULT_SENTENCE_THRESHOLD if resolved is None else resolved)
        model.config.sat_export_source_checkpoint = str(model_name_or_path)
        model.config.save_pretrained(directory)
        sat.tokenizer.save_pretrained(directory)
        manifest = {"source": str(model_name_or_path), "precision": "float32", "opset": 17,
                    "character_head": character_head, "inputs": names,
                    "default_sentence_threshold": model.config.sat_export_default_threshold,
                    "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                                 "onnx": onnx.__version__, "onnxruntime": ort.__version__},
                    "checks": checks, "files_sha256": {}}
        for path in sorted(directory.iterdir()):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                manifest["files_sha256"][path.name] = digest.hexdigest()
        (directory / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Refuse a late collision too; never replace someone else's export.
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        directory.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local checkpoint or SaT model name")
    parser.add_argument("output", help="New output directory")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    print(export_onnx(args.model, args.output, tokenizer_name_or_path=args.tokenizer, threads=args.threads))


if __name__ == "__main__":
    main()
