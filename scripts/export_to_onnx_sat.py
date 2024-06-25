from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
from onnxruntime.transformers.optimizer import optimize_model  # noqa
from transformers import AutoModelForTokenClassification, HfArgumentParser

import wtpsplit  # noqa
import wtpsplit.models  # noqa


@dataclass
class Args:
    model_name_or_path: str = "segment-any-text/sat-12l-no-limited-lookahead"
    output_dir: str = "sat-12l-no-limited-lookahead"
    device: str = "cpu"
    # TODO: lora merging here


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, force_download=True)
    # model = model.half()  # CUDA ONLY!
    model = model.to(args.device)

    torch.onnx.export(
        model,
        {
            "attention_mask": torch.zeros((1, 14), dtype=torch.long, device=args.device),
            "input_ids": torch.zeros((1, 14), dtype=torch.long, device=args.device),
        },
        output_dir / "model.onnx",
        verbose=True,
        input_names=["attention_mask", "input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"}
        },
        # opset_version=11
    )

    # m = optimize_model(
    #     str(output_dir / "model.onnx"),
    #     model_type="bert",
    #     optimization_options=None,
    #     opt_level=0,
    #     use_gpu=False,
    # )

    # optimized_model_path = output_dir / "model_optimized.onnx"
    # onnx.save_model(m.model, optimized_model_path)

    onnx_model = onnx.load(output_dir / "model.onnx")
    onnx.checker.check_model(onnx_model, full_check=True)