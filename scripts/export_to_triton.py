"""
Script to export SaT models to NVIDIA Triton Inference Server format.

This script:
1. Exports the model to ONNX format (if not already exported)
2. Creates a Triton model repository structure
3. Generates the necessary config.pbtxt for Triton

Usage:
    python scripts/export_to_triton.py \
        --model_name_or_path segment-any-text/sat-3l-sm \
        --output_dir triton_models/sat-3l-sm \
        --triton_model_name sat_3l_sm
"""

from dataclasses import dataclass
from pathlib import Path
import json

import adapters  # noqa
import onnx
import torch
from adapters.models import MODEL_MIXIN_MAPPING  # noqa
from adapters.models.bert.mixin_bert import BertModelAdaptersMixin  # noqa
from huggingface_hub import hf_hub_download
from onnxruntime.transformers.optimizer import optimize_model  # noqa
from transformers import AutoModelForTokenClassification, HfArgumentParser

import wtpsplit  # noqa
import wtpsplit.models  # noqa
from wtpsplit.utils import Constants

MODEL_MIXIN_MAPPING["SubwordXLMRobertaModel"] = BertModelAdaptersMixin


@dataclass
class Args:
    model_name_or_path: str = "segment-any-text/sat-3l-sm"
    output_dir: str = "triton_models/sat-3l-sm"
    triton_model_name: str = "sat_3l_sm"
    triton_version: str = "1"
    device: str = "cuda"
    max_batch_size: int = 32
    use_lora: bool = False
    lora_path: str = None
    style_or_domain: str = "ud"
    language: str = "en"


def create_triton_config(
    model_name: str,
    max_batch_size: int,
    input_shape: list,
    output_shape: list,
) -> str:
    """Create a Triton config.pbtxt file content."""
    config = f"""name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [{input_shape}]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [{input_shape}]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP16
    dims: [{output_shape}]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]

dynamic_batching {{
  preferred_batch_size: [1, 2, 4, 8, 16, 32]
  max_queue_delay_microseconds: 100
}}
"""
    return config


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    output_dir = Path(args.output_dir)
    model_repo_dir = output_dir / args.triton_version
    model_repo_dir.mkdir(exist_ok=True, parents=True)

    print(f"Exporting model to Triton format at {output_dir}")

    # Load and export model to ONNX
    model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, force_download=False)
    model = model.to(args.device)

    # Fetch config.json from huggingface hub
    hf_hub_download(
        repo_id=args.model_name_or_path,
        filename="config.json",
        local_dir=output_dir,
    )

    # LoRA SETUP
    if args.use_lora:
        model_type = model.config.model_type
        model.config.model_type = "xlm-roberta"
        adapters.init(model)
        model.config.model_type = model_type
        
        if not args.lora_path:
            for file in [
                "adapter_config.json",
                "head_config.json",
                "pytorch_adapter.bin",
                "pytorch_model_head.bin",
            ]:
                hf_hub_download(
                    repo_id=args.model_name_or_path,
                    subfolder=f"loras/{args.style_or_domain}/{args.language}",
                    filename=file,
                    local_dir=Constants.CACHE_DIR,
                )
            lora_load_path = str(Constants.CACHE_DIR / "loras" / args.style_or_domain / args.language)
        else:
            lora_load_path = args.lora_path

        print(f"Using LoRA weights from {lora_load_path}.")
        model.load_adapter(
            lora_load_path,
            set_active=True,
            with_head=True,
            load_as="sat-lora",
        )
        model.merge_adapter("sat-lora")
        print("LoRA setup done.")

    model = model.half()

    # Export to ONNX
    onnx_path = model_repo_dir / "model.onnx"
    print(f"Exporting model to ONNX at {onnx_path}")
    
    dummy_inputs = (
        torch.randint(0, model.config.vocab_size, (1, 512), dtype=torch.int64, device=args.device),
        torch.ones((1, 512), dtype=torch.int64, device=args.device),
    )

    torch.onnx.export(
        model,
        dummy_inputs,
        onnx_path,
        verbose=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17,  # Use opset 17 for compatibility with Triton's ONNX Runtime
        dynamo=False,  # Disable dynamo to ensure dynamic_axes are respected
    )
    
    # Downgrade IR version for Triton compatibility (Triton 24.03 supports max IR version 9)
    onnx_model_tmp = onnx.load(onnx_path)
    if onnx_model_tmp.ir_version > 9:
        print(f"Downgrading ONNX IR version from {onnx_model_tmp.ir_version} to 9 for Triton compatibility")
        onnx_model_tmp.ir_version = 9
        onnx.save(onnx_model_tmp, onnx_path)

    # Optimize ONNX model
    print("Optimizing ONNX model...")
    m = optimize_model(
        str(onnx_path),
        model_type="bert",
        num_heads=0,
        hidden_size=0,
        optimization_options=None,
        opt_level=0,
        use_gpu=True,
    )

    # Add Microsoft domain opset import (required for BiasGelu and other MS operators)
    # Check if com.microsoft domain is already present
    has_ms_domain = any(
        opset.domain == "com.microsoft" for opset in m.model.opset_import
    )
    if not has_ms_domain:
        ms_opset = onnx.helper.make_opsetid("com.microsoft", 1)
        m.model.opset_import.append(ms_opset)
        print("Added com.microsoft opset import for Microsoft-specific operators")

    # Save optimized model
    optimized_onnx_path = model_repo_dir / "model.onnx"
    onnx.save_model(m.model, optimized_onnx_path)
    print(f"Saved optimized ONNX model to {optimized_onnx_path}")

    # Verify ONNX model
    onnx_model = onnx.load(optimized_onnx_path)
    print("Checking ONNX model...")
    # Note: full_check=False because Microsoft operators are not in the standard ONNX spec
    # The model will still work correctly with ONNX Runtime which supports these operators
    try:
        onnx.checker.check_model(onnx_model, full_check=True)
        print("ONNX model is valid!")
    except onnx.checker.ValidationError as e:
        if "com.microsoft" in str(e):
            print("ONNX model uses Microsoft-specific operators (e.g., BiasGelu).")
            print("This is expected and the model will work with ONNX Runtime.")
            # Perform basic check without full validation of custom ops
            onnx.checker.check_model(onnx_model, full_check=False)
            print("Basic ONNX model structure is valid!")
        else:
            raise

    # Create Triton config
    config_path = output_dir / "config.pbtxt"
    config_content = create_triton_config(
        model_name=args.triton_model_name,
        max_batch_size=args.max_batch_size,
        input_shape="-1",  # Dynamic sequence length
        output_shape="-1, 1",  # Dynamic sequence length, 1 output class (binary segmentation)
    )
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"Created Triton config at {config_path}")
    print(f"\nTriton model repository structure created successfully!")
    print(f"\nTo use this model with Triton:")
    print(f"1. Copy {output_dir} to your Triton model repository")
    print(f"2. Start Triton server with the model repository")
    print(f"3. Use the following code to run inference:")
    print(f"""
from wtpsplit import SaT

sat = SaT(
    "{args.model_name_or_path}",
    triton_url="localhost:8001",
    triton_model_name="{args.triton_model_name}"
)
sat.split("This is a test. This is another sentence.")
""")
