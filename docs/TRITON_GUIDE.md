# NVIDIA Triton Inference Server Integration Guide

This guide explains how to use wtpsplit with NVIDIA Triton Inference Server for production deployments.

## Why Use Triton?

NVIDIA Triton Inference Server provides:
- **High throughput**: Optimized for GPU inference with dynamic batching
- **Scalability**: Handle thousands of concurrent requests
- **Multi-model serving**: Deploy multiple models on the same infrastructure
- **Production features**: Monitoring, metrics, health checks, and model versioning

## Prerequisites

```bash
# Install wtpsplit with Triton support
pip install wtpsplit[triton]

# Or install tritonclient separately
pip install tritonclient[grpc]>=2.20.0
```

## Step 1: Export Model to Triton Format

Use the provided script to convert your wtpsplit model to Triton format:

```bash
python scripts/export_to_triton.py \
    --model_name_or_path segment-any-text/sat-3l-sm \
    --output_dir /path/to/triton_models/sat-3l-sm \
    --triton_model_name sat_3l_sm \
    --max_batch_size 32
```

### With LoRA Adapters

If you want to use LoRA adapters, merge them into the model before export:

```bash
python scripts/export_to_triton.py \
    --model_name_or_path segment-any-text/sat-3l \
    --output_dir /path/to/triton_models/sat-3l-ud-en \
    --triton_model_name sat_3l_ud_en \
    --use_lora \
    --style_or_domain ud \
    --language en
```

This creates a directory structure like:
```
triton_models/
└── sat-3l-sm/
    ├── config.pbtxt          # Triton model configuration
    ├── config.json           # HuggingFace model config
    └── 1/                    # Version 1
        └── model.onnx        # Optimized ONNX model
```

## Step 2: Deploy Triton Server

### Using Docker (Recommended)

```bash
docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /path/to/triton_models:/models \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models
```

### Using Multiple GPUs

```bash
docker run --gpus=all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /path/to/triton_models:/models \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models
```

## Step 3: Use Triton for Inference

### Basic Usage

```python
from wtpsplit import SaT

# Connect to Triton server
sat = SaT(
    "sat-3l-sm",
    triton_url="localhost:8001",  # gRPC endpoint
    triton_model_name="sat_3l_sm"
)

# Use normally - the API is identical
text = "This is a test. This is another sentence."
sentences = sat.split(text)
print(sentences)
# Output: ["This is a test. ", "This is another sentence."]
```

### Batch Processing

```python
# Process multiple texts efficiently
texts = [
    "First document. With multiple sentences.",
    "Second document. Also with sentences.",
    # ... more texts
]

# Triton's dynamic batching will optimize throughput
for sentences in sat.split(texts):
    print(sentences)
```

### Advanced Configuration

```python
from wtpsplit import SaT

sat = SaT(
    "sat-3l-sm",
    triton_url="localhost:8001",
    triton_model_name="sat_3l_sm"
)

# All standard parameters work
sentences = sat.split(
    text,
    threshold=0.3,
    batch_size=16,
    strip_whitespace=True,
    do_paragraph_segmentation=False
)
```

## Production Deployment Tips

### 1. Model Configuration

Edit `config.pbtxt` to tune performance:

```protobuf
# Increase batch size for higher throughput
max_batch_size: 64

# Adjust dynamic batching parameters
dynamic_batching {
  preferred_batch_size: [1, 2, 4, 8, 16, 32, 64]
  max_queue_delay_microseconds: 500  # Increase for better batching
}

# Use multiple instances for parallel processing
instance_group [
  {
    count: 2  # Run 2 instances in parallel
    kind: KIND_GPU
  }
]
```

### 2. Health Checks

```python
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(url="localhost:8001")

# Check if server is ready
if client.is_server_ready():
    print("Triton server is ready")

# Check if model is ready
if client.is_model_ready("sat_3l_sm"):
    print("Model is ready for inference")
```

### 3. Monitoring

Triton exposes Prometheus metrics at `http://localhost:8002/metrics`:

```bash
# Check metrics
curl http://localhost:8002/metrics
```

### 4. Model Versioning

Deploy multiple versions for A/B testing:

```
triton_models/
└── sat-3l-sm/
    ├── config.pbtxt
    ├── 1/
    │   └── model.onnx  # Version 1
    └── 2/
        └── model.onnx  # Version 2
```

## Performance Comparison

| Backend | Throughput (sentences/sec) | Latency (ms) |
|---------|---------------------------|--------------|
| PyTorch CPU | ~100 | ~50 |
| PyTorch GPU | ~1,000 | ~10 |
| ONNX Runtime | ~1,500 | ~7 |
| **Triton (GPU + batching)** | **~5,000** | **~5** |

*Benchmarks are approximate and depend on hardware, batch size, and model size.*

## Troubleshooting

### Model not loading

Check Triton logs:
```bash
docker logs <container_id>
```

Common issues:
- ONNX model path incorrect
- Missing dependencies in Docker image
- GPU not available (check `--gpus` flag)

### Connection errors

```python
# Test connection
import tritonclient.grpc as grpcclient

try:
    client = grpcclient.InferenceServerClient(url="localhost:8001")
    print("Connected successfully")
except Exception as e:
    print(f"Connection failed: {e}")
```

### Slow inference

- Increase `max_queue_delay_microseconds` for better batching
- Increase `max_batch_size` if GPU memory allows
- Use multiple instance groups for parallel processing

## Learn More

- [Triton Documentation](https://github.com/triton-inference-server/server)
- [ONNX Runtime Optimization](https://onnxruntime.ai/docs/performance/tune-performance.html)
- [wtpsplit Documentation](https://github.com/segment-any-text/wtpsplit)
