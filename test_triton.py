"""
Tests for NVIDIA Triton Inference Server integration.

Note: These tests require a running Triton Inference Server.
To run these tests:
1. Export a model using scripts/export_to_triton.py
2. Start Triton server with the model
3. Run: pytest test_triton.py
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch


def test_triton_wrapper_import():
    """Test that the Triton wrapper can be imported."""
    from wtpsplit.extract import SaTTritonWrapper
    assert SaTTritonWrapper is not None


def test_triton_wrapper_initialization():
    """Test Triton wrapper initialization with mock client."""
    from wtpsplit.extract import SaTTritonWrapper
    from transformers import AutoConfig
    
    # Mock Triton client
    mock_client = Mock()
    mock_config = AutoConfig.from_pretrained("segment-any-text/sat-3l-sm")
    
    wrapper = SaTTritonWrapper(mock_config, mock_client, "test_model")
    
    assert wrapper.config == mock_config
    assert wrapper.triton_client == mock_client
    assert wrapper.model_name == "test_model"


def test_triton_wrapper_inference():
    """Test Triton wrapper inference with mock response."""
    from wtpsplit.extract import SaTTritonWrapper
    from transformers import AutoConfig
    
    # Mock config
    mock_config = AutoConfig.from_pretrained("segment-any-text/sat-3l-sm")
    
    # Mock Triton client and response
    mock_client = Mock()
    mock_response = Mock()
    mock_logits = np.random.randn(1, 10, 2).astype(np.float32)
    mock_response.as_numpy.return_value = mock_logits
    mock_client.infer.return_value = mock_response
    
    wrapper = SaTTritonWrapper(mock_config, mock_client, "test_model")
    
    # Test inference
    input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int64)
    attention_mask = np.ones((1, 10), dtype=np.float16)
    
    with patch('wtpsplit.extract.grpcclient') as mock_grpc:
        # Mock InferInput and InferRequestedOutput
        mock_infer_input = Mock()
        mock_infer_output = Mock()
        mock_grpc.InferInput.return_value = mock_infer_input
        mock_grpc.InferRequestedOutput.return_value = mock_infer_output
        
        result = wrapper(input_ids, attention_mask)
    
    assert "logits" in result
    assert result["logits"].shape == mock_logits.shape


@pytest.mark.skip(reason="Requires running Triton server")
def test_sat_with_triton_integration():
    """
    Integration test for SaT with Triton.
    
    This test requires:
    1. A running Triton server at localhost:8001
    2. A model named 'sat_3l_sm' deployed on the server
    
    To set up:
        python scripts/export_to_triton.py \
            --model_name_or_path segment-any-text/sat-3l-sm \
            --output_dir triton_models/sat-3l-sm \
            --triton_model_name sat_3l_sm
        
        docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
            -v $(pwd)/triton_models:/models \
            nvcr.io/nvidia/tritonserver:23.10-py3 \
            tritonserver --model-repository=/models
    """
    from wtpsplit import SaT
    
    sat = SaT(
        "sat-3l-sm",
        triton_url="localhost:8001",
        triton_model_name="sat_3l_sm"
    )
    
    text = "This is a test sentence. This is another sentence."
    splits = sat.split(text, threshold=0.25)
    
    assert isinstance(splits, list)
    assert len(splits) > 0
    assert "".join(splits) == text


def test_sat_triton_initialization_error_no_client():
    """Test that SaT raises error when tritonclient is not installed."""
    from wtpsplit import SaT
    
    with patch('wtpsplit.extract.grpcclient', side_effect=ImportError):
        with pytest.raises(ValueError, match="Please install `tritonclient"):
            sat = SaT(
                "sat-3l-sm",
                triton_url="localhost:8001",
                triton_model_name="sat_3l_sm"
            )


def test_sat_triton_requires_model_name():
    """Test that SaT raises error when triton_url is set but triton_model_name is not."""
    from wtpsplit import SaT
    
    with pytest.raises(ValueError, match="Please specify a `triton_model_name`"):
        sat = SaT(
            "sat-3l-sm",
            triton_url="localhost:8001"
        )


def test_sat_triton_no_lora_support():
    """Test that SaT raises error when trying to use LoRA with Triton."""
    from wtpsplit import SaT
    
    with patch('tritonclient.grpc.InferenceServerClient') as mock_client_class:
        mock_client = Mock()
        mock_client.is_model_ready.return_value = True
        mock_client_class.return_value = mock_client
        
        with pytest.raises(ValueError, match="LoRA is not supported with Triton"):
            sat = SaT(
                "sat-3l-sm",
                triton_url="localhost:8001",
                triton_model_name="sat_3l_sm",
                lora_path="/some/path"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
