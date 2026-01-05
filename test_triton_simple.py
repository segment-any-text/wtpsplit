#!/usr/bin/env python
"""Simple test to verify Triton integration works correctly."""

import sys


def test_triton_imports():
    """Test that all Triton-related imports work."""
    print("Testing imports...")
    try:
        from wtpsplit.extract import SaTTritonWrapper
        from wtpsplit import SaT
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_triton_wrapper_creation():
    """Test creating a Triton wrapper with mock objects."""
    print("\nTesting Triton wrapper creation...")
    try:
        from wtpsplit.extract import SaTTritonWrapper
        from unittest.mock import Mock
        from transformers import AutoConfig
        
        # Create mock objects
        mock_client = Mock()
        mock_config = AutoConfig.from_pretrained("segment-any-text/sat-3l-sm")
        
        # Create wrapper
        wrapper = SaTTritonWrapper(mock_config, mock_client, "test_model")
        
        assert wrapper.config == mock_config
        assert wrapper.triton_client == mock_client
        assert wrapper.model_name == "test_model"
        
        print("✓ Triton wrapper creation successful")
        return True
    except Exception as e:
        print(f"✗ Wrapper creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sat_triton_params():
    """Test SaT initialization with Triton parameters."""
    print("\nTesting SaT initialization with Triton parameters...")
    try:
        from wtpsplit import SaT
        
        # Test that triton_url without triton_model_name raises error
        try:
            sat = SaT("sat-3l-sm", triton_url="localhost:8001")
            print("✗ Should have raised ValueError for missing triton_model_name")
            return False
        except ValueError as e:
            if "triton_model_name" in str(e):
                print("✓ Correctly raises error when triton_model_name is missing")
            else:
                print(f"✗ Wrong error message: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ SaT parameter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_export_script_exists():
    """Test that the export script exists and can be imported."""
    print("\nTesting export script...")
    try:
        import os
        script_path = "scripts/export_to_triton.py"
        if os.path.exists(script_path):
            print(f"✓ Export script exists at {script_path}")
            return True
        else:
            print(f"✗ Export script not found at {script_path}")
            return False
    except Exception as e:
        print(f"✗ Export script test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Triton Integration Tests")
    print("=" * 60)
    
    tests = [
        test_triton_imports,
        test_triton_wrapper_creation,
        test_sat_triton_params,
        test_export_script_exists,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {len(results) - sum(results)} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
