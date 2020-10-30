import onnx
from onnx import helper


def postprocess(model_path, metadata={}):
    model = onnx.load(model_path)
    helper.set_model_props(model, metadata)

    onnx.save(model, model_path)
