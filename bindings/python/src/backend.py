import onnxruntime
from tqdm.auto import tqdm


def create_session(path, use_cuda):
    # onnxruntime automatically prioritizes GPU if supported
    # if use_cuda=True force it to error if GPU is not available
    
    if use_cuda is None:
        providers = [provider for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider in onnxruntime.get_available_providers()]
    else:
        if use_cuda:
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

    session = onnxruntime.InferenceSession(path, providers=providers)

    return session


def predict_batch(session, inputs):
    return session.run(None, {"input": inputs})[0]


def get_metadata(session):
    return session.get_modelmeta().custom_metadata_map


def get_progress_bar(total):
    return tqdm(total=total)
