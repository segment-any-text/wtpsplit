import onnxruntime
from tqdm.auto import tqdm


def create_session(path, use_cuda):
    session = onnxruntime.InferenceSession(path)

    # onnxruntime automatically prioritizes GPU if supported
    # if use_cuda=True force it to error if GPU is not available
    if use_cuda is not None:
        if use_cuda:
            session.set_providers(["CUDAExecutionProvider"])
        else:
            session.set_providers(["CPUExecutionProvider"])

    return session


def predict_batch(session, inputs):
    return session.run(None, {"input": inputs})[0]


def get_metadata(session):
    return session.get_modelmeta().custom_metadata_map


def get_progress_bar(total):
    return tqdm(total=total)