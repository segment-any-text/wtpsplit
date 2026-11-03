"""Optional ONNX Runtime helpers. Importing this module does not load ORT."""


def cpu_session_options(threads: int = 4, *, allow_spinning: bool = False):
    """Create options for one CPU session without changing process-wide settings.

    ``threads`` includes the calling thread. Budget threads across concurrent
    sessions/workers. Disabling idle spinning reduces contention; it is not a
    guarantee of lower latency for every workload.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for pool in ("intra_op", "inter_op"):
        options.add_session_config_entry(f"session.{pool}.allow_spinning", "1" if allow_spinning else "0")
    return options
