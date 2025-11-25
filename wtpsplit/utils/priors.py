import numpy as np

def create_prior_function(name, kwargs):
    if name == "uniform":
        max_length = kwargs.get("max_length")
        def prior(length):
            if max_length is not None and length > max_length:
                return 0.0
            return 1.0
        return prior
    
    elif name == "clipped_polynomial":
        alpha = kwargs.get("alpha", 0.5)
        mu = kwargs.get("mu", 3.0)

        def prior(length):
            val = 1.0 - alpha * ((length - mu) ** 2)
            return max(val, 0.0)
        return prior

    elif name == "gaussian":
        mu = kwargs.get("mu", 20.0)
        sigma = kwargs.get("sigma", 5.0)
        
        def prior(length):
            return np.exp(-0.5 * ((length - mu) / sigma) ** 2)
        return prior

    else:
        raise ValueError(f"Unknown prior: {name}")
