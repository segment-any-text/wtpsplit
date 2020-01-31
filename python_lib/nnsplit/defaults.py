import torch

# see train.py for training defaults

DEFAULT_CUT_LENGTH = 100
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
