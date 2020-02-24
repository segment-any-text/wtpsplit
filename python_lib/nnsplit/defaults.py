import torch

# see train.py for training defaults

CUT_LENGTH = 500
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
