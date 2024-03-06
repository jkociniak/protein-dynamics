import torch


def validate_tensor(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError("NaN or Inf in " + name)