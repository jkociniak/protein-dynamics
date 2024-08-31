import torch


def rot_matrix_2d(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.tensor(((c, -s), (s, c)))
    return R
