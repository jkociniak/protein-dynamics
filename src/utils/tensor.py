import torch
import matplotlib.pyplot as plt
import numpy as np


def rot_matrix_2d(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.tensor(((c, -s), (s, c)))
    return R


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def gradients(y, x):
    # Euclidean case
    # x dim = (batch, ...,  n)
    # y dim = (batch, ..., m)
    #assert len(x.shape) == 2 and len(y.shape) == 2
    assert x.shape[0] == y.shape[0]
    assert x.shape[:-1] == y.shape[:-1]

    all_grads = torch.zeros((*y.shape[:-1], y.shape[-1], x.shape[-1]))
    for i, yi in enumerate(y.unbind(dim=-1)):
        all_grads[..., i, :] = torch.autograd.grad(yi, x,
                                  grad_outputs=torch.ones_like(yi),
                                  create_graph=True)[0]

    return all_grads


def gradients_pc(y, x):
    # Pointcloud case
    # x dim = (batches * vertices, vertex_dim)
    # y dim = (batches, output_dim)
    assert len(x.shape) == 2 and len(y.shape) == 2
    n_vertices = x.shape[0] // y.shape[0]

    all_grads = torch.zeros((y.shape[0], y.shape[1], n_vertices, x.shape[1]))
    for i, yi in enumerate(y.unbind(dim=1)):
        tmp = torch.autograd.grad(yi, x,
                                  grad_outputs=torch.ones_like(yi),
                                  create_graph=True)[0]
        all_grads[:, i, :, :] = tmp.view(y.shape[0], n_vertices, x.shape[1])

    return all_grads


def validate_tensor(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError("NaN in " + name)
    elif torch.isinf(tensor).any():
        raise ValueError("Inf in " + name)



def directional_div(points, grads):
    dot_grad = (grads * grads).sum(dim=-1, keepdim=True)
    hvp = torch.ones_like(dot_grad)
    hvp = 0.5 * torch.autograd.grad(dot_grad, points, hvp, retain_graph=True, create_graph=True)[0]
    div = (grads * hvp).sum(dim=-1) / (torch.sum(grads ** 2, dim=-1) + 1e-5)
    return div


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.show()