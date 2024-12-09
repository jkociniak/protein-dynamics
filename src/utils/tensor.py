import torch


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
    assert x.device == y.device
    assert x.shape[0] == y.shape[0]
    assert x.shape[:-1] == y.shape[:-1]

    all_grads = torch.zeros((*y.shape[:-1], y.shape[-1], x.shape[-1]), device=x.device)
    for i, yi in enumerate(y.unbind(dim=-1)):
        o = torch.autograd.grad(yi, x,
                                  grad_outputs=torch.ones_like(yi),
                                  create_graph=True)[0]
        #print(o)
        all_grads[..., i, :] = o

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
