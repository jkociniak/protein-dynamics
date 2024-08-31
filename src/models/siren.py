import torch
import torch.nn as nn
import numpy as np

from ..utils.tensor import validate_tensor


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        out1 = self.linear(input)
        out = torch.sin(self.omega_0 * out1)
        return out

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30., hidden_omega_0=30., first_hidden=None):
        super().__init__()

        assert hidden_layers > 0
        self.hparams = {
            'in_features': in_features,
            'hidden_features': hidden_features,
            'hidden_layers': hidden_layers,
            'out_features': out_features,
            'outermost_linear': outermost_linear,
            'first_omega_0': first_omega_0,
            'hidden_omega_0': hidden_omega_0
        }

        self.net = []
        if first_hidden is not None:
            self.net.append(SineLayer(in_features, first_hidden,
                                      is_first=True, omega_0=first_omega_0))

            self.net.append(SineLayer(first_hidden, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

            for i in range(hidden_layers-2):
                self.net.append(SineLayer(hidden_features, hidden_features,
                                          is_first=False, omega_0=hidden_omega_0))
        else:
            self.net.append(SineLayer(in_features, hidden_features,
                                      is_first=True, omega_0=first_omega_0))

            for i in range(hidden_layers-1):
                self.net.append(SineLayer(hidden_features, hidden_features,
                                          is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            # with torch.no_grad():
            #     final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
            #                                   np.sqrt(6 / hidden_features) / hidden_omega_0)

            # final_linear2 = nn.Linear(hidden_features // 2, out_features)
            #
            # with torch.no_grad():
            #     final_linear2.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
            #                                   np.sqrt(6 / hidden_features) / hidden_omega_0)
            #
            # final_linear3 = nn.Linear(hidden_features, out_features)
            #
            # with torch.no_grad():
            #     final_linear3.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
            #                                   np.sqrt(6 / hidden_features) / hidden_omega_0)
            #
            relu_head = nn.Sequential(final_linear)
            self.net.append(relu_head)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords


class SirenGELU(nn.Module):
    def __init__(self,
                 in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=False,
                 embedding_type=False, head_layers=0,
                 first_omega_0=30., hidden_omega_0=30.):
        super().__init__()

        self.hparams = {
            'in_features': in_features,
            'hidden_features': hidden_features,
            'hidden_layers': hidden_layers,
            'out_features': out_features,
            'embedding_type': embedding_type,
            'head_layers': head_layers,
            'outermost_linear': outermost_linear,
            'first_omega_0': first_omega_0,
            'hidden_omega_0': hidden_omega_0
        }

        self.emb_gelu = None
        self.emb_sine = None
        if embedding_type == 'gelu':
            self.emb_gelu = nn.Sequential(nn.Linear(in_features, hidden_features), nn.GELU())
        elif embedding_type == 'sine':
            self.emb_sine = SineLayer(in_features, hidden_features,
                                      is_first=True, omega_0=first_omega_0)
        elif embedding_type == 'combined':
            self.emb_gelu = nn.Sequential(nn.Linear(in_features, hidden_features), nn.GELU())
            self.emb_sine = SineLayer(in_features, hidden_features,
                                      is_first=True, omega_0=first_omega_0)
        else:
            raise ValueError('Invalid embedding type')

        sine_layers = []
        for i in range(hidden_layers-1):
            sine_layers.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.sine_net = nn.Sequential(*sine_layers)
        #self.sine_head = SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0)

        head = []
        if outermost_linear:
            for i in range(head_layers-1):
                head.append(nn.GELU())
                linear = nn.Linear(hidden_features, hidden_features // 2)
                with torch.no_grad():
                    linear.weight.uniform_(-np.sqrt(6 / (hidden_features // 2)) / hidden_omega_0,
                                            np.sqrt(6 / (hidden_features // 2)) / hidden_omega_0)
                head.append(linear)

            last_in = hidden_features // 2 if head_layers > 1 else hidden_features
            linear = nn.Linear(last_in, out_features)
            with torch.no_grad():
                linear.weight.uniform_(-np.sqrt(6 / (hidden_features // 2)) / hidden_omega_0,
                                        np.sqrt(6 / (hidden_features // 2)) / hidden_omega_0)

            head.append(linear)
        else:
            head.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.head = nn.Sequential(*head)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        if self.emb_gelu is not None and self.emb_sine is not None:
            emb = self.emb_gelu(coords) + self.emb_sine(coords)
        elif self.emb_gelu is None:
            emb = self.emb_sine(coords)
        else:
            emb = self.emb_gelu(coords)
        x = self.sine_net(emb)
        output = self.head(x)
        return output, coords


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def directional_div(points, grads):
    dot_grad = (grads * grads).sum(dim=-1, keepdim=True)
    hvp = torch.ones_like(dot_grad)
    hvp = 0.5 * torch.autograd.grad(dot_grad, points, hvp, retain_graph=True, create_graph=True)[0]
    div = (grads * hvp).sum(dim=-1) / (torch.sum(grads ** 2, dim=-1) + 1e-5)
    return div