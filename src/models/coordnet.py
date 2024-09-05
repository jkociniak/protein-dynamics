import torch
import torch.nn as nn
import numpy as np


class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        with torch.no_grad():
            self.linear.weight.uniform_(-1 / in_features,
                                         1 / in_features)

    def forward(self, input):
        out = torch.relu(self.linear(input))
        return out


class GeLULayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / in_features,
                                         1 / in_features)
        self.gelu = nn.GELU()

    # def init_weights_sine(self):
    #     with torch.no_grad():
    #         self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / 15.,
    #                                      np.sqrt(6 / self.in_features) / 15.)

    def forward(self, input):
        out = self.gelu(self.linear(input))
        return out


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30., init='mfgi'):
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

    @staticmethod
    def first_layer_mfgi_init(m):
        periods = [1, 30]  # Number of periods of sine the values of each section of the output vector should hit
        # periods = [1, 60] # Number of periods of sine the values of each section of the output vector should hit
        portion_per_period = np.array([0.25, 0.75])  # Portion of values per section/period
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                num_output = m.weight.size(0)
                num_per_period = (portion_per_period * num_output).astype(int)  # Number of values per section/period
                assert len(periods) == len(num_per_period)
                assert sum(num_per_period) == num_output
                weights = []
                for i in range(0, len(periods)):
                    period = periods[i]
                    num = num_per_period[i]
                    scale = 30 / period
                    weights.append(torch.zeros(num, num_input).uniform_(-np.sqrt(3 / num_input) / scale,
                                                                        np.sqrt(3 / num_input) / scale))
                W0_new = torch.cat(weights, axis=0)
                m.weight.data = W0_new

    def forward(self, input):
        out1 = self.linear(input)
        out = torch.sin(self.omega_0 * out1)
        return out


class FourierEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, sigma=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        assert isinstance(sigma, (type(None), float))
        self.sigma = 1.
        if sigma is not None:
            assert sigma > 0.
            self.sigma = sigma

        # we wrap this stuff as Parameters to allow Pytorch Lightning to recognize them
        # in order to be able to determine the right device
        B = self.sigma * torch.randn(output_dim, input_dim)
        self.B = nn.Parameter(B, requires_grad=False)
        a = torch.tensor(1., dtype=torch.float32).view(1, -1)
        self.a = nn.Parameter(a, requires_grad=False)

    def forward(self, x):
        Bx = torch.einsum('BNd,kd->BNk', x, self.B)
        cosines = self.a * torch.cos(2 * torch.pi * Bx)
        sines = self.a * torch.sin(2 * torch.pi * Bx)
        out = torch.cat([cosines, sines], dim=2)
        assert out.shape == (x.shape[0], x.shape[1], 2 * self.output_dim)
        return out

encoder_cfg = {
    '_target_': 'src.models.coordnet.CoordNetSmall',
    'in_features': 2,
    'emb_features': 2048,
    'out_features': 1,
    'hidden_features': 512,
    'embedding_type': 'relu',
    'embedding_params': None,
    'first_hidden_type': 'relu',
    'first_hidden_params': None,
    'second_hidden_type': 'relu',
    'second_hidden_params': None
}

class CoordNetSmall(nn.Module):
    def __init__(self,
                 in_features, out_features, emb_features, hidden_features,
                 embedding_type='relu', embedding_params=None,
                 first_hidden_type='relu', first_hidden_params=None,
                 second_hidden_type='relu', second_hidden_params=None):
        super().__init__()

        self.hparams = {
            'in_features': in_features,
            'out_features': out_features,
            'emb_features': emb_features,
            'hidden_features': hidden_features,

            'embedding_type': embedding_type,
            'embedding_params': embedding_params,  # defaults: 'omega_0': 30.0 for SineLayer, 'sigma': 1.0 for FourierEmbedding

            'first_hidden_type': first_hidden_type,
            'first_hidden_params': first_hidden_params,  # defaults: 'omega_0': 30.0 for SineLayer
            'second_hidden_type': second_hidden_type,
            'second_hidden_params': second_hidden_params  # defaults: 'omega_0': 30.0 for SineLayer
        }

        if embedding_type == 'relu':
            self.emb = ReLULayer(in_features, emb_features)
        elif embedding_type == 'gelu':
            self.emb = GeLULayer(in_features, emb_features)
        elif embedding_type == 'sine':
            init = embedding_params.get('init', None)
            self.emb = SineLayer(in_features, emb_features,
                                 is_first=True, omega_0=embedding_params['omega_0'], init=init)
        elif embedding_type == 'fourier':
            self.emb = FourierEmbedding(in_features, emb_features,
                                        sigma=embedding_params['sigma'])
            emb_features = 2 * emb_features
        else:
            raise ValueError('Invalid embedding type')

        head_layers = []
        if first_hidden_type == 'relu':
            head_layers.append(ReLULayer(emb_features, hidden_features))
        elif first_hidden_type == 'gelu':
            head_layers.append(GeLULayer(emb_features, hidden_features))
        elif first_hidden_type == 'sine':
            init = first_hidden_params.get('init', None)
            head_layers.append(SineLayer(emb_features, hidden_features,
                                         is_first=False, omega_0=first_hidden_params['omega_0'], init=init))
        else:
            raise ValueError('Invalid first hidden layer type')

        if second_hidden_type == 'relu':
            head_layers.append(ReLULayer(hidden_features, hidden_features))
        elif second_hidden_type == 'gelu':
            head_layers.append(GeLULayer(hidden_features, hidden_features))
        elif second_hidden_type == 'sine':
            head_layers.append(SineLayer(hidden_features, hidden_features,
                                         is_first=False, omega_0=second_hidden_params['omega_0']))
        else:
            raise ValueError('Invalid second hidden layer type')

        head_layers.append(nn.Linear(hidden_features, out_features))
        self.head = nn.Sequential(*head_layers)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        emb = self.emb(coords)
        output = self.head(emb)
        return output, coords