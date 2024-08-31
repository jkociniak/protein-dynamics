import numpy as np
import torch
import torch.nn as nn
from torch.func import vmap, grad, functional_call, jacfwd


class EuclNoBatchWrapper(nn.Module):
    def __init__(self, base_module):
        super().__init__()
        self.base_module = base_module

    def forward(self, x):
        # x dimensions: (input_dim) = (D)
        x = x.unsqueeze(0)
        x = self.base_module(x)
        x = x.squeeze(0)
        return x


class EuclideanPointEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward_2d_batch(self, x):
        nb, nl, _ = x.shape
        x = x.flatten(start_dim=0, end_dim=1)
        x = self.forward(x)
        x = x.unflatten(dim=0, sizes=(nb, nl))
        return x

    def aux_fun_grad(self, x, twodim_batch=False):
        """
        Gradient of the function from remark 1.1 from overleaf
        :param x: N x d tensor or N x M x d tensor
        :param twodim_batch: if True, x must be a N x M x d tensor
        :return: N x d tensor or N x M x d tensor
        """

        # to use vmap we need a function that does not expect batch dimension
        single_example_encoder = EuclNoBatchWrapper(self)

        def aux_fun(params, x):
            encoding = functional_call(single_example_encoder, params, x)  # dimensions: (enc_dim)
            out = 0.5 * torch.linalg.vector_norm(encoding) ** 2  # dimensions (1)
            return out

        aux_fun_grad = grad(aux_fun, argnums=1)
        batch_aux_fun_grad = vmap(aux_fun_grad, in_dims=(None, 0))
        model_params = dict(single_example_encoder.named_parameters())

        if twodim_batch:
            assert len(x.shape) == 3
            assert x.shape[2] == self.input_dim
            batch_2d_aux_fun_grad = vmap(batch_aux_fun_grad, in_dims=(None, 0))
            aux_fun_grad_val = batch_2d_aux_fun_grad(model_params, x)  # dimensions: (N, M, d)
        else:
            assert len(x.shape) == 2
            assert x.shape[1] == self.input_dim
            aux_fun_grad_val = batch_aux_fun_grad(model_params, x)  # dimensions: (N, d)
        return aux_fun_grad_val

    def hessian_fro_norm(self, x):
        # we asssume x of dimensions (N, M, d)
        x_enc = self.forward_2d_batch(x)  # dimensions: (N, M, enc_dim)
        x_enc_norm = torch.linalg.norm(x_enc, dim=2)  # dimensions: (N, M)
        aux_fun_grads = self.aux_fun_grad(x, twodim_batch=True)  # dimensions: (N, M, d)
        scalar_coeff = 16 / (x_enc_norm ** 2 + 1) ** 4  # dimensions: (N, M)
        grad_norm = torch.linalg.norm(aux_fun_grads, dim=2) ** 4  # dimensions: (N, M)
        fro_norm = scalar_coeff * grad_norm  # dimensions: (N, M)

        # 1. mean frobenius norm of the hessian
        mean_fro_norm = torch.mean(scalar_coeff * grad_norm)
        return mean_fro_norm, x_enc_norm

    def enc_grad(self, x, twodim_batch=False):
        # to use vmap we need a function that does not expect batch dimension
        single_example_encoder = EuclNoBatchWrapper(self)

        def aux_fun(params, x):
            return functional_call(single_example_encoder, params, x)  # dimensions: (enc_dim)

        aux_fun_grad = jacfwd(aux_fun, argnums=1)
        batch_aux_fun_grad = vmap(aux_fun_grad, in_dims=(None, 0))
        model_params = dict(single_example_encoder.named_parameters())

        if twodim_batch:
            assert len(x.shape) == 3
            assert x.shape[2] == self.input_dim
            batch_2d_aux_fun_grad = vmap(batch_aux_fun_grad, in_dims=(None, 0))
            aux_fun_grad_val = batch_2d_aux_fun_grad(model_params, x)  # dimensions: (N, M, enc_dim, d)
        else:
            assert len(x.shape) == 2
            assert x.shape[1] == self.input_dim
            aux_fun_grad_val = batch_aux_fun_grad(model_params, x)  # dimensions: (N, enc_dim, d)
        return aux_fun_grad_val


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 2, np.sqrt(6 / num_input) / 2)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def init_weights_normal(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


class FourierEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, append=False, sigma=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.append = append

        assert isinstance(sigma, (type(None), float))
        self.sigma = 1.
        if sigma is not None:
            assert sigma > 0.
            self.sigma = sigma

        # we wrap this stuff as Parameters to allow Pytorch Lightning to recognize them
        # in order to be able to determine the right device
        B = self.sigma * torch.randn(output_dim, input_dim)
        self.B = nn.Parameter(B, requires_grad=False)
        a = torch.tensor(1., dtype=torch.float32).view(1, -1) #/ torch.arange(1, output_dim + 1, dtype=torch.float32).view(1, -1)
        self.a = nn.Parameter(a, requires_grad=False)

    def forward(self, x):
        Bx = torch.einsum('Nd,kd->Nk', x, self.B)
        cosines = self.a * torch.cos(2 * torch.pi * Bx)
        sines = self.a * torch.sin(2 * torch.pi * Bx)
        if self.append:
            out = torch.cat([x, cosines, sines], dim=1)
        else:
            out = torch.cat([cosines, sines], dim=1)
        return out


class MLP(EuclideanPointEncoder):
    def __init__(self, in_features, out_features, hidden_features, num_hidden_layers,
                 outermost_linear=False, nonlinearity='relu', fourier_features=None,
                 fourier_append=False):
        super().__init__(input_dim=in_features, output_dim=out_features)

        self.embedding = lambda x: x
        if fourier_features is not None:
            assert isinstance(fourier_features, int) and fourier_features > 0
            self.embedding = FourierEmbedding(in_features, fourier_features, append=fourier_append)
            in_features = 2 * fourier_features if not fourier_append else in_features + 2 * fourier_features

        #Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        #special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        self.layers = nn.ModuleList()
        self.layers.extend((nn.Linear(in_features, hidden_features), nl))

        for i in range(num_hidden_layers):
            self.layers.extend((nn.Linear(hidden_features, hidden_features), nl))

        if outermost_linear:
            self.layers.append(nn.Linear(hidden_features, out_features))
        else:
            self.layers.extend((nn.Linear(hidden_features, out_features), nl))

        if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.layers[0].apply(first_layer_init)

        for i in range(1, len(self.layers)):
            self.layers[i].apply(nl_weight_init)

    def forward(self, coords):
        x = self.embedding(coords)
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = layer(x) + x

        x = self.layers[-1](x)
        return x


class MLP2(EuclideanPointEncoder):
    def __init__(self, in_features, out_features, hidden_features, num_hidden_layers,
                 outermost_linear=False, nonlinearity='relu', skip_connection=False):
        super().__init__(input_dim=in_features, output_dim=out_features)
        assert isinstance(skip_connection, bool)
        self.skip_connection = skip_connection

        #Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        #special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        self.layers = nn.ModuleList()
        self.layers.extend((nn.Linear(in_features, hidden_features), nl))

        for i in range(num_hidden_layers):
            self.layers.extend((nn.Linear(hidden_features, hidden_features), nl))

        if outermost_linear:
            self.layers.append(nn.Linear(hidden_features, out_features))
        else:
            self.layers.extend((nn.Linear(hidden_features, out_features), nl))

        if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.layers[0].apply(first_layer_init)

        for i in range(1, len(self.layers)):
            self.layers[i].apply(nl_weight_init)

    def forward(self, coords):
        x = self.layers[0](coords)
        for layer in self.layers[1:-1]:
            if self.skip_connection:
                x = layer(x) + x
            else:
                x = layer(x)

        x = self.layers[-1](x)
        return x


class FourierMLP(EuclideanPointEncoder):
    def __init__(self, embedding, mlp):
        super().__init__(embedding.input_dim, mlp.output_dim)
        self.embedding = embedding
        self.mlp = mlp

    def forward(self, x):
        return self.mlp(self.embedding(x))


class EnsembleEuclideanMLPEncoder(EuclideanPointEncoder):
    def __init__(self, input_dim, hidden_dim, output_dim, n_models=10, num_hidden_layers=5, outermost_linear=False, nonlinearity='relu'):
        super().__init__(input_dim, output_dim)
        self.n_models = n_models
        self.models = nn.ModuleList([MLP(in_features=input_dim, hidden_features=hidden_dim,
                                         out_features=output_dim, num_hidden_layers=num_hidden_layers,
                                         outermost_linear=outermost_linear, nonlinearity=nonlinearity) for _ in range(n_models)])

    def forward(self, x):
        preds = torch.stack([model(x) for model in self.models], dim=0)
        assert preds.shape[0] == self.n_models
        return preds.mean(dim=0)


class DoubleMLP(EuclideanPointEncoder):
    def __init__(self, base_mlp_params, fourier_mlp_params, weights=(0.5, 0.5)):
        assert 'in_features' in base_mlp_params and 'out_features' in base_mlp_params

        input_dim = base_mlp_params['in_features']
        output_dim = base_mlp_params['out_features']
        super().__init__(input_dim, output_dim)

        assert 'fourier_features' in base_mlp_params and base_mlp_params['fourier_features'] is None
        self.base_mlp = MLP(**base_mlp_params)

        assert 'fourier_features' in fourier_mlp_params and fourier_mlp_params['fourier_features'] is not None
        self.fourier_mlp = MLP(**fourier_mlp_params)

        assert isinstance(weights, tuple) and len(weights) == 2
        assert isinstance(weights[0], float) and 0. <= weights[0]
        assert isinstance(weights[1], float) and 0. <= weights[1]
        self.weights = weights

    def forward(self, x):
        out = self.base_mlp(x) * self.weights[0] + self.fourier_mlp(x) * self.weights[1]
        return out


class DoubleMLP2(EuclideanPointEncoder):
    def __init__(self, base_mlp_params, fourier_mlp_params, weights=(0.5, 0.5)):
        assert 'in_features' in base_mlp_params and 'out_features' in base_mlp_params

        input_dim = base_mlp_params['in_features']
        output_dim = base_mlp_params['out_features']
        super().__init__(input_dim, output_dim)

        base_mlp_params['out_features'] = base_mlp_params['hidden_features']
        assert 'fourier_features' in base_mlp_params and base_mlp_params['fourier_features'] is None
        self.base_mlp = MLP(**base_mlp_params)

        fourier_mlp_params['out_features'] = fourier_mlp_params['hidden_features']
        assert 'fourier_features' in fourier_mlp_params and fourier_mlp_params['fourier_features'] is not None
        self.fourier_mlp = MLP(**fourier_mlp_params)

        hidden_features = base_mlp_params['hidden_features']
        self.common_layers = nn.Sequential(nn.Linear(2 * hidden_features, hidden_features), nn.ReLU(inplace=True),
                                           nn.Linear(hidden_features, hidden_features), nn.ReLU(inplace=True),
                                           nn.Linear(hidden_features, output_dim))

        self.base_proj = nn.Linear(hidden_features, output_dim)

        assert isinstance(weights, tuple) and len(weights) == 2
        assert isinstance(weights[0], float) and 0. <= weights[0]
        assert isinstance(weights[1], float) and 0. <= weights[1]
        self.weights = weights

    def forward(self, x):
        base_repr = self.base_mlp(x)
        base_out = self.base_proj(base_repr)
        fourier_repr = self.fourier_mlp(x)
        common_repr = torch.cat([base_repr, fourier_repr], dim=1)
        common_out = self.common_layers(common_repr)
        out = base_out * self.weights[0] + common_out * self.weights[1]
        return out
