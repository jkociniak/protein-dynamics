import torch
import torch.nn as nn

# class EuclideanPointEncoder(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#     def forward_2d_batch(self, x):
#         nb, nl, _ = x.shape
#         x = x.flatten(start_dim=0, end_dim=1)
#         x = self.forward(x)
#         x = x.unflatten(dim=0, sizes=(nb, nl))
#         return x
#
#     def aux_fun_grad(self, x, twodim_batch=False):
#         """
#         Gradient of the function from remark 1.1 from overleaf
#         :param x: N x d tensor or N x M x d tensor
#         :param twodim_batch: if True, x must be a N x M x d tensor
#         :return: N x d tensor or N x M x d tensor
#         """
#
#         # to use vmap we need a function that does not expect batch dimension
#         single_example_encoder = EuclNoBatchWrapper(self)
#
#         def aux_fun(params, x):
#             encoding = functional_call(single_example_encoder, params, x)  # dimensions: (enc_dim)
#             out = 0.5 * torch.linalg.vector_norm(encoding) ** 2  # dimensions (1)
#             return out
#
#         aux_fun_grad = grad(aux_fun, argnums=1)
#         batch_aux_fun_grad = vmap(aux_fun_grad, in_dims=(None, 0))
#         model_params = dict(single_example_encoder.named_parameters())
#
#         if twodim_batch:
#             assert len(x.shape) == 3
#             assert x.shape[2] == self.input_dim
#             batch_2d_aux_fun_grad = vmap(batch_aux_fun_grad, in_dims=(None, 0))
#             aux_fun_grad_val = batch_2d_aux_fun_grad(model_params, x)  # dimensions: (N, M, d)
#         else:
#             assert len(x.shape) == 2
#             assert x.shape[1] == self.input_dim
#             aux_fun_grad_val = batch_aux_fun_grad(model_params, x)  # dimensions: (N, d)
#         return aux_fun_grad_val
#
#     def hessian_fro_norm(self, x, smoothness_loss=False):
#         # we asssume x of dimensions (N, M, d)
#         x_enc = self.forward_2d_batch(x)  # dimensions: (N, M, enc_dim)
#         x_enc_norm = torch.linalg.norm(x_enc, dim=2)  # dimensions: (N, M)
#         aux_fun_grads = self.aux_fun_grad(x, twodim_batch=True)  # dimensions: (N, M, d)
#         scalar_coeff = 16 / (x_enc_norm ** 2 + 1) ** 4  # dimensions: (N, M)
#         grad_norm = torch.linalg.norm(aux_fun_grads, dim=2) ** 4  # dimensions: (N, M)
#         fro_norm = scalar_coeff * grad_norm  # dimensions: (N, M)
#
#         # 1. mean frobenius norm of the hessian
#         mean_fro_norm = torch.mean(scalar_coeff * grad_norm)
#
#         # 2. curve smoothness loss
#         if smoothness_loss:
#             norm_ends = fro_norm[:, 1:]
#             norm_starts = fro_norm[:, :-1]
#             losses = (norm_ends - norm_starts) ** 2  # dimensions: (N, M - 1)
#             mean_smoothness_error = torch.mean(losses)
#
#             return mean_fro_norm, x_enc_norm, mean_smoothness_error
#         else:
#             return mean_fro_norm, x_enc_norm
#
#     def enc_grad(self, x, twodim_batch=False):
#         # to use vmap we need a function that does not expect batch dimension
#         single_example_encoder = EuclNoBatchWrapper(self)
#
#         def aux_fun(params, x):
#             return functional_call(single_example_encoder, params, x)  # dimensions: (enc_dim)
#
#         aux_fun_grad = jacfwd(aux_fun, argnums=1)
#         batch_aux_fun_grad = vmap(aux_fun_grad, in_dims=(None, 0))
#         model_params = dict(single_example_encoder.named_parameters())
#
#         if twodim_batch:
#             assert len(x.shape) == 3
#             assert x.shape[2] == self.input_dim
#             batch_2d_aux_fun_grad = vmap(batch_aux_fun_grad, in_dims=(None, 0))
#             aux_fun_grad_val = batch_2d_aux_fun_grad(model_params, x)  # dimensions: (N, M, enc_dim, d)
#         else:
#             assert len(x.shape) == 2
#             assert x.shape[1] == self.input_dim
#             aux_fun_grad_val = batch_aux_fun_grad(model_params, x)  # dimensions: (N, enc_dim, d)
#         return aux_fun_grad_val


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


def init_weights_normal(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, num_hidden_layers,
                 outermost_linear=False):
        super().__init__()

        nl = nn.ReLU(inplace=True)

        self.layers = nn.ModuleList()
        self.layers.extend((nn.Linear(in_features, hidden_features), nl))

        for i in range(num_hidden_layers):
            self.layers.extend((nn.Linear(hidden_features, hidden_features), nl))

        if outermost_linear:
            self.layers.append(nn.Linear(hidden_features, out_features))
        else:
            self.layers.extend((nn.Linear(hidden_features, out_features), nl))

        for i in range(1, len(self.layers)):
            self.layers[i].apply(init_weights_normal)

    def forward(self, coords):
        x = self.layers[0](coords)
        for layer in self.layers[1:]:
            x = layer(x)

        return x


class FFN(nn.Module):
    def __init__(self, in_features, out_features, fourier_features, sigma, hidden_features, num_hidden_layers,
                 outermost_linear=False):
        super().__init__()
        self.embedding = FourierEmbedding(in_features, fourier_features, sigma)
        mlp_params = {'in_features': fourier_features * 2,
                      'out_features': out_features,
                      'hidden_features': hidden_features,
                      'num_hidden_layers': num_hidden_layers,
                      'outermost_linear': outermost_linear}
        self.mlp = MLP(**mlp_params)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.mlp(self.embedding(coords))
        return output, coords
