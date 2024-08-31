import torch
import torch.nn as nn


class PointCloudEncoder(nn.Module):
    def __init__(self, input_dim, n_atoms, output_dim):
        super().__init__()
        self.n_atoms = n_atoms
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward_2d_batch(self, x):
        nb, nl, _, _ = x.shape
        x = x.flatten(start_dim=0, end_dim=1)
        x = self.forward(x)
        x = x.unflatten(dim=0, sizes=(nb, nl))
        return x


class PointCloudConvEncoder(PointCloudEncoder):
    def __init__(self, n_atoms, input_dim, output_dim, kernel_size=3):
        super().__init__(n_atoms, input_dim, output_dim)
        self.kernel_size = kernel_size

        # we will apply a convolution over atom chain with input_dim as channels
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=4, kernel_size=kernel_size)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=kernel_size)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=kernel_size)
        # self.pool3 = nn.MaxPool1d(kernel_size=2)

        # flatten and cast to output_dim via linear layer
        self.linear = nn.Linear(52 * 8, output_dim)

    def forward(self, x):
        # x dimensions: (batch, n_atoms, input_dim) = (B, N, D)
        x = x.permute(0, 2, 1)  # dimensions: (B, D, N)

        x = self.conv1(x)
        x = self.pool1(x)
        x = torch.relu(x)
        # x dimensions: (B, 4, N/2)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.relu(x)
        # x dimensions: (B, 8, N/4)
        # x = self.conv3(x)
        # x = self.pool3(x)
        # x = torch.relu(x)
        # x dimensions: (B*M, 16, N/8)

        x = x.view(x.size(0), -1)  # dimensions: (B*M, D*N/8)
        #print(x.shape)
        x = self.linear(x)  # dimensions (B*M, output_dim)
        return x

    def forward_2d_batch(self, x):
        nb, nl, _, _ = x.shape
        x = x.flatten(start_dim=0, end_dim=1)
        x = self.forward(x)
        x = x.unflatten(dim=0, sizes=(nb, nl))
        return x


class PCSingleExampleWrapper(nn.Module):
    def __init__(self, base_module):
        super().__init__()
        self.base_module = base_module

    def forward(self, x):
        # x dimensions: (n_atoms, input_dim) = (N, D)
        x = x.unsqueeze(0)
        x = self.base_module(x)
        x = x.squeeze(0)
        return x


class PointCloudMLPEncoder(PointCloudEncoder):
    def __init__(self, n_atoms, input_dim, hidden_dim, output_dim):
        super().__init__(n_atoms, input_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(n_atoms * input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x dimensions: (batch, n_atoms, input_dim) = (B, N, D)
        x = x.flatten(start_dim=1, end_dim=2)  # dimensions: (B, N*D)
        x = self.linear(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = torch.relu(x)
        x = self.linear4(x)
        #x = torch.tanh(x)
        return x