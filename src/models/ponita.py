import torch
import torch.nn as nn
from ponita.models import Ponita


class MyPonita(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_features, hidden_layers):
        super().__init__()

        # # Store some of the relevant args
        # self.lr = args.lr
        # self.weight_decay = args.weight_decay
        # self.epochs = args.epochs
        # self.warmup = args.warmup
        # if args.layer_scale == 0.:
        #     args.layer_scale = None

        # # For rotation augmentations during training and testing
        # self.train_augm = args.train_augm
        # self.rotation_transform = RandomRotate(['pos'], n=2)

        # Make the model
        self.model = Ponita(in_channels,
                            hidden_features,
                            out_channels,
                            hidden_layers,
                            output_dim_vec=0,
                            radius=None,
                            num_ori=10,
                            basis_dim=256,
                            degree=3,
                            widening_factor=4,
                            layer_scale=None,
                            multiple_readouts=False,
                            task_level='graph',
                            lift_graph=True)

    def forward(self, graph):
        # Only utilize the scalar (energy) prediction
        coords = graph.pos.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        graph.pos = coords
        scalar_out, vec_out = self.model(graph)
        return scalar_out, coords