import torch
from torch import nn

from positional_embeddings import PositionalEmbedding


class ResidualMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int = 128,
        hidden_layers: int = 3,
    ):
        super().__init__()

        self.act = nn.GELU()

        self.in_layer = nn.Linear(in_features, hidden_size)
        self.out_layer = nn.Linear(hidden_size, out_features)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )

    def forward(self, x):
        x = self.act(self.in_layer(x))

        # hidden layers with skip connection
        for layer in self.hidden_layers:
            x = x + self.act(layer(x))

        return self.out_layer(x)


class LowDimensionalDiffusionModel(nn.Module):
    """
    Model architecture for training diffusion models on low-dimensional data (e.g 1D/2D)
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
    ):
        super().__init__()

        # positional embeddings
        self.time_emb = PositionalEmbedding(emb_size, time_emb)
        self.coord_emb = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        # size of concatenated embeddings
        concat_size = len(self.time_emb.layer) + len(self.coord_emb.layer) * in_features

        self.joint_mlp = ResidualMLP(concat_size, 2, hidden_size, hidden_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # positional encoding of x
        d = x.shape[1]
        coord_emb = [self.coord_emb(x[:, i]) for i in range(d)]
        coord_emb = torch.cat(coord_emb, dim=1)

        # positional encoding of time
        t_emb = self.time_emb(t).to(x.device)

        # join into single feature
        x = torch.cat((coord_emb, t_emb), dim=-1)

        x = self.joint_mlp(x)
        return x
