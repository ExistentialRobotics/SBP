import torch.nn as nn


def init_weights_kaiming(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class MLP(nn.Module):
    def __init__(self, input_dim: int = 768, output_dim: int = 768):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
        )

        self.apply(init_weights_kaiming)

    def forward(self, x):
        return self.mlp(x)
