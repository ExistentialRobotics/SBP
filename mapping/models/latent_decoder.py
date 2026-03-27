import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LatentDecoder"]


def _init_weights_kaiming(m: nn.Module):
    """Kaiming initialization for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class LatentDecoder(nn.Module):
    """MLP decoder for voxel grid latent features.

    Architecture: fc1→ln1→relu → fc2→ln2→relu → fc3→ln3→relu → fc4→ln4→relu → fc5.

    State-dict keys (fc1…fc5, ln1…ln4) are compatible with
    ``manipulation.module.mlp.ImplicitDecoder`` (pe_type="none").
    """

    def __init__(
        self,
        voxel_feature_dim: int = 128,
        hidden_dim: int = 240,
        output_dim: int = 768,
    ):
        super().__init__()

        self.fc1 = nn.Linear(voxel_feature_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)

        self.fc5 = nn.Linear(hidden_dim, output_dim)

        self.apply(_init_weights_kaiming)

    def forward(self, voxel_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel_features: (..., voxel_feature_dim) input features.

        Returns:
            (..., output_dim) decoded features.
        """
        x = F.relu(self.ln1(self.fc1(voxel_features)), inplace=True)
        x = F.relu(self.ln2(self.fc2(x)), inplace=True)
        x = F.relu(self.ln3(self.fc3(x)), inplace=True)
        x = F.relu(self.ln4(self.fc4(x)), inplace=True)
        return self.fc5(x)
