import torch
import torch.nn as nn
from torch_geometric.nn.pool import fps, radius
from torch_geometric.nn import MLP, PointTransformerConv
from torch_geometric.utils import to_dense_batch


class PointTransformerBlock(nn.Module):
    """Point Transformer block with Set Abstraction (FPS + radius grouping + PointTransformerConv)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        ratio: float,
        radius_val: float,
        nsample: int,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ratio = ratio
        self.r = radius_val
        self.k = nsample

        self.in_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        pos_nn = MLP([3, out_dim, out_dim], plain_last=False, norm=None)
        attn_nn = MLP([out_dim, out_dim], plain_last=False, norm=None)

        self.conv = PointTransformerConv(
            in_channels=out_dim, out_channels=out_dim, pos_nn=pos_nn, attn_nn=attn_nn
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (N, C_in) features
        pos : (N, 3) coordinates
        batch : (N,) batch indices

        Returns
        -------
        x_out : (M, C_out) features of downsampled points
        pos_out : (M, 3) coordinates of downsampled points
        batch_out : (M,) batch indices of downsampled points
        """
        x = self.in_proj(x)

        # Furthest Point Sampling
        if self.ratio < 1.0:
            idx = fps(pos, batch, ratio=self.ratio)
        else:
            idx = torch.arange(pos.shape[0], device=pos.device)

        # Radius grouping
        row, col = radius(
            x=pos, y=pos[idx], r=self.r,
            batch_x=batch, batch_y=batch[idx],
            max_num_neighbors=self.k,
        )
        edge_index = torch.stack([col, row], dim=0)

        # Bipartite convolution
        x_out = self.conv(x=(x, x[idx]), pos=(pos, pos[idx]), edge_index=edge_index)
        x_out = self.norm(x_out)

        return x_out, pos[idx], batch[idx]


class GlobalSceneEncoder(nn.Module):
    """Encode a (padded) point cloud into a single global feature via cascading PointTransformer blocks."""

    def __init__(
        self,
        in_dim: int = 384,
        out_dim: int = 384,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # (in_dim, out_dim, ratio, radius, nsample)
        sa_configs = [
            (in_dim, out_dim, 0.25, 1.0, 32),
            (out_dim, out_dim, 0.25, 2.0, 32),
            (out_dim, out_dim, 0.25, 4.0, 32),
            (out_dim, out_dim, 0.25, 8.0, 32),
        ]
        for i, (id_, od_, ratio, r, k) in enumerate(sa_configs, 1):
            setattr(self, f"sa{i}", PointTransformerBlock(id_, od_, ratio, r, k, heads, dropout))

        self.proj = nn.Sequential(nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim))

        # Attention pooling: learnable query (PMA-style)
        self.pool_query = nn.Parameter(torch.randn(1, 1, out_dim))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=heads, dropout=dropout, batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(out_dim)

    def forward(self, pts: torch.Tensor, pad: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        pts : (B, N, 3 + in_dim) concatenated coordinates and features
        pad : (B, N) bool mask where True = PAD token

        Returns
        -------
        global_feat : (B, out_dim) global scene feature
        """
        B, N, _ = pts.shape
        xyz = pts[..., :3].contiguous()
        feat = pts[..., 3:].contiguous()

        # Convert to PyG flat-batch format
        if pad is not None:
            mask = ~pad
            batch_vec_full = torch.arange(B, device=pts.device).unsqueeze(1).expand(B, N)
            pos = xyz[mask]
            x = feat[mask]
            batch = batch_vec_full[mask]
        else:
            pos = xyz.view(-1, 3)
            x = feat.view(-1, feat.shape[-1])
            batch = torch.arange(B, device=pts.device).repeat_interleave(N)

        x, pos, batch = self.sa1(x, pos, batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x, pos, batch = self.sa3(x, pos, batch)
        x, pos, batch = self.sa4(x, pos, batch)

        x = self.proj(x)

        # Attention pooling: convert flat batch back to dense (B, M, D) + key_padding_mask
        x_dense, attn_mask = to_dense_batch(x, batch)  # (B, M_max, D), (B, M_max)

        query = self.pool_query.expand(B, -1, -1)  # (B, 1, D)
        key_pad_mask = ~attn_mask  # True = ignore
        pooled, _ = self.pool_attn(
            query, x_dense, x_dense, key_padding_mask=key_pad_mask,
        )  # (B, 1, D)
        global_feat = self.pool_norm(pooled.squeeze(1))  # (B, D)
        return global_feat
