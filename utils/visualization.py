"""PCA visualization utilities for VoxelHashTable-based latent maps.

Adapted from 4d_latent_mapping/utils/visualization.py for the SBP pipeline.
Key difference: queries VoxelHashTable with aggregated coords instead of
reading stored features from NeuralPointMap.
"""

import numpy as np
import torch


class TorchPCA:
    """
    GPU-accelerated PCA following scikit-learn's implementation.

    Uses full SVD via torch.linalg.svd (equivalent to sklearn's svd_solver='full').
    """

    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = whiten
        self.mean_ = None
        self.components_ = None  # (n_components, n_features) like sklearn
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_samples_ = None
        self.n_features_ = None

    def fit(self, X):
        """
        Fit PCA on X.

        Args:
            X: (n_samples, n_features) tensor
        """
        self.n_samples_, self.n_features_ = X.shape

        # Center the data
        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_

        # Full SVD
        # X = U @ S @ Vt
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

        # Determine n_components
        max_components = min(self.n_samples_, self.n_features_)
        if self.n_components is None:
            n_components = max_components
        else:
            n_components = min(self.n_components, max_components)

        # Components are first n_components rows of Vt
        self.components_ = Vt[:n_components]  # (n_components, n_features)

        # Explained variance (sklearn uses n_samples - 1 for unbiased estimate)
        self.explained_variance_ = (S[:n_components] ** 2) / (self.n_samples_ - 1)

        total_var = (S ** 2).sum() / (self.n_samples_ - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        # Store singular values for whitening
        self._S = S[:n_components]

        return self

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        Args:
            X: (n_samples, n_features) tensor

        Returns:
            (n_samples, n_components) numpy array
        """
        X_centered = X - self.mean_

        # Project: X_centered @ components.T
        X_transformed = X_centered @ self.components_.T

        if self.whiten:
            X_transformed = X_transformed / torch.sqrt(self.explained_variance_)

        # Pad with zeros if fewer components than requested (for RGB visualization)
        actual_components = X_transformed.shape[1]
        if self.n_components is not None and actual_components < self.n_components:
            padding = torch.zeros(X.shape[0], self.n_components - actual_components, device=X.device)
            X_transformed = torch.cat([X_transformed, padding], dim=1)

        return X_transformed.detach().cpu().numpy()

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


# --------------------------------------------------------------------------- #
#  PCA Visualization for VoxelHashTable
# --------------------------------------------------------------------------- #

def run_pca_visualization(
    server,
    coords: torch.Tensor,
    grid,
    decoder: torch.nn.Module,
    env_name: str,
    epoch_label: int,
    config: dict,
) -> None:
    """
    Visualize decoded VoxelHashTable features using PCA coloring in viser.

    Queries the grid with aggregated world coordinates, decodes, then runs
    PCA on the decoded features to produce an RGB-colored point cloud.

    Args:
        server: viser.ViserServer instance
        coords: (N, 3) world coordinates aggregated over the epoch
        grid: VoxelHashTable instance (trained)
        decoder: MLP decoder (latent -> vision features)
        env_name: Environment name for labeling
        epoch_label: Current epoch number
        config: Full config dict with 'visualization', 'scene_min', 'scene_max'
    """
    vis_config = config.get('visualization', {})
    z_threshold = vis_config.get('z_threshold', 2.5)
    max_points = vis_config.get('max_points', 500000)
    point_size = vis_config.get('point_size', 0.005)
    n_components = 3
    whiten = vis_config.get('whiten', True)
    quantile_low = vis_config.get('quantile_low', 0.02)
    quantile_high = vis_config.get('quantile_high', 0.98)

    x_center = (config['scene_min'][0] + config['scene_max'][0]) / 2.0
    y_center = (config['scene_min'][1] + config['scene_max'][1]) / 2.0

    print(f"\n[VIS] Running PCA visualization for {env_name} (epoch: {epoch_label})...")

    num_points = coords.shape[0]
    if num_points == 0:
        print("[VIS] No coordinates to visualize; skipping.")
        return

    # Downsample if too many points
    if num_points > max_points:
        indices = torch.randperm(num_points, device=coords.device)[:max_points]
        coords = coords[indices]

    # Query grid and decode
    with torch.no_grad():
        voxel_feat = grid.query_voxel_feature(coords)
        decoded_feat = decoder(voxel_feat)

    # PCA on decoded features
    pca = TorchPCA(n_components=n_components, whiten=whiten)
    colors = pca.fit_transform(decoded_feat)

    # Robust quantile normalization
    q_low = np.quantile(colors, quantile_low, axis=0)
    q_high = np.quantile(colors, quantile_high, axis=0)
    colors = np.clip(colors, q_low, q_high)
    colors = (colors - q_low) / (q_high - q_low + 1e-8)

    coords_np = coords.cpu().numpy()

    # Z-threshold filter
    mask = coords_np[:, 2] <= z_threshold
    coords_np = coords_np[mask]
    colors = colors[mask]

    if len(coords_np) == 0:
        print("[VIS] No points after Z filtering; skipping.")
        return

    # Center by scene midpoint
    coords_np[:, 0] -= x_center
    coords_np[:, 1] -= y_center

    print(f"[VIS] Displaying {len(coords_np)} points (after decoder) for {env_name}")

    server.add_point_cloud(
        name=f"/pca/{env_name}/after_decoder",
        points=coords_np,
        colors=(colors * 255).astype(np.uint8),
        point_size=point_size,
    )
