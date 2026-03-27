from typing import Tuple, Union

import numpy as np
import torch

__all__ = [
    "calculate_intrinsics",
    "unproject_depth_to_world",
    "solve_rigid_transform",
    "refine_registration_icp",
    "voxel_downsample",
]

# --------------------------------------------------------------------------- #
#  Camera intrinsics                                                          #
# --------------------------------------------------------------------------- #

def calculate_intrinsics(
    image_width: int,
    image_height: int,
    horizontal_aperture: float,
    focal_length: float,
) -> np.ndarray:
    """
    Calculates camera intrinsic matrix from camera parameters.

    Args:
        image_width: Width of the output image in pixels
        image_height: Height of the output image in pixels
        horizontal_aperture: Horizontal aperture of the camera sensor
        focal_length: Focal length of the camera lens

    Returns:
        3x3 intrinsic matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    """
    vertical_aperture = horizontal_aperture * (image_height / image_width)

    cx = image_width / 2.0
    cy = image_height / 2.0
    fx = focal_length * image_width / horizontal_aperture
    fy = focal_length * image_height / vertical_aperture

    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# --------------------------------------------------------------------------- #
#  3-D coordinates from depth map                                             #
# --------------------------------------------------------------------------- #

def unproject_depth_to_world(
    depth: torch.Tensor,
    cam_to_world: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    original_height: int = None,
    original_width: int = None,
    original_size: int = None,
):
    """
    Compute 3D world coordinates from depth map.

    Args:
        depth: Depth map tensor (B, H, W) or (B, 1, H, W)
        cam_to_world: Camera-to-world transform matrix (B, 1, 3, 4) or (B, 3, 4)
        fx, fy, cx, cy: Original camera intrinsics (before any resizing)
        original_height, original_width: Original image dimensions that intrinsics correspond to
        original_size: Original image size for square images (alternative to height/width)

    Returns:
        coords_world: World coordinates (B, 3, H, W)
    """
    # Handle original_size for square images
    if original_size is not None:
        original_height = original_size
        original_width = original_size
    elif original_height is None or original_width is None:
        original_height = 480
        original_width = 480

    device = depth.device

    if depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)

    B, H_feat, W_feat = depth.shape

    # Scale intrinsics from original resolution to current depth map resolution
    scale_x = W_feat / float(original_width)
    scale_y = H_feat / float(original_height)
    fx_scaled = fx * scale_x
    fy_scaled = fy * scale_y
    cx_scaled = cx * scale_x
    cy_scaled = cy * scale_y

    u = torch.arange(W_feat, device=device).view(1, -1).expand(H_feat, W_feat) + 0.5
    v = torch.arange(H_feat, device=device).view(-1, 1).expand(H_feat, W_feat) + 0.5
    u = u.unsqueeze(0).expand(B, -1, -1)
    v = v.unsqueeze(0).expand(B, -1, -1)

    x_cam = (u - cx_scaled) * depth / fx_scaled
    y_cam = (v - cy_scaled) * depth / fy_scaled
    z_cam = depth
    ones = torch.ones_like(z_cam)
    coords_hom = torch.stack([x_cam, y_cam, z_cam, ones], dim=1)  # (B, 4, H, W)

    cam_to_world = cam_to_world.squeeze(1)
    ones_row = torch.tensor([0, 0, 0, 1], device=device, dtype=cam_to_world.dtype).view(1, 1, 4)
    ones_row = ones_row.expand(B, 1, 4)
    cam_to_world_4x4 = torch.cat([cam_to_world, ones_row], dim=1)

    coords_hom_flat = coords_hom.view(B, 4, -1)
    world_coords_hom = torch.bmm(cam_to_world_4x4, coords_hom_flat)
    coords_world = world_coords_hom[:, :3, :].view(B, 3, H_feat, W_feat)
    return coords_world


# --------------------------------------------------------------------------- #
#  Voxel downsampling                                                         #
# --------------------------------------------------------------------------- #

def voxel_downsample(coords: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """
    Downsample 3D points by keeping one point per voxel.

    Args:
        coords: (N, 3) point coordinates
        voxel_size: Size of each voxel cell

    Returns:
        (M, 3) downsampled coordinates (M <= N)
    """
    voxel_indices = torch.div(coords, voxel_size, rounding_mode='floor').long()
    # Hash voxel indices to scalar keys
    keys = voxel_indices[:, 0] * 73856093 + voxel_indices[:, 1] * 19349669 + voxel_indices[:, 2] * 83492791
    _, unique_idx = torch.unique(keys, return_inverse=True)
    # Pick one point per voxel via scatter (last occurrence per voxel)
    num_voxels = unique_idx.max().item() + 1
    selected = torch.zeros(num_voxels, dtype=torch.long, device=coords.device)
    selected.scatter_(0, unique_idx, torch.arange(coords.size(0), device=coords.device))
    return coords[selected]
