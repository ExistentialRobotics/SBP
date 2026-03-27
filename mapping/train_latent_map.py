"""
Train latent feature decoder on VoxelHashTable representation using HDF5 dataset.

This module provides the core training functionality that can be:
1. Run standalone via main()
2. Called programmatically via train()

Key difference from 4d_latent_mapping/train_neural_points.py:
- Uses VoxelHashTable (multi-resolution hash grid) instead of NeuralPointMap
- No encoder, no contrastive losses (inter/intra), no gradient isolation
- Single optimizer for all grids + decoder
"""
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import h5py

# Add project root to sys.path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.geometry import unproject_depth_to_world
from mapping.representations.voxel_hash_table import VoxelHashTable
from mapping.models.latent_decoder import LatentDecoder
from mapping.dataset.dataset import HDF5Dataset, collate_fn, EnvBatchSampler, hdf5_worker_init_fn
from mapping.losses.cosine_similarity import cosine_similarity_loss
from mapping.config.train_config import TrainConfig
from utils.logger import TrainLogger
from utils.visualization import run_pca_visualization
from utils.geometry import voxel_downsample


# --------------------------------------------------------------------------- #
#  Dataset Setup
# --------------------------------------------------------------------------- #

def get_intrinsics_and_feature_dim(
    dataset_path: Path,
    target_envs: List[str],
    feature_key: Optional[str],
    default_feature_dim: int,
) -> Tuple[float, float, float, float, int, int, int]:
    """
    Get camera intrinsics and feature dimension from first HDF5 file.

    Returns:
        (fx, fy, cx, cy, H_orig, W_orig, feature_dim)
    """
    first_hdf5 = None
    if target_envs:
        for env in target_envs:
            p = dataset_path / env
            if p.exists():
                if p.is_file():
                    first_hdf5 = p
                else:
                    first_hdf5 = next(p.glob("*.hdf5"), None)
                if first_hdf5:
                    break
    else:
        first_hdf5 = next(dataset_path.rglob("*.hdf5"), None)

    if not first_hdf5:
        raise FileNotFoundError(f"No HDF5 files found in {dataset_path}")

    with h5py.File(first_hdf5, 'r') as f:
        if "intrinsics" not in f:
            raise ValueError("'intrinsics' not found in HDF5")
        K = f["intrinsics"][:]
        fx, fy, cx, cy = K[0], K[1], K[2], K[3]

        H_orig, W_orig, _ = f['rgb'].shape[1:]

        if feature_key is not None and feature_key in f:
            feature_dim = f[feature_key].shape[1]
        else:
            feature_dim = default_feature_dim

    return fx, fy, cx, cy, H_orig, W_orig, feature_dim


def create_dataset_and_loader(
    dataset_path: Path,
    cfg: TrainConfig,
) -> Tuple[HDF5Dataset, DataLoader]:
    """Create HDF5 dataset and dataloader with EnvBatchSampler."""
    dataset = HDF5Dataset(
        dataset_path,
        cfg.target_envs,
        cfg.num_images,
        cfg.image_size,
        cfg.patch_size,
        feature_key=cfg.feature_key,
    )

    if len(dataset) == 0:
        raise ValueError("No valid training data found")

    batch_sampler = EnvBatchSampler(dataset, batch_size=cfg.batch_size, drop_last=False)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn,
        worker_init_fn=hdf5_worker_init_fn,
    )

    return dataset, dataloader


# --------------------------------------------------------------------------- #
#  Training Step
# --------------------------------------------------------------------------- #

def filter_valid_points(
    coords_flat: torch.Tensor,
    depth_flat: torch.Tensor,
    feats_flat: torch.Tensor,
    cfg: TrainConfig,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Filter points by scene bounds and depth.

    Returns:
        (coords_valid, feats_valid) or (None, None) if no valid points
    """
    in_x = (coords_flat[:, 0] > cfg.scene_min[0]) & (coords_flat[:, 0] < cfg.scene_max[0])
    in_y = (coords_flat[:, 1] > cfg.scene_min[1]) & (coords_flat[:, 1] < cfg.scene_max[1])
    in_z = (coords_flat[:, 2] > cfg.scene_min[2]) & (coords_flat[:, 2] < cfg.scene_max[2])
    valid_mask = in_x & in_y & in_z

    # Depth filter
    valid_mask = valid_mask & (depth_flat >= 0.01)

    if valid_mask.sum() == 0:
        return None, None

    return coords_flat[valid_mask], feats_flat[valid_mask]


def train_step(
    data: dict,
    grids: Dict[str, VoxelHashTable],
    decoder: LatentDecoder,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    intrinsics: Tuple[float, float, float, float, int, int],
    device: str,
) -> Tuple[Optional[float], Optional[float], Optional[torch.Tensor]]:
    """
    Execute a single training step.

    Returns:
        (loss_value, cos_sim_value, coords_valid) or (None, None, None) if skipped
    """
    fx, fy, cx, cy, H_orig, W_orig = intrinsics

    # Get data tensors
    depth_t = data["depth_t"].to(device)
    cam_to_world_t = data["cam_to_world_t"].unsqueeze(1).to(device)
    env_name = data["env_name"][0]

    # Vision features
    if data.get("vision_feat") is None or data["vision_feat"][0] is None:
        return None, None, None
    vis_feat = data["vision_feat"].to(device)

    # 3D coordinates
    coords_world = unproject_depth_to_world(
        depth_t, cam_to_world_t,
        fx=fx, fy=fy, cx=cx, cy=cy,
        original_height=H_orig, original_width=W_orig,
    )

    B, C_, Hf, Wf = vis_feat.shape
    assert (Hf, Wf) == depth_t.shape[1:], (
        f"Vision feature spatial dims ({Hf}, {Wf}) != depth spatial dims {depth_t.shape[1:]}"
    )

    # Flatten tensors
    feats_flat = vis_feat.permute(0, 2, 3, 1).reshape(-1, C_)
    coords_flat = coords_world.permute(0, 2, 3, 1).reshape(-1, 3)
    depth_flat = depth_t.reshape(-1)

    # Filter valid points
    coords_valid, feats_valid = filter_valid_points(
        coords_flat, depth_flat, feats_flat, cfg,
    )

    if coords_valid is None:
        return None, None, None

    # Get grid for this environment
    grid = grids[env_name]

    # Forward pass
    voxel_feat = grid.query_voxel_feature(coords_valid)
    pred_feat = decoder(voxel_feat)

    # Loss
    loss, cos_sim = cosine_similarity_loss(pred_feat, feats_valid)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach(), cos_sim.mean().detach(), coords_valid


# --------------------------------------------------------------------------- #
#  Core Training Function
# --------------------------------------------------------------------------- #

def train(
    dataset_dir: str,
    output_dir: str,
    config: dict,
    decoder: Optional[LatentDecoder] = None,
    device: str = "cuda",
    use_wandb: bool = False,
    use_tensorboard: bool = False,
    viser_server=None,
) -> Tuple[Dict[str, VoxelHashTable], LatentDecoder]:
    """
    Main training function for VoxelHashTable + MLP decoder.

    Args:
        dataset_dir: Path to directory containing HDF5 files
        output_dir: Path to save grids and decoder
        config: Training configuration dict
        device: torch device string
        use_wandb: Whether to use wandb logging
        use_tensorboard: Whether to use tensorboard logging

    Returns:
        grids: dict of env_name -> VoxelHashTable
        decoder: Trained LatentDecoder (latent -> vision features)
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Task name from dataset directory (e.g., "set_table" from "data/mapping/set_table")
    task_name = dataset_path.name
    grid_output_path = output_path / task_name
    grid_output_path.mkdir(parents=True, exist_ok=True)

    # Parse configuration
    cfg = TrainConfig.from_dict(config)

    # Get intrinsics and feature dimension
    model_config = config['dino_model'] if cfg.model_type == "dino" else config['eva_clip_model']
    default_feature_dim = model_config.get('feature_dim', 1024)

    fx, fy, cx, cy, H_orig, W_orig, feature_dim = get_intrinsics_and_feature_dim(
        dataset_path, cfg.target_envs, cfg.feature_key, default_feature_dim
    )
    intrinsics = (fx, fy, cx, cy, H_orig, W_orig)

    print(f"[Train] Feature dimension: {feature_dim}")
    print(f"[Train] Dataset: {dataset_path}")
    print(f"[Train] Output: {output_path}")

    # Create dataset and dataloader
    dataset, dataloader = create_dataset_and_loader(dataset_path, cfg)
    print(f"[Train] Loaded {len(dataset)} samples from {len(dataset.env_names)} environments")

    # Decoder: input_dim = feature_dim * num_levels (concatenated multi-level features)
    if decoder is None:
        decoder_input_dim = cfg.feature_dim * cfg.num_levels
        decoder = LatentDecoder(
            voxel_feature_dim=decoder_input_dim,
            hidden_dim=cfg.decoder_hidden_dim,
            output_dim=feature_dim,
        ).to(device)

        decoder_path = output_path / "latent_decoder.pt"
        if decoder_path.exists():
            decoder_ckpt = torch.load(decoder_path, map_location=device, weights_only=True)
            decoder_state = decoder_ckpt.get("model", decoder_ckpt)
            decoder.load_state_dict(decoder_state)
            loaded_epoch = decoder_ckpt.get("epoch", "unknown")
            print(f"[LOAD] Loaded existing decoder from {decoder_path} (epoch {loaded_epoch})")
    else:
        decoder = decoder.to(device)

    # Create per-environment VoxelHashTables
    grids: Dict[str, VoxelHashTable] = {}
    for env_name in dataset.env_names:
        grid_path = grid_output_path / (dataset.env_output_paths[env_name] + ".dense.pt")
        if grid_path.exists():
            grid = VoxelHashTable.load_dense(
                grid_path, device=device,
                one_to_one=cfg.one_to_one,
                resolution=cfg.resolution,
                num_levels=cfg.num_levels,
                level_scale=cfg.level_scale,
                feature_dim=cfg.feature_dim,
                hash_table_size=cfg.hash_table_size,
                scene_bound_min=cfg.scene_min,
                scene_bound_max=cfg.scene_max,
            )
            print(f"[LOAD] Loaded existing grid for {env_name}")
        else:
            grid = VoxelHashTable(
                one_to_one=cfg.one_to_one,
                resolution=cfg.resolution,
                num_levels=cfg.num_levels,
                level_scale=cfg.level_scale,
                feature_dim=cfg.feature_dim,
                hash_table_size=cfg.hash_table_size,
                scene_bound_min=cfg.scene_min,
                scene_bound_max=cfg.scene_max,
                device=device,
                mode="train",
            )
        grids[env_name] = grid

    # Single optimizer for decoder (if trainable) + all grids
    train_decoder = config.get("train_decoder", True)  # from raw config (not in TrainConfig)
    if train_decoder:
        params = list(decoder.parameters())
    else:
        for p in decoder.parameters():
            p.requires_grad = False
        params = []
    for grid in grids.values():
        params.extend(list(grid.parameters()))
    optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)

    # Setup logging
    logger = TrainLogger(
        output_path, dataset_path.name, config,
        use_wandb=use_wandb, use_tensorboard=use_tensorboard,
    )

    print(f"--- Starting training ({len(dataset)} samples, batch_size={cfg.batch_size}) ---")

    # Training loop
    for epoch in range(cfg.num_epochs):
        loss_history = []
        epoch_coords = {env_name: [] for env_name in dataset.env_names}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", file=sys.stdout)
        for i, data in enumerate(pbar):
            if not data:
                continue

            loss_val, cos_sim_val, coords_valid = train_step(
                data, grids, decoder, optimizer,
                cfg, intrinsics, device,
            )

            if loss_val is None:
                continue

            env_name = data["env_name"][0]
            epoch_coords[env_name].append(coords_valid.detach())
            loss_history.append(loss_val)
            global_step = epoch * len(dataloader) + i

            # Logging
            if (i + 1) % cfg.log_interval == 0:
                loss_scalar = loss_val.item()
                logger.log_step(global_step, loss_scalar, cos_sim_val.item(), coords_valid.shape[0])
                pbar.set_postfix_str(f"Loss: {loss_scalar:.4f}")

        # Epoch summary
        avg_loss = sum(l.item() for l in loss_history) / len(loss_history) if loss_history else 0
        print(f"[Epoch {epoch+1}/{cfg.num_epochs}] Avg. Loss: {avg_loss:.4f}")

        logger.log_epoch(epoch + 1, avg_loss, 0)

        # PCA visualization
        if cfg.run_pca and viser_server is not None and cfg.vis_interval > 0:
            if (epoch + 1) % cfg.vis_interval == 0:
                for env_name, coord_list in epoch_coords.items():
                    if not coord_list:
                        continue
                    all_coords = torch.cat(coord_list, dim=0)
                    all_coords = voxel_downsample(all_coords, voxel_size=0.02)
                    run_pca_visualization(
                        viser_server, all_coords, grids[env_name], decoder,
                        env_name, epoch + 1, config,
                    )

        # Save checkpoint
        is_last_epoch = (epoch + 1) == cfg.num_epochs
        if (epoch + 1) % cfg.save_interval == 0 or is_last_epoch:
            # Save decoder
            torch.save({"model": decoder.state_dict(), "epoch": epoch + 1}, output_path / "latent_decoder.pt")
            print(f"[SAVE] Saved decoder to {output_path / 'latent_decoder.pt'}")

            # Save per-environment grids
            export_bounds_min = config.get("export_bounds_min")
            export_bounds_max = config.get("export_bounds_max")
            for env_name, grid in grids.items():
                grid_path = grid_output_path / (dataset.env_output_paths[env_name] + ".pt")
                grid_path.parent.mkdir(parents=True, exist_ok=True)
                # Sparse for inference/visualization
                grid.save_sparse(grid_path, bounds_min=export_bounds_min, bounds_max=export_bounds_max)
                # Dense for training resume
                grid.save_dense(grid_path.with_suffix(".dense.pt"))
                print(f"[SAVE] Saved grid for {env_name} to {grid_path}")

    # Cleanup
    logger.close()

    return grids, decoder


# --------------------------------------------------------------------------- #
#  Main (standalone execution)
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train VoxelHashTable latent map.")
    parser.add_argument("--config", type=str, default="mapping/config/config.yaml",
                        help="Path to configuration YAML file")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Override dataset directory from config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory from config")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--no_tensorboard", action="store_true",
                        help="Disable tensorboard logging")
    parser.add_argument("--no_viser", action="store_true",
                        help="Disable viser visualization")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Determine paths
    dataset_dir = args.dataset_dir or config['dataset_dir']
    output_dir = args.output_dir or config.get('output_dir', 'models')

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] Device: {device}")

    # Save config to output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "config.yaml", 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    # Setup viser server for PCA visualization
    viser_server = None
    if config.get('run_pca', False) and not args.no_viser:
        import viser
        viser_server = viser.ViserServer(host="0.0.0.0", port=8080)
        print("[Init] Viser server started at http://0.0.0.0:8080")

    # Run training
    use_wandb = config.get('wandb', {}).get('enabled', True) and not args.no_wandb
    use_tensorboard = not args.no_tensorboard

    grids, decoder = train(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        config=config,
        device=device,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        viser_server=viser_server,
    )

    print(f"\n[Done] Training complete. Results saved to {output_dir}")

    # Keep viser server alive after training
    if viser_server is not None:
        print("[VIS] Viser server running. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
