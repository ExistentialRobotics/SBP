import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
from pathlib import Path
import h5py

__all__ = ["HDF5Dataset", "collate_fn", "EnvBatchSampler", "hdf5_worker_init_fn"]



class HDF5Dataset(Dataset):
    """
    HDF5-based dataset for loading RGB, depth, poses, and optional DINO features.

    Expected HDF5 structure:
    - rgb: (N, H, W, 3) uint8
    - depth: (N, H, W) uint16 (mm)
    - poses: (N, 4, 4) float64 - world-to-camera (OpenCV convention)
    - intrinsics: (4,) float32 - camera intrinsics [fx, fy, cx, cy]
    - dino/eva_clip: (N, C, Hf, Wf) float32 (optional) - pre-computed vision features
    Note: File handles are cached for performance. Call close() when done or use
    as context manager.
    """

    def __init__(self, dataset_dir, target_envs, num_images, image_size, patch_size, feature_key=None):
        # File handle cache for faster access
        self._file_cache = {}
        self.image_size = image_size
        self.patch_size = patch_size
        self.feat_h = self.image_size // patch_size
        self.feature_key = feature_key  # HDF5 dataset key ("dino", "eva_clip", or None)

        self.samples = []
        self.cam_to_world_poses = {}
        self.env_names = []
        self._env_to_scene = {}            # {env_name: scene_name}
        self.env_output_paths = {}         # {env_name: relative path for output}

        # Pre-loaded data: depth/sam resized to feat_h resolution
        self._preloaded = {}  # {hdf5_path_str: {"depth": np.array, "sam": np.array}}

        dataset_dir = Path(dataset_dir)

        if not target_envs:
            # Look for HDF5 files
            hdf5_files = sorted(list(dataset_dir.rglob("*.hdf5")))
        else:
            # Check if target_envs are paths to HDF5 files or task directories
            hdf5_files = []
            for env in target_envs:
                path = dataset_dir / env
                if path.is_file() and path.suffix == ".hdf5":
                    hdf5_files.append(path)
                elif path.is_dir():
                    hdf5_files.extend(sorted(list(path.glob("*.hdf5"))))

        print(f"\nFound {len(hdf5_files)} HDF5 files.")
        self.hdf5_files = hdf5_files

        for hdf5_path in self.hdf5_files:
            env_name = hdf5_path.stem  # episode_00090010
            scene_name = hdf5_path.parent.name
            # Include task name to be unique: task-0009_episode_00090010
            unique_env_name = f"{scene_name}_{env_name}"
            self._env_to_scene[unique_env_name] = scene_name
            # Relative output path mirroring dataset_dir structure
            # e.g. data/set_table/37/ep.hdf5 → set_table/37/ep
            rel = hdf5_path.relative_to(dataset_dir)
            self.env_output_paths[unique_env_name] = str(rel.parent / rel.stem)

            print(f"--- Processing environment: {unique_env_name} ({hdf5_path.name}) ---")

            with h5py.File(hdf5_path, 'r') as f:
                if "depth" not in f or "poses" not in f:
                    print(f"  [WARN] Missing depth or poses in {hdf5_path}. Skipping.")
                    continue

                num_frames = f["depth"].shape[0]

                # Load poses (N, 4, 4) world-to-camera (OpenCV convention)
                # Convert to cam_to_world_poses
                poses_w2c = f["poses"][:]  # (N, 4, 4)
                poses_c2w = np.linalg.inv(poses_w2c)
                self.cam_to_world_poses[unique_env_name] = poses_c2w

                # Check for vision features
                has_features = self.feature_key is not None and self.feature_key in f
                if self.feature_key is not None and not has_features:
                    print(f"  [WARN] feature_key='{self.feature_key}' but dataset missing in {hdf5_path}.")

                # Check for SAM masks
                has_sam = "sam_masks" in f
                if has_sam:
                    print(f"  [INFO] SAM masks available ({f['sam_masks'].shape[0]} frames)")

                # Indices to use
                indices = range(num_frames)
                if num_images > 0:
                    indices = indices[:num_images]
                n_load = len(indices)

                # --- Pre-load depth/seg/sam at feat_h resolution --- #
                hdf5_key = str(hdf5_path)
                preloaded = {}
                batch_load = 1000

                # Depth → float32 meters at (feat_h, feat_h)
                chunks = []
                for s in range(0, n_load, batch_load):
                    e = min(s + batch_load, n_load)
                    raw = f['depth'][s:e].astype(np.float32) / 1000.0
                    t = torch.from_numpy(raw).unsqueeze(1)
                    t = F.interpolate(t, (self.feat_h, self.feat_h), mode="nearest-exact").squeeze(1)
                    chunks.append(t.numpy())
                preloaded["depth"] = np.concatenate(chunks, axis=0)

                # SAM masks → int32 at (feat_h, feat_h)
                if has_sam:
                    chunks = []
                    for s in range(0, n_load, batch_load):
                        e = min(s + batch_load, n_load)
                        raw = f['sam_masks'][s:e].astype(np.float32)
                        t = torch.from_numpy(raw).unsqueeze(1)
                        t = F.interpolate(t, (self.feat_h, self.feat_h), mode="nearest-exact").squeeze(1)
                        chunks.append(t.to(torch.int32).numpy())
                    preloaded["sam"] = np.concatenate(chunks, axis=0)

                self._preloaded[hdf5_key] = preloaded
                mem_mb = sum(v.nbytes for v in preloaded.values()) / 1024 / 1024
                cached_keys = "/".join(preloaded.keys())
                print(f"  [PRELOAD] Cached {cached_keys} at {self.feat_h}x{self.feat_h}: {mem_mb:.1f} MB ({n_load} frames)")

                for idx in indices:
                    self.samples.append({
                        "hdf5_path": hdf5_key,
                        "idx": idx,
                        "env_name": unique_env_name,
                        "has_features": has_features,
                        "has_sam": has_sam,
                    })

                self.env_names.append(unique_env_name)

        print(f"\nDataset initialized with {len(self.samples)} total samples from {len(self.env_names)} environments.")

    def _get_file(self, hdf5_path: str):
        """Get cached file handle or open new one. Returns None if file cannot be opened."""
        if hdf5_path not in self._file_cache:
            if not Path(hdf5_path).exists():
                return None
            self._file_cache[hdf5_path] = h5py.File(hdf5_path, 'r')
        return self._file_cache[hdf5_path]

    def close(self):
        """Close all cached file handles."""
        for f in self._file_cache.values():
            f.close()
        self._file_cache.clear()

    def __del__(self):
        """Cleanup file handles on deletion."""
        self.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        hdf5_path = sample_info["hdf5_path"]
        frame_idx = sample_info["idx"]
        env_name = sample_info["env_name"]
        has_features = sample_info["has_features"]
        has_sam = sample_info.get("has_sam", False)

        preloaded = self._preloaded.get(hdf5_path)
        if preloaded is None:
            return None

        # Depth: already (feat_h, feat_h) float32 meters
        depth_t = torch.from_numpy(preloaded["depth"][frame_idx])

        # Pose
        E_cv = self.cam_to_world_poses[env_name][frame_idx][:3, :]
        cam_to_world_t = torch.from_numpy(E_cv).float()

        # Vision features: only remaining HDF5 read per frame (lzf, per-frame chunked → fast)
        vision_feat = None
        if self.feature_key is not None and has_features:
            f = self._get_file(hdf5_path)
            if f is not None and self.feature_key in f:
                vision_feat = torch.from_numpy(f[self.feature_key][frame_idx].astype(np.float32))

        # SAM mask: already (feat_h, feat_h) int32
        sam_mask = None
        if has_sam and "sam" in preloaded:
            sam_mask = torch.from_numpy(preloaded["sam"][frame_idx]).long()

        return {
            "depth_t": depth_t,
            "cam_to_world_t": cam_to_world_t,
            "env_name": env_name,
            "vision_feat": vision_feat,
            "sam_mask": sam_mask,
        }


def collate_fn(batch):
    """
    Custom collate function to filter out None values from the batch.
    This is used to handle cases where `__getitem__` returns None for invalid samples.
    Also handles non-tensor fields like window_instance_ids (sets).
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return {}

    # Separate tensor and non-tensor fields
    tensor_keys = []
    non_tensor_keys = []
    for key in batch[0].keys():
        if all(isinstance(s[key], (torch.Tensor, np.ndarray)) for s in batch):
            tensor_keys.append(key)
        else:
            non_tensor_keys.append(key)

    # Collate tensor fields using default_collate
    tensor_batch = [{k: sample[k] for k in tensor_keys} for sample in batch]
    result = torch.utils.data.dataloader.default_collate(tensor_batch)

    # Keep non-tensor fields as lists
    for key in non_tensor_keys:
        result[key] = [sample[key] for sample in batch]

    return result


def hdf5_worker_init_fn(worker_id: int):
    """Clear stale h5py file handles after fork. Workers re-open on demand via _get_file."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_info.dataset._file_cache.clear()


class EnvBatchSampler(Sampler):
    """
    BatchSampler that ensures each batch contains samples from only one environment.
    Samples within each env are shuffled, and env order is also shuffled each epoch.
    """

    def __init__(self, dataset: HDF5Dataset, batch_size: int, drop_last: bool = False):
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group sample indices by env_name
        self.env_to_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            self.env_to_indices[sample["env_name"]].append(idx)

        self.env_names = list(self.env_to_indices.keys())

    def __iter__(self):
        # Shuffle env order
        env_order = np.random.permutation(self.env_names).tolist()

        for env_name in env_order:
            indices = self.env_to_indices[env_name].copy()
            np.random.shuffle(indices)

            # Yield batches for this env
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        total = 0
        for indices in self.env_to_indices.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                n_batches += 1
            total += n_batches
        return total
