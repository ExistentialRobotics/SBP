import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
#  small helpers                                                              #
# --------------------------------------------------------------------------- #
def _primes(dev):  # 3-tuple of large primes
    return torch.tensor([73856093, 19349669, 83492791], device=dev, dtype=torch.long)


def _corner_offsets(dev):  # (8,3) corner offsets
    return torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        device=dev,
        dtype=torch.long,
    )


# --------------------------------------------------------------------------- #
#  base level — shared grid indexing & trilinear query                        #
# --------------------------------------------------------------------------- #
class _BaseLevel(nn.Module):
    """Common logic for _TrainLevel and _InferLevel.

    Subclasses must set: res, one_to_one, smin, grid_shape, primes,
    corner_offsets, buckets, d, voxel_features.
    Subclasses must implement: _lookup(idxg) -> Tensor.
    """

    res: float
    one_to_one: bool
    smin: torch.Tensor
    grid_shape: torch.Tensor
    primes: torch.Tensor
    corner_offsets: torch.Tensor
    buckets: int
    d: int

    def _grid_indices(self, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute continuous coords, base voxel, and 8-corner indices.

        Returns (q, base, idx) where:
            q:    (N, 3) continuous grid coordinates
            base: (N, 3) floor of q (long)
            idx:  (N, 8, 3) corner voxel indices

        Note:
            In hash mode (one_to_one=False), no bounds checking is performed.
            Callers must ensure points lie within scene bounds before querying.
            Out-of-bounds points will hash to arbitrary buckets and return
            meaningless features. Use filter_valid_points() during training
            or apply manual bounds filtering during inference.
        """
        if self.one_to_one:
            q = (pts - self.smin) / self.res
            mask = ((q < 0) | (q >= self.grid_shape)).any(dim=-1)
            assert not mask.any(), f"Points out of bounds:\n{pts[mask]}"
        else:
            q = pts / self.res
        base = torch.floor(q).long()
        idx = base[:, None, :] + self.corner_offsets[None, :, :]
        if self.one_to_one:
            idx = torch.min(idx, (self.grid_shape - 1).view(1, 1, 3))
        return q, base, idx

    def _lookup(self, idxg: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def query(self, pts: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q, base, idx = self._grid_indices(pts)
        feat = self._lookup(idx)
        frac = q - base.float()
        wx = torch.stack([1 - frac[:, 0], frac[:, 0]], 1)
        wy = torch.stack([1 - frac[:, 1], frac[:, 1]], 1)
        wz = torch.stack([1 - frac[:, 2], frac[:, 2]], 1)
        w = wx[:, [0, 1, 0, 1, 0, 1, 0, 1]] * wy[:, [0, 0, 1, 1, 0, 0, 1, 1]] * wz[:, [0, 0, 0, 0, 1, 1, 1, 1]]
        return (feat * w.unsqueeze(-1)).sum(1)


# --------------------------------------------------------------------------- #
#  dense level (train)                                                        #
# --------------------------------------------------------------------------- #
class _TrainLevel(_BaseLevel):
    def __init__(self, res, d, buckets, smin, smax, primes, one_to_one, dev):
        super().__init__()
        self.res, self.d, self.buckets = res, d, buckets
        self.one_to_one = one_to_one

        self.register_buffer("smin", torch.tensor(smin).float().to(dev), persistent=False)
        self.smin: torch.Tensor
        self.register_buffer("smax", torch.tensor(smax).float().to(dev), persistent=False)
        self.smax: torch.Tensor

        self.register_buffer("primes", primes.clone(), persistent=False)
        self.primes: torch.Tensor

        xs = torch.arange(smin[0], smax[0], res, device=dev)
        ys = torch.arange(smin[1], smax[1], res, device=dev)
        zs = torch.arange(smin[2], smax[2], res, device=dev)
        self.register_buffer(
            "grid_shape",
            torch.tensor((xs.numel(), ys.numel(), zs.numel()), device=dev),
            persistent=False,
        )
        self.grid_shape: torch.Tensor

        # indexing="ij" for (x,y,z) meshgrid
        # [(x,y,z) for x in xs for y in ys for z in zs]
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")

        self.register_buffer("coords", torch.stack([gx, gy, gz], -1).view(-1, 3), persistent=False)
        self.coords: torch.Tensor
        self.N = self.coords.size(0)

        self.register_buffer("corner_offsets", _corner_offsets(dev), persistent=False)
        self.corner_offsets: torch.Tensor

        if self.one_to_one:
            # One-to-one mapping: each coord maps to a unique voxel
            # logging.info("Using one-to-one mapping for voxel features. The behavior is like a dense grid.")
            self.buckets = self.N
            n_collisions = 0
            self.primes[2] = 1
            self.primes[1] = self.grid_shape[2]
            self.primes[0] = self.grid_shape[1] * self.grid_shape[2]
        else:
            # Hash mapping: coords to buckets
            # logging.info("Using hash mapping for voxel features.")
            # idx = torch.floor((self.coords - self.smin) / self.res).long()
            # hv = (idx * self.primes).sum(-1) % self.buckets
            # _, counts = hv.unique(return_counts=True)
            # n_collisions = int(counts[counts > 1].sum())
            n_collisions = 0

        self.voxel_features = nn.Parameter(torch.zeros(self.buckets, self.d, device=dev).normal_(0, 0.01))

        self.register_buffer("col", torch.tensor(n_collisions, device=self.voxel_features.device), persistent=False)
        self.col: torch.Tensor

        self.register_buffer("access", torch.zeros(self.buckets, dtype=torch.bool, device=dev), persistent=False)
        self.access: torch.BoolTensor

        logging.info(f"Level filled: {self.buckets} voxels")

    # ---------- public utils
    @torch.no_grad()
    def collision_stats(self):
        return dict(total=self.N, col=int(self.col))

    @torch.no_grad()
    def get_accessed_indices(self):
        return torch.nonzero(self.access).flatten()

    @torch.no_grad()
    def reset_access_log(self):
        self.access.zero_()

    @torch.no_grad()
    def export_sparse(self, bounds_min=None, bounds_max=None):
        accessed_indices = self.get_accessed_indices()

        # Optional spatial bounding box filter
        if bounds_min is not None and bounds_max is not None:
            coords = self.coords[accessed_indices]
            bmin = torch.tensor(bounds_min, dtype=coords.dtype, device=coords.device)
            bmax = torch.tensor(bounds_max, dtype=coords.dtype, device=coords.device)
            in_bounds = ((coords >= bmin) & (coords <= bmax)).all(dim=-1)
            accessed_indices = accessed_indices[in_bounds]

        return dict(
            one_to_one=self.one_to_one,
            resolution=self.res,
            buckets=self.buckets,
            grid_shape=self.grid_shape.cpu(),
            primes=self.primes.cpu(),
            accessed_indices=accessed_indices.cpu(),
            coords=self.coords[accessed_indices].cpu() if self.one_to_one else self.coords.cpu(),
            features=self.voxel_features[accessed_indices].cpu(),
            smin=self.smin.cpu(),
            smax=self.smax.cpu(),
            col=self.col.cpu(),
        )

    # ---------- internals
    def _lookup(self, idxg: torch.Tensor) -> torch.Tensor:
        vid = (idxg * self.primes).sum(-1) % self.buckets
        self.access[vid] = True  # log access
        return self.voxel_features[vid]


# --------------------------------------------------------------------------- #
#  sparse level (infer)                                                       #
# --------------------------------------------------------------------------- #
class _InferLevel(_BaseLevel):

    def __init__(self, state_dict):
        """
        Load a state_dict from _TrainLevel.export_sparse()
        """
        super().__init__()
        self.one_to_one = state_dict["one_to_one"]
        self.res = float(state_dict["resolution"])
        self.buckets = state_dict["buckets"]

        self.register_buffer("grid_shape", state_dict["grid_shape"].to(torch.long), persistent=False)
        self.grid_shape: torch.Tensor

        self.register_buffer("primes", state_dict["primes"].to(torch.long), persistent=False)
        self.primes: torch.Tensor

        self.register_buffer("coords", state_dict["coords"].to(torch.float32), persistent=False)
        self.coords: torch.Tensor

        self.voxel_features = nn.Parameter(state_dict["features"].to(torch.float32), requires_grad=False)
        self.d = self.voxel_features.size(-1)

        self.register_buffer("smin", state_dict["smin"].to(torch.float32), persistent=False)
        self.smin: torch.Tensor
        self.register_buffer("smax", state_dict["smax"].to(torch.float32), persistent=False)
        self.smax: torch.Tensor

        self.register_buffer("col", state_dict["col"].to(torch.long), persistent=False)
        self.col: torch.Tensor

        # build the extra mapping from hash value to voxel index
        # if -1, then not accessed during training
        self.register_buffer(
            "access",
            torch.full((self.buckets,), -1, dtype=torch.long, device=self.coords.device),
            persistent=False,
        )
        self.access: torch.Tensor
        accessed_indices = state_dict["accessed_indices"]
        self.access[accessed_indices] = torch.arange(accessed_indices.numel(), device=self.coords.device)

        self.register_buffer("corner_offsets", _corner_offsets(self.coords.device), persistent=False)
        self.corner_offsets: torch.Tensor

    def collision_stats(self):
        return dict(total=self.coords.size(0), col=int(self.col))

    def get_accessed_indices(self):
        return torch.nonzero(self.access >= 0).flatten()

    def reset_access_log(self):
        pass

    def _lookup(self, idxg: torch.Tensor) -> torch.Tensor:
        vid = (idxg * self.primes).sum(-1) % self.buckets
        vid = self.access[vid]
        valid = vid >= 0
        out = torch.zeros(*idxg.shape[:-1], self.d, device=self.coords.device, dtype=self.voxel_features.dtype)
        if valid.any():
            out[valid] = self.voxel_features[vid[valid]]
        return out


# --------------------------------------------------------------------------- #
#  public pyramid                                                             #
# --------------------------------------------------------------------------- #
class VoxelHashTable(nn.Module):
    """
    mode='train' → dense levels, mode='infer' → sparse levels
    """

    def __init__(
        self,
        one_to_one: bool = True,
        resolution: float = 0.12,
        num_levels: int = 2,
        level_scale: float = 2.0,
        feature_dim: int = 32,
        hash_table_size: int = 2**21,
        scene_bound_min: tuple[float, ...] = (-2.6, -8.1, 0),
        scene_bound_max: tuple[float, ...] = (4.6, 4.7, 3.1),
        device: str = "cuda:0",
        mode: str = "train",
        sparse_data: Optional[Dict] = None,
    ):
        super().__init__()
        self.mode, self.d = mode, feature_dim
        dev = torch.device(device)
        primes = _primes(dev)
        self.levels = nn.ModuleList()

        if mode == "train":
            # Iterate coarse → fine by reversing the exponent.
            for lv in range(num_levels):
                res = resolution * (level_scale ** (num_levels - 1 - lv))
                self.levels.append(
                    _TrainLevel(
                        res, feature_dim, hash_table_size, scene_bound_min, scene_bound_max, primes, one_to_one, dev
                    )
                )
        elif mode == "infer":
            if sparse_data is None:
                raise ValueError("sparse_data is required in infer mode")
            # Sort payloads from coarse (larger res) → fine (smaller res)
            sorted_levels = sorted(sparse_data["levels"], key=lambda p: p["resolution"], reverse=True)
            for level_state_dict in sorted_levels:
                self.levels.append(_InferLevel(level_state_dict))
        else:
            raise ValueError("mode must be 'train' or 'infer'")

    # forward -----------------------------------------------------------------
    def query_voxel_feature(self, pts):  # (M,3) → (M, d*L)
        per = [lv.query(pts) for lv in self.levels]
        return torch.cat(per, -1)

    # utils -------------------------------------------------------------------
    @torch.no_grad()
    def collision_stats(self):
        return {f"level_{i}": lv.collision_stats() for i, lv in enumerate(self.levels)}

    @torch.no_grad()
    def get_accessed_indices(self):
        return [lv.get_accessed_indices() for lv in self.levels]

    @torch.no_grad()
    def reset_access_log(self):
        for lv in self.levels:
            lv.reset_access_log()

    # save / load -------------------------------------------------------------
    @torch.no_grad()
    def export_sparse(self, bounds_min=None, bounds_max=None):
        if self.mode != "train":
            raise RuntimeError("export_sparse only in train mode")
        return dict(num_levels=len(self.levels), feature_dim=self.d,
                    levels=[lv.export_sparse(bounds_min=bounds_min, bounds_max=bounds_max) for lv in self.levels])

    # dense weight file
    def save_dense(self, path):
        num_levels = len(self.levels)
        finest = self.levels[-1]
        config = {
            "one_to_one": finest.one_to_one,
            "resolution": float(finest.res),
            "num_levels": num_levels,
            "level_scale": float(self.levels[0].res / self.levels[1].res) if num_levels > 1 else 2.0,
            "feature_dim": self.d,
            "hash_table_size": finest.buckets,
            "scene_bound_min": finest.smin.tolist(),
            "scene_bound_max": finest.smax.tolist(),
        }
        torch.save({"state_dict": self.state_dict(), "config": config}, path)

    # sparse file
    def save_sparse(self, path, bounds_min=None, bounds_max=None):
        torch.save(self.export_sparse(bounds_min=bounds_min, bounds_max=bounds_max), path)

    @staticmethod
    def load_dense(path, device="cuda:0", **kwargs):
        chk = torch.load(path, map_location="cpu")
        # Use saved config if available, then allow caller kwargs to override
        cfg = chk.get("config", {})
        cfg.update(kwargs)
        vt = VoxelHashTable(
            one_to_one=cfg.get("one_to_one", True),
            resolution=cfg.get("resolution", 0.12),
            num_levels=cfg.get("num_levels", 2),
            level_scale=cfg.get("level_scale", 2.0),
            feature_dim=cfg.get("feature_dim", 32),
            hash_table_size=cfg.get("hash_table_size", 2**21),
            scene_bound_min=tuple(cfg.get("scene_bound_min", (-2.6, -8.1, 0))),
            scene_bound_max=tuple(cfg.get("scene_bound_max", (4.6, 4.7, 3.1))),
            device=device,
            mode="train",
        )
        vt.load_state_dict(chk["state_dict"])
        return vt.to(device)

    @staticmethod
    def load_sparse(path, device="cuda:0"):
        sparse = torch.load(path, map_location="cpu")
        vt = VoxelHashTable(mode="infer", sparse_data=sparse, device=device)
        return vt.to(device)
