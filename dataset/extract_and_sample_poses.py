"""
Camera pose extraction from .pt data files.

Usage:
    # Default: data_dir=data/, output_dir=dataset/camera_params, sample_ratio=0.1
    python dataset/extract_and_sample_poses.py

    # Custom options
    python dataset/extract_and_sample_poses.py --data_dir data/ --sample_ratio 0.2

Outputs:
    - dataset/camera_params/intrinsic.npy: (3, 3) intrinsic matrix
    - dataset/camera_params/camera_poses.npy: Dict of {key: (N, 7)}
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
from tqdm import tqdm


def sample_poses_uniform(poses: np.ndarray, ratio: float, offset: int = 0) -> np.ndarray:
    """Sample poses uniformly with optional offset for time step diversity."""
    n_poses = len(poses)
    n_samples = max(1, int(n_poses * ratio))

    if n_samples >= n_poses:
        return poses

    interval = n_poses / n_samples
    start_offset = (offset % max(1, int(interval))) if offset > 0 else 0
    indices = np.array(
        [min(int(start_offset + i * interval), n_poses - 1) for i in range(n_samples)]
    )
    indices = np.unique(indices)

    return poses[indices]


def process_pt_file(file_path: str) -> dict:
    """Process a single .pt file and extract camera params."""
    import torch

    file_name = os.path.basename(file_path)
    file_stem = os.path.splitext(file_name)[0]

    print(f"\n[Worker] Processing {file_name}...")

    data = None
    try:
        data = torch.load(file_path, map_location="cpu", mmap=True)
    except Exception as e:
        print(f"[Worker] mmap load failed ({e}), trying standard load...")
        try:
            data = torch.load(file_path, map_location="cpu")
        except Exception as e2:
            print(f"[Worker] Standard load failed: {e2}")
            return None

    result = {"intrinsic": None, "camera_poses": {}}

    if "intrinsic" in data:
        intrinsic = data["intrinsic"]
        if isinstance(intrinsic, torch.Tensor):
            intrinsic = intrinsic.numpy()
        result["intrinsic"] = intrinsic

    traj_keys = [k for k in data.keys() if k.startswith("traj_")]
    cameras = ["fetch_head", "fetch_hand"]

    for key in traj_keys:
        traj_data = data[key]
        for cam_name in cameras:
            if cam_name in traj_data:
                cam_data = traj_data[cam_name]
                if "camera_pose" in cam_data:
                    camera_pose = cam_data["camera_pose"]
                    if isinstance(camera_pose, torch.Tensor):
                        camera_pose = camera_pose.numpy()
                    full_key = f"{file_stem}_{key}_{cam_name}"
                    result["camera_poses"][full_key] = camera_pose

    return result


def extract_params(data_dir: str, output_dir: str, sample_ratio: float = 1.0):
    """Extract camera params from all .pt files in data_dir."""
    output_int_file = os.path.join(output_dir, "intrinsic.npy")
    output_pose_file = os.path.join(output_dir, "camera_poses.npy")

    os.makedirs(output_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    if not pt_files:
        print(f"No .pt files found in {data_dir}")
        return

    print(f"Found {len(pt_files)} .pt files to process.")

    temp_dir = os.path.join(output_dir, "temp_extraction")
    os.makedirs(temp_dir, exist_ok=True)

    temp_files = []

    for file_path in pt_files:
        file_name = os.path.basename(file_path)
        temp_out = os.path.join(temp_dir, f"{file_name}.npy")
        temp_files.append(temp_out)

        if os.path.exists(temp_out):
            os.remove(temp_out)

        print(f"\n--- Launching worker for {file_name} ---")
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            file_path,
            temp_out,
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {file_path}: {e}")

    print("\nMerging results...")
    final_camera_poses = {}
    final_intrinsic = None

    for temp_f in tqdm(temp_files, desc="Merging"):
        if not os.path.exists(temp_f):
            print(f"Warning: Missing temp file {temp_f}")
            continue

        try:
            data = np.load(temp_f, allow_pickle=True).item()
            if "camera_poses" in data:
                final_camera_poses.update(data["camera_poses"])
            if final_intrinsic is None and data.get("intrinsic") is not None:
                final_intrinsic = data["intrinsic"]
        except Exception as e:
            print(f"Error merging {temp_f}: {e}")

    # Apply sampling if needed
    if sample_ratio < 1.0:
        total_before = sum(len(p) for p in final_camera_poses.values())
        for i, key in enumerate(final_camera_poses):
            final_camera_poses[key] = sample_poses_uniform(
                final_camera_poses[key], sample_ratio, offset=i
            )
        total_after = sum(len(p) for p in final_camera_poses.values())
        print(f"Sampled {total_before} -> {total_after} poses ({sample_ratio:.0%})")

    if final_intrinsic is not None:
        np.save(output_int_file, final_intrinsic)
        print(f"\nSaved intrinsic to {output_int_file}")

    if final_camera_poses:
        np.save(output_pose_file, final_camera_poses)
        print(f"Saved {len(final_camera_poses)} camera poses to {output_pose_file}")

    print("Cleaning up temp files...")
    shutil.rmtree(temp_dir)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Extract camera poses from .pt files")
    parser.add_argument("--data_dir", type=str, default="data/", help="Directory containing .pt files")
    parser.add_argument("--output_dir", type=str, default="dataset/camera_params", help="Output directory (default: dataset/camera_params)")
    parser.add_argument("--sample_ratio", type=float, default=0.2, help="Sampling ratio (0.0-1.0)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    args, remaining = parser.parse_known_args()

    if args.worker and len(remaining) >= 2:
        file_path, out_path = remaining[0], remaining[1]
        result = process_pt_file(file_path)
        if result:
            np.save(out_path, result, allow_pickle=True)
            print(f"[Worker] Saved result to {out_path}")
    elif args.data_dir:
        output_dir = args.output_dir or "dataset/camera_params"
        extract_params(args.data_dir, output_dir, args.sample_ratio)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
