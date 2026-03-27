"""
Generate vision embeddings for a single HDF5 file.

This script reads RGB frames from an HDF5 file, generates embeddings using
either DINOv3 (text-aligned) or EVA-CLIP, and writes them back to the same
file as a 'dino' or 'eva_clip' dataset.

Usage:
    # DINOv3 (default)
    python dataset/generate_embedding.py \
        --model dino \
        --input_path /mnt/mshab_dataset/data/set_table/40/episode_0000.hdf5 \
        --batch_size 64 --image_size 480

    # EVA-CLIP
    python dataset/generate_embedding.py \
        --model eva_clip \
        --input_path /mnt/mshab_dataset/data/set_table/40/episode_0000.hdf5 \
        --batch_size 64 --image_size 518
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Add project root to sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class HDF5Dataset(Dataset):
    """Dataset for loading RGB frames from HDF5."""

    def __init__(self, hdf5_path: str, transform=None):
        self.hdf5_path = hdf5_path
        self.transform = transform

        with h5py.File(hdf5_path, "r") as f:
            if "rgb" not in f:
                raise ValueError(f"No 'rgb' dataset found in {hdf5_path}")
            self.length = f["rgb"].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        with h5py.File(self.hdf5_path, "r") as f:
            rgb = f["rgb"][idx]  # (H, W, 3) uint8

        # Convert to tensor: (3, H, W) float [0, 1]
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor


def load_dino_model(
    args: argparse.Namespace, device: str
) -> torch.nn.Module:
    """Load DINOv3 (text-aligned) model."""
    from mapping.models.vision_backbone import DINOv3Wrapper

    repo_dir = Path("third_party/dinov3")
    if not repo_dir.exists():
        repo_dir = Path(__file__).resolve().parent.parent / "third_party" / "dinov3"

    if not repo_dir.exists():
        tqdm.write(
            f"Error: Could not find third_party/dinov3 at "
            f"./third_party/dinov3 or {repo_dir}"
        )
        sys.exit(1)

    tqdm.write(f"Loading DINOv3 (text-aligned) model from {repo_dir}...")
    tqdm.write(f"Loading backbone weights from: {args.backbone_weights}")
    tqdm.write(f"Loading DINOTxt weights from: {args.dinotxt_weights}")
    dinotxt_model, tokenizer = torch.hub.load(
        str(repo_dir),
        "dinov3_vitl16_dinotxt_tet1280d20h24l",
        source="local",
        backbone_weights=args.backbone_weights,
        weights=args.dinotxt_weights,
    )
    model = DINOv3Wrapper(dinotxt_model, tokenizer).to(device).eval()
    tqdm.write(f"DINOv3 model loaded. Feature dim: {model.feature_dim}")
    return model


def load_eva_clip_model(
    args: argparse.Namespace, device: str
) -> torch.nn.Module:
    """Load EVA-CLIP model."""
    import open_clip

    from mapping.models.vision_backbone import EvaClipWrapper

    tqdm.write(
        f"Loading EVA-CLIP model: {args.eva_clip_model} "
        f"(pretrained: {args.eva_clip_pretrained})..."
    )
    clip_model, _, _ = open_clip.create_model_and_transforms(
        args.eva_clip_model, args.eva_clip_pretrained
    )
    model = EvaClipWrapper(clip_model).to(device).eval()
    tqdm.write(f"EVA-CLIP model loaded. Feature dim: {model.output_dim}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Generate vision embeddings (DINOv3 or EVA-CLIP) for a single HDF5 file."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["dino", "eva_clip"],
        default="dino",
        help="Vision model to use: 'dino' (DINOv3 text-aligned) or 'eva_clip' (EVA02-L-14).",
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to a single .hdf5 file."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Image size for inference. Default: 224 for eva_clip, 512 for dino.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for inference."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading. Default 0 for HDF5 safety.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset in HDF5.",
    )
    parser.add_argument(
        "--flush_threshold",
        type=int,
        default=50,
        help="Number of batches to buffer before flushing to disk.",
    )

    # DINOv3-specific arguments
    parser.add_argument(
        "--backbone_weights",
        type=str,
        default="/mnt/dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        help="[dino] Path to DINOv3 backbone weights.",
    )
    parser.add_argument(
        "--dinotxt_weights",
        type=str,
        default="/mnt/dino/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
        help="[dino] Path to DINOTxt vision head and text encoder weights.",
    )

    # EVA-CLIP-specific arguments
    parser.add_argument(
        "--eva_clip_model",
        type=str,
        default="EVA02-L-14",
        help="[eva_clip] Open-CLIP model name.",
    )
    parser.add_argument(
        "--eva_clip_pretrained",
        type=str,
        default="merged2b_s4b_b131k",
        help="[eva_clip] Pretrained weights tag.",
    )

    args = parser.parse_args()

    # Set model-specific default image size
    if args.image_size is None:
        args.image_size = 224 if args.model == "eva_clip" else 256

    hdf5_file = Path(args.input_path)
    if not hdf5_file.exists():
        tqdm.write(f"Error: Input file {hdf5_file} does not exist.")
        return 1

    if hdf5_file.suffix != ".hdf5":
        tqdm.write(f"Error: Input file {hdf5_file} is not an HDF5 file.")
        return 1

    dataset_name = args.model  # 'dino' or 'eva_clip'

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tqdm.write(f"Using device: {device}")
    tqdm.write(f"Model: {args.model}, Dataset name: '{dataset_name}'")

    # Check if dataset already exists
    with h5py.File(hdf5_file, "r") as f:
        if dataset_name in f:
            if not args.overwrite:
                tqdm.write(
                    f"Skipping {hdf5_file.name}: '{dataset_name}' dataset already exists "
                    "(use --overwrite to replace)."
                )
                return 0

    # If overwrite, delete existing dataset
    if args.overwrite:
        with h5py.File(hdf5_file, "r+") as f:
            if dataset_name in f:
                del f[dataset_name]
                tqdm.write(
                    f"Deleted existing '{dataset_name}' dataset from {hdf5_file.name}"
                )

    # Load Model
    if args.model == "dino":
        model = load_dino_model(args, device)
    elif args.model == "eva_clip":
        model = load_eva_clip_model(args, device)

    # Transform
    transform = transforms.Compose(
        [
            transforms.Resize(
                (args.image_size, args.image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
        ]
    )

    # Setup Dataset and DataLoader
    dataset = HDF5Dataset(hdf5_file, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    tqdm.write(f"Processing {len(dataset)} frames from {hdf5_file.name}...")

    buffer = []

    # Inference loop with memory efficient attention
    for batch_idx, images in enumerate(
        tqdm(dataloader, desc=f"Inference {hdf5_file.name}")
    ):
        images = images.to(device)
        with torch.no_grad(), sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        ):
            # model returns (B, C, Hf, Wf)
            features = model(images).cpu().numpy()
            buffer.append(features)

        # Flush if threshold reached
        if len(buffer) >= args.flush_threshold:
            data_to_write = np.concatenate(buffer, axis=0)
            buffer = []  # Clear buffer immediately to free memory

            with h5py.File(hdf5_file, "r+") as f:
                if dataset_name in f:
                    dset = f[dataset_name]
                    dset.resize(dset.shape[0] + data_to_write.shape[0], axis=0)
                    dset[-data_to_write.shape[0] :] = data_to_write
                else:
                    maxshape = (None,) + data_to_write.shape[1:]
                    f.create_dataset(
                        dataset_name,
                        data=data_to_write,
                        maxshape=maxshape,
                        chunks=(1,) + data_to_write.shape[1:],
                        compression="lzf",
                    )

    # Flush remaining buffer
    if buffer:
        data_to_write = np.concatenate(buffer, axis=0)
        buffer = []
        with h5py.File(hdf5_file, "r+") as f:
            if dataset_name in f:
                dset = f[dataset_name]
                dset.resize(dset.shape[0] + data_to_write.shape[0], axis=0)
                dset[-data_to_write.shape[0] :] = data_to_write
            else:
                maxshape = (None,) + data_to_write.shape[1:]
                f.create_dataset(
                    dataset_name,
                    data=data_to_write,
                    maxshape=maxshape,
                    chunks=(1,) + data_to_write.shape[1:],
                    compression="lzf",
                )

    # Verify
    with h5py.File(hdf5_file, "r") as f:
        if dataset_name in f:
            tqdm.write(
                f"Done. '{dataset_name}' shape: {f[dataset_name].shape}, "
                f"dtype: {f[dataset_name].dtype}"
            )
        else:
            tqdm.write(f"Error: '{dataset_name}' dataset was not created.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
