"""Training configuration dataclass for VoxelHashTable-based latent map training."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TrainConfig:
    """Training configuration for VoxelHashTable + LatentDecoder."""

    # Scene bounds
    scene_min: Tuple[float, float, float]
    scene_max: Tuple[float, float, float]

    # VoxelHashTable parameters
    resolution: float
    num_levels: int
    level_scale: float
    feature_dim: int
    hash_table_size: int
    one_to_one: bool

    # Decoder (LatentDecoder)
    decoder_hidden_dim: int

    # Training
    learning_rate: float
    batch_size: int
    num_epochs: int
    log_interval: int
    save_interval: int

    # Model
    model_type: str
    image_size: int
    patch_size: int
    feature_key: str  # HDF5 dataset key for vision features ("dino" or "eva_clip")

    # Optional
    run_pca: bool = False
    vis_interval: int = 0
    num_images: int = -1
    target_envs: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, config: dict) -> "TrainConfig":
        """Create TrainConfig from config dictionary."""
        grid = config["grid"]
        training = config["training"]
        decoder = config["decoder"]

        model_type = config["model_type"]
        if model_type == "dino":
            model_config = config["dino_model"]
        elif model_type == "eva_clip":
            model_config = config["eva_clip_model"]
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'dino' or 'eva_clip'.")

        return cls(
            scene_min=tuple(config["scene_min"]),
            scene_max=tuple(config["scene_max"]),
            resolution=grid["resolution"],
            num_levels=grid["levels"],
            level_scale=grid["level_scale"],
            feature_dim=grid["feature_dim"],
            hash_table_size=grid["hash_table_size"],
            one_to_one=grid.get("one_to_one", True),
            decoder_hidden_dim=decoder["hidden_dim"],
            learning_rate=training["optimizer_lr"],
            batch_size=training.get("batch_size", 1),
            num_epochs=training["epochs"],
            log_interval=training["log_interval"],
            save_interval=training.get("save_interval", 10),
            model_type=model_type,
            image_size=model_config["image_size"],
            patch_size=model_config["patch_size"],
            feature_key=model_type,
            run_pca=config.get("run_pca", False),
            vis_interval=config.get("vis_interval", 0),
            num_images=config.get("num_images", -1),
            target_envs=config.get("target_envs", []),
        )
