"""
SBP Dataset Module.

This module provides utilities for generating HDF5 datasets from ManiSkill environments.

Main Components:
    - HDF5DataSaver: Buffered HDF5 data saving
    - load_camera_poses: Load camera poses from npy files
    - render_episode: Render a single episode from poses
    - Pose augmentation utilities
"""

from dataset.render_from_camera_poses import (
    HDF5DataSaver,
    get_build_config_name,
    get_episode_name,
    get_filtered_task_plans,
    get_max_task_plan_count,
    load_camera_poses,
    poses_7d_to_extrinsic_cv,
    extrinsic_cv_to_sapien_pose,
)

from dataset.extract_and_sample_poses import (
    process_pt_file,
    extract_params,
)

__all__ = [
    # Core
    "HDF5DataSaver",
    "get_build_config_name",
    "get_episode_name",
    "get_filtered_task_plans",
    "get_max_task_plan_count",
    "load_camera_poses",
    "poses_7d_to_extrinsic_cv",
    "extrinsic_cv_to_sapien_pose",
    # Extraction
    "process_pt_file",
    "extract_params",
]
