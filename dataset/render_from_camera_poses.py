"""
Render RGB-D dataset from camera poses to HDF5 format.

This module provides the core functionality for generating HDF5 datasets
from ManiSkill environments using pre-defined camera poses.

Input:
    - camera_poses.npy: Dict of {key: (N, 7)} position + quaternion (wxyz)
      OR extrinsics.npy: Dict of {key: (N, 4, 4)} world-to-camera matrices
    - ManiSkill environment configuration

Output:
    - episode_XXXXXXXX.hdf5 with:
        - rgb: (N, H, W, 3) uint8
        - depth: (N, H, W) uint16 (mm)
        - poses: (N, 4, 4) float32 (world-to-cam, OpenCV)
        - intrinsics: (4,) float32 [fx, fy, cx, cy]
        - seg_instance_id: (N, H, W) int16

Usage:
    python dataset/render_from_camera_poses.py \\
        --poses_file dataset/camera_params/camera_poses.npy \\
        --output_dir data/hdf5 \\
        --task set_table --subtask pick \\
        --build_config_idx 41 --task_plan_idx 0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import transforms3d as t3d

# Add repo root to path for imports
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "third_party" / "mshab"))
sys.path.insert(0, str(repo_root / "third_party" / "ManiSkill"))


# --------------------------------------------------------------------------- #
#  Constants                                                                  #
# --------------------------------------------------------------------------- #

# Rotation matrix from camera (SAPIEN/OpenGL) to optical frame (OpenCV)
# Used for coordinate conversion between SAPIEN and OpenCV conventions
ORC_MAT = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float32
)


# --------------------------------------------------------------------------- #
#  HDF5 Data Saver                                                            #
# --------------------------------------------------------------------------- #


class HDF5DataSaver:
    """Handles buffered HDF5 data saving for efficient I/O.

    This class buffers frames in memory and periodically flushes them to disk
    to reduce I/O overhead.

    Attributes:
        output_path: Path to the output HDF5 file.
        flush_threshold: Number of frames to buffer before flushing.
        instance_id_to_name: Mapping from instance ID to object name.
    """

    def __init__(
        self,
        output_path: str,
        flush_threshold: int = 100,
        instance_id_to_name: Optional[Dict[int, str]] = None,
    ):
        self.output_path = output_path
        self.flush_threshold = flush_threshold
        self.output_file: Optional[h5py.File] = None
        self.data_buffers: Dict[str, List[np.ndarray]] = {}
        self.buffer_size = 0
        self.instance_id_to_name = instance_id_to_name or {}

    def open(self):
        """Open the HDF5 file for writing."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.output_file = h5py.File(self.output_path, "w")
        self.data_buffers = {}
        self.buffer_size = 0

    def close(self):
        """Flush remaining data and close the HDF5 file."""
        if self.output_file:
            self.flush_buffer()
            if self.instance_id_to_name:
                self.output_file.attrs["instance_id_to_name"] = json.dumps(
                    {str(k): v for k, v in self.instance_id_to_name.items()}
                )
            self.output_file.close()
            self.output_file = None

    def save_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        seg_instance_id: np.ndarray,
    ):
        """Save a single frame to the buffer.

        Args:
            rgb: RGB image (H, W, 3) uint8.
            depth: Depth image (H, W) float32 in meters or uint16 in mm.
            pose: 4x4 extrinsic matrix (world-to-camera, OpenCV).
            seg_instance_id: Instance segmentation (H, W) int16.
        """
        # RGB: Keep as uint8
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        # Depth: Convert to uint16 (mm) if needed
        if depth.dtype in (np.float32, np.float64):
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth = (depth * 1000).clip(0, 65535).astype(np.uint16)
        elif depth.dtype == np.int16:
            depth = depth.astype(np.uint16)
        elif depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)

        # Pose: Ensure 4x4 matrix, float32
        if pose.shape == (3, 4):
            pose_4x4 = np.eye(4, dtype=np.float32)
            pose_4x4[:3, :] = pose
            pose = pose_4x4
        if pose.dtype != np.float32:
            pose = pose.astype(np.float32)

        # Segmentation: Ensure int16
        if seg_instance_id.dtype != np.int16:
            seg_instance_id = seg_instance_id.astype(np.int16)

        # Add to buffers
        data_map = {
            "rgb": rgb,
            "depth": depth,
            "poses": pose,
            "seg_instance_id": seg_instance_id,
        }

        for key, data in data_map.items():
            if key not in self.data_buffers:
                self.data_buffers[key] = []
            self.data_buffers[key].append(data)

        self.buffer_size += 1

        if self.buffer_size >= self.flush_threshold:
            self.flush_buffer()

    def flush_buffer(self):
        """Flush buffered data to HDF5 file."""
        if not self.output_file or not self.data_buffers:
            return

        for key, data_list in self.data_buffers.items():
            if not data_list:
                continue

            data_np = np.stack(data_list)

            if key in self.output_file:
                dset = self.output_file[key]
                dset.resize(dset.shape[0] + data_np.shape[0], axis=0)
                dset[-data_np.shape[0] :] = data_np
            else:
                maxshape = (None,) + data_np.shape[1:]
                compression = "lzf"
                self.output_file.create_dataset(
                    key,
                    data=data_np,
                    maxshape=maxshape,
                    chunks=(1,) + data_np.shape[1:],
                    compression=compression,
                )

            data_list.clear()

        self.buffer_size = 0

    def save_intrinsics(self, intrinsics: np.ndarray):
        """Save camera intrinsics.

        Args:
            intrinsics: (3, 3) intrinsic matrix or (4,) [fx, fy, cx, cy].
        """
        if self.output_file:
            if intrinsics.shape == (3, 3):
                fx = intrinsics[0, 0]
                fy = intrinsics[1, 1]
                cx = intrinsics[0, 2]
                cy = intrinsics[1, 2]
                intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)
            elif intrinsics.shape != (4,):
                raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")

            self.output_file.create_dataset(
                "intrinsics", data=intrinsics.astype(np.float32)
            )
            self.output_file.attrs["depth_scale"] = 1000.0

    def update_instance_id_map(self, id_map: Dict[int, str]):
        """Update the instance ID to name mapping.

        Args:
            id_map: Mapping from instance ID to object name.
        """
        self.instance_id_to_name.update(id_map)


# --------------------------------------------------------------------------- #
#  Pose Loading and Conversion                                                #
# --------------------------------------------------------------------------- #


def load_camera_poses(poses_file: str) -> Tuple[Dict[str, np.ndarray], str]:
    """Load camera poses from file.

    Supports two formats:
    1. camera_poses.npy: Dict of {key: (N, 7)} - position (3) + quaternion (4, wxyz)
    2. extrinsics.npy: Dict of {key: (N, 4, 4)} - world-to-camera matrices
    3. Single array: (N, 7) or (N, 4, 4) wrapped as {"default": array}

    Args:
        poses_file: Path to the poses file.

    Returns:
        Tuple of (poses_dict, format_type) where format_type is "7d" or "4x4".
    """
    data = np.load(poses_file, allow_pickle=True)

    # Handle numpy scalar (dict saved with allow_pickle)
    if data.ndim == 0:
        poses_dict = data.item()
    else:
        # Single array, wrap in dict
        poses_dict = {"default": data}

    # Detect format from first trajectory
    first_key = next(iter(poses_dict))
    first_poses = poses_dict[first_key]

    if first_poses.ndim == 2 and first_poses.shape[-1] == 7:
        return poses_dict, "7d"
    elif first_poses.ndim == 3 and first_poses.shape[-2:] == (4, 4):
        return poses_dict, "4x4"
    elif first_poses.ndim == 2 and first_poses.shape[-2:] == (4, 4):
        # Single pose, wrap
        for key in poses_dict:
            poses_dict[key] = poses_dict[key][np.newaxis, ...]
        return poses_dict, "4x4"
    else:
        raise ValueError(f"Unknown pose format: shape {first_poses.shape}")


def poses_7d_to_extrinsic_cv(poses_7d: np.ndarray) -> np.ndarray:
    """Convert 7D poses to 4x4 extrinsic matrices (OpenCV convention).

    Args:
        poses_7d: (N, 7) array with position (3) + quaternion (4, wxyz).

    Returns:
        (N, 4, 4) world-to-camera extrinsic matrices (OpenCV convention).
    """
    N = poses_7d.shape[0]
    extrinsics = np.zeros((N, 4, 4), dtype=np.float32)

    for i in range(N):
        pos = poses_7d[i, :3]
        quat = poses_7d[i, 3:]  # wxyz

        # Quaternion to rotation matrix (camera-to-world in SAPIEN/OpenGL)
        R_c2w_gl = t3d.quaternions.quat2mat(quat)

        # Convert from OpenGL to OpenCV convention using ORC_MAT
        R_w2c_cv = ORC_MAT @ R_c2w_gl.T
        t_w2c_cv = -R_w2c_cv @ pos

        # Build world-to-camera extrinsic (OpenCV)
        extrinsics[i, :3, :3] = R_w2c_cv
        extrinsics[i, :3, 3] = t_w2c_cv
        extrinsics[i, 3, 3] = 1.0

    return extrinsics


def extrinsic_cv_to_sapien_pose(extrinsic_cv: np.ndarray) -> np.ndarray:
    """Convert 4x4 extrinsic to SAPIEN 7D pose for rendering.

    Args:
        extrinsic_cv: 4x4 world-to-camera matrix (OpenCV convention).

    Returns:
        (7,) array with position (3) + quaternion (4, wxyz) for SAPIEN.
    """
    R_w2c = extrinsic_cv[:3, :3]
    t_w2c = extrinsic_cv[:3, 3]

    # Convert to camera-to-world in SAPIEN/OpenGL convention
    R_c2w_gl = R_w2c.T @ ORC_MAT
    t_c2w_gl = -R_w2c.T @ t_w2c

    # Convert rotation matrix to quaternion (wxyz)
    quat = t3d.quaternions.mat2quat(R_c2w_gl)

    return np.concatenate([t_c2w_gl, quat]).astype(np.float32)


def sapien_pose_to_extrinsic_cv(pose_7d: np.ndarray) -> np.ndarray:
    """Convert SAPIEN 7D pose to 4x4 extrinsic matrix (OpenCV convention).

    Args:
        pose_7d: (7,) array with position (3) + quaternion (4, wxyz).

    Returns:
        4x4 world-to-camera extrinsic matrix (OpenCV convention).
    """
    pos = pose_7d[:3]
    quat = pose_7d[3:]  # wxyz

    # Quaternion to rotation matrix (camera-to-world in SAPIEN/OpenGL)
    R_c2w_gl = t3d.quaternions.quat2mat(quat)

    # Convert from OpenGL to OpenCV convention using ORC_MAT
    # R_c2w_cv = R_c2w_gl @ ORC_MAT.T
    # R_w2c_cv = R_c2w_cv.T = ORC_MAT @ R_c2w_gl.T
    R_w2c_cv = ORC_MAT @ R_c2w_gl.T

    # Translation: world-to-camera
    t_w2c_cv = -R_w2c_cv @ pos

    # Build world-to-camera extrinsic (OpenCV)
    extrinsic_cv = np.eye(4, dtype=np.float32)
    extrinsic_cv[:3, :3] = R_w2c_cv
    extrinsic_cv[:3, 3] = t_w2c_cv

    return extrinsic_cv


# --------------------------------------------------------------------------- #
#  Build Config Utilities                                                     #
# --------------------------------------------------------------------------- #


def get_build_config_name(build_config_idx: int, include_staging: bool = True) -> str:
    """Get build_config_name from build_config_idx.

    Args:
        build_config_idx: Index from scene_configs.json.
        include_staging: Whether staging scenes are included (usually True).

    Returns:
        Build config name (e.g., "v3_sc1_staging_13.scene_instance.json").
    """
    import json
    import mani_skill

    # ManiSkill metadata directory
    metadata_dir = Path(mani_skill.__file__).parent / "utils/scene_builder/replicacad/metadata"
    scene_configs_path = metadata_dir / "scene_configs.json"

    with open(scene_configs_path, "r") as f:
        data = json.load(f)

    build_configs = data["scenes"]
    if include_staging:
        build_configs += data.get("staging_scenes", [])

    if build_config_idx >= len(build_configs):
        raise IndexError(
            f"build_config_idx {build_config_idx} out of range (max: {len(build_configs) - 1})"
        )

    return build_configs[build_config_idx]


def get_filtered_task_plans(config, build_config_idx: int, unique_only: bool = True) -> list:
    """Get task plans filtered by build_config_idx.

    Args:
        config: OmegaConf configuration with task, subtask, split, obj fields.
        build_config_idx: Build config index.
        unique_only: If True, return only first task_plan per unique init_config.

    Returns:
        List of task plans for the specific build_config.
    """
    from mshab.envs.planner import plan_data_from_file

    # Get build_config_name
    build_config_name = get_build_config_name(build_config_idx)

    # Load all task plans
    ms_asset_dir = os.environ.get("MS_ASSET_DIR", os.path.join(os.path.expanduser("~"), ".maniskill"))
    ms_data = os.path.join(ms_asset_dir, "data")

    task_plan_base = os.path.join(
        ms_data,
        f"scene_datasets/replica_cad_dataset/rearrange/task_plans/{config.task}/{config.subtask}/{config.split}",
    )
    task_plan_fp = os.path.join(task_plan_base, f"{config.obj}.json")

    if not os.path.exists(task_plan_fp):
        task_plan_fp = os.path.join(task_plan_base, "all.json")

    if not os.path.exists(task_plan_fp):
        raise FileNotFoundError(f"Task plan file not found: {task_plan_fp}")

    plan_data = plan_data_from_file(task_plan_fp)

    # Filter by build_config_name
    filtered_plans = [p for p in plan_data.plans if p.build_config_name == build_config_name]

    # Filter to unique init_configs only
    if unique_only:
        seen_init_configs = set()
        unique_plans = []
        for plan in filtered_plans:
            if plan.init_config_name not in seen_init_configs:
                unique_plans.append(plan)
                seen_init_configs.add(plan.init_config_name)
        filtered_plans = unique_plans

    return filtered_plans


# --------------------------------------------------------------------------- #
#  Episode Name Utilities                                                     #
# --------------------------------------------------------------------------- #


def get_episode_name(config, task_plan_idx: int) -> str:
    """Get episode name from task plan's init_config_name.

    This function extracts the episode name from filtered task plans
    (filtered by build_config_idx).

    Args:
        config: OmegaConf configuration with task, subtask, split, obj, build_config_idx fields.
        task_plan_idx: Task plan index (local index within filtered plans).

    Returns:
        Episode filename (e.g., "episode_0100.hdf5").
    """
    # Get filtered task plans
    filtered_plans = get_filtered_task_plans(config, config.build_config_idx)

    if task_plan_idx >= len(filtered_plans):
        raise IndexError(
            f"task_plan_idx {task_plan_idx} out of range (max: {len(filtered_plans) - 1})"
        )

    # Extract episode name from init_config_name
    # e.g., "train/prepare_groceries/episode_100.json" -> "episode_0100"
    init_config_name = filtered_plans[task_plan_idx].init_config_name
    episode_stem = Path(init_config_name).stem  # "episode_100"

    # Extract number and format to 4 digits
    # episode_100 -> episode_0100
    parts = episode_stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        episode_num = int(parts[1])
        episode_name = f"{parts[0]}_{episode_num:04d}"  # 4-digit format
    else:
        episode_name = episode_stem

    return f"{episode_name}.hdf5"


def get_max_task_plan_count(config) -> int:
    """Get total number of task plans for the given config filtered by build_config_idx.

    Args:
        config: OmegaConf configuration with task, subtask, split, obj, build_config_idx fields.

    Returns:
        Number of task plans available for the specific build_config.
    """
    filtered_plans = get_filtered_task_plans(config, config.build_config_idx)
    return len(filtered_plans)


# --------------------------------------------------------------------------- #
#  ManiSkill Environment Setup                                                #
# --------------------------------------------------------------------------- #


def setup_maniskill_env(config):
    """Setup ManiSkill environment based on configuration.

    This function imports ManiSkill modules and creates the environment class.
    Import is done here to avoid loading heavy dependencies unnecessarily.

    Args:
        config: OmegaConf configuration object with task, subtask, etc.

    Returns:
        Tuple of (DatasetGenEnv class, plan_data_from_file function).
    """
    import gymnasium as gym
    import sapien
    import torch
    from sapien import physx

    from mani_skill.agents.base_agent import Keyframe
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.building import actors
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.scene_builder.replicacad.rearrange.scene_builder import (
        ReplicaCADRearrangeSceneBuilder,
    )
    from mani_skill.utils.structs.actor import Actor
    from mani_skill.utils.structs.pose import Pose
    from mshab.envs.planner import plan_data_from_file
    from mshab.envs.sequential_task import SequentialTaskEnv

    # Fake classes to avoid loading actual robot
    class FakeLink:
        def __init__(self, num_envs: int, device: str):
            self.num_envs = num_envs
            self.device = device
            self.pose = Pose.create(sapien.Pose())
            self.pose.raw_pose = self.pose.raw_pose.expand((self.num_envs, 7))
            self.pose.raw_pose = self.pose.raw_pose.to(self.device)
            self.linear_velocity = torch.zeros((self.num_envs, 3), device=self.device)

    class FakeRobot:
        def __init__(self, num_envs: int, device: str):
            self.num_envs = num_envs
            self.device = device
            self.qpos = torch.zeros((self.num_envs, 15), device=self.device)
            self.qvel = torch.zeros((self.num_envs, 15), device=self.device)

        def set_pose(self, pose):
            pass

        def set_qpos(self, qpos):
            pass

        def get_net_contact_forces(self, link_names):
            f = torch.zeros((self.num_envs, len(link_names), 3))
            if physx.is_gpu_enabled():
                f = f.cuda()
            return f

        def hide_visual(self):
            pass

    class FakeController:
        def reset(self):
            pass

    class FakeAgent:
        def __init__(self, num_envs: int, device: str):
            self.num_envs = num_envs
            self.device = device
            self._sensor_configs = []
            self.robot = FakeRobot(self.num_envs, self.device)
            self.keyframes = dict(rest=Keyframe(pose=sapien.Pose(), qpos=np.zeros(15)))
            self.controller = FakeController()
            self.tcp = FakeLink(self.num_envs, self.device)
            self.base_link = FakeLink(self.num_envs, self.device)
            self.single_action_space = gym.spaces.Box(0, 1, dtype=np.float32)
            self.action_space = self.single_action_space
            if self.num_envs > 1:
                self.action_space = gym.vector.utils.batch_space(
                    self.action_space, self.num_envs
                )

        def get_controller_state(self):
            return {}

        def set_controller_state(self, state):
            pass

        def reset(self, init_qpos=None):
            pass

        def is_grasping(self, objects, min_force=0.5, max_angle=85):
            return torch.tensor([False] * self.num_envs, device=self.device)

        def is_static(self, threshold=0.2, base_threshold=0.05):
            return torch.tensor([False] * self.num_envs, device=self.device)

        @property
        def tcp_pose(self):
            pose = Pose.create(sapien.Pose())
            pose.raw_pose = pose.raw_pose.expand((self.num_envs, 7))
            pose.raw_pose = pose.raw_pose.to(self.device)
            return pose

        def get_proprioception(self):
            return dict(
                qpos=torch.zeros((self.num_envs, 15), device=self.device),
                qvel=torch.zeros((self.num_envs, 15), device=self.device),
            )

        def before_simulation_step(self):
            pass

        def set_action(self, action):
            pass

    # Create environment class with unique name
    env_name = f"DatasetGenEnv_{config.task}_{config.subtask}"

    @register_env(env_name, max_episode_steps=200)
    class DatasetGenEnv(SequentialTaskEnv):
        def __init__(
            self,
            *args,
            load_agent=False,
            hide_episode_objects=False,
            image_width=512,
            image_height=512,
            spawn_data_fp=None,
            **kwargs,
        ):
            self.load_agent = load_agent
            self.hide_episode_objects = hide_episode_objects
            self.camera_mount: Actor = None
            self.camera_mount_offset = 0.01
            self.image_width = image_width
            self.image_height = image_height
            self.spawn_data_fp = spawn_data_fp
            super().__init__(*args, **kwargs)
            self.set_episode_objects()

        def set_episode_objects(self):
            if self.hide_episode_objects:
                scene_builder: ReplicaCADRearrangeSceneBuilder = self.scene_builder
                for env_ycb_objects in scene_builder.ycb_objs_per_env:
                    for obj_name, objs in env_ycb_objects.items():
                        for obj in objs:
                            scene_builder.hide_actor(obj)

        def step(self, action):
            if self.load_agent:
                super().step(action)
                self.set_episode_objects()
                return
            self.set_episode_objects()

        def _load_agent(self, options: dict):
            if self.load_agent:
                super()._load_agent(options)
            else:
                self.agent = FakeAgent(self.num_envs, self.device)
            self.camera_mount = actors.build_sphere(
                self.scene,
                radius=0.001,
                color=np.array([0.0, 0.0, 0.0, 1.0]),
                name="sphere",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(),
            )

        def _after_reconfigure(self, options):
            if self.load_agent:
                super()._after_reconfigure(options)
                return
            self.force_articulation_link_ids = []
            self.robot_cumulative_force = torch.zeros(self.num_envs, device=self.device)
            self.spawn_data = torch.load(self.spawn_data_fp, map_location=self.device)
            self.spawn_selection_idxs = [None] * self.num_envs

        def _initialize_episode(self, env_idx, options):
            with torch.device(self.device):
                self.robot_cumulative_force[env_idx] = 0
                if env_idx.numel() == self.num_envs:
                    self.task_plan_idxs = options.get("task_plan_idxs", None)
                if self.task_plan_idxs is None or env_idx.numel() < self.num_envs:
                    if self.task_plan_idxs is None:
                        self.task_plan_idxs = torch.zeros(self.num_envs, dtype=torch.int)
                    low = torch.zeros(env_idx.numel(), dtype=torch.int)
                    high = self.num_task_plans_per_bci[env_idx]
                    size = (env_idx.numel(),)
                    self.task_plan_idxs[env_idx] = (
                        torch.randint(2**63 - 1, size=size) % (high - low).int()
                        + low.int()
                    ).int()
                else:
                    self.task_plan_idxs = self.task_plan_idxs.int()

                sampled_task_plans = [
                    self.build_config_idx_to_task_plans[bci][tpi]
                    for bci, tpi in zip(self.build_config_idxs, self.task_plan_idxs)
                ]

                if options.get("init_config_idxs") is not None:
                    if options["init_config_idxs"][0] == -1:
                        self.init_config_idxs = [
                            self.scene_builder.init_config_names_to_idxs[
                                tp.init_config_name
                            ]
                            for tp in sampled_task_plans
                        ]
                    else:
                        self.init_config_idxs = options["init_config_idxs"]
                else:
                    self.init_config_idxs = [
                        self.scene_builder.init_config_names_to_idxs[tp.init_config_name]
                        for tp in sampled_task_plans
                    ]

                super(SequentialTaskEnv, self)._initialize_episode(env_idx, options)

                self.process_task_plan(
                    env_idx,
                    sampled_subtask_lists=[tp.subtasks for tp in sampled_task_plans],
                )

                self.subtask_pointer[env_idx] = 0
                self.subtask_steps_left[env_idx] = self.task_cfgs[
                    self.task_plan[0].type
                ].horizon

                self.resting_qpos = torch.tensor(self.agent.keyframes["rest"].qpos[3:-2])

        @property
        def _default_human_render_camera_configs(self):
            robot_camera_pose = sapien_utils.look_at(
                [self.camera_mount_offset, 0, 0], ([1.0, 0.0, 0.0])
            )
            return CameraConfig(
                uid="render_camera",
                pose=robot_camera_pose,
                width=self.image_width,
                height=self.image_height,
                fov=1.75,
                near=0.01,
                far=10,
                mount=self.camera_mount,
                shader_pack="minimal",  # Reduce GPU memory usage
            )

        def render_at_poses(self, poses: np.ndarray = None) -> dict:
            """Render at given camera poses.

            Args:
                poses: (N, 7) array with position (3) + quaternion (4, wxyz).

            Returns:
                Dict with rgb, depth, segmentation, cam_pose, extrinsic_cv, intrinsic_cv.
            """
            camera = self.scene.human_render_cameras["render_camera"]
            if poses is not None:
                for i in range(len(poses)):
                    rot = t3d.quaternions.quat2mat(poses[i, 3:])
                    poses[i, :3] -= self.camera_mount_offset * rot[:, 0]
                self.camera_mount.set_pose(torch.from_numpy(poses).to(self.device))
            self.scene.update_render()
            camera.camera.take_picture()
            # Explicitly enable segmentation in get_obs
            obs = {k: v for k, v in camera.get_obs(
                rgb=True, depth=True, segmentation=True, position=False
            ).items()}

            if "position" in obs:
                obs["position"][..., 1] *= -1
                obs["position"][..., 2] *= -1
            obs["cam_pose"] = poses
            obs["extrinsic_cv"] = camera.camera.get_extrinsic_matrix()
            obs["intrinsic_cv"] = camera.camera.get_intrinsic_matrix()
            return obs

    return DatasetGenEnv, plan_data_from_file


# --------------------------------------------------------------------------- #
#  Rendering Functions                                                        #
# --------------------------------------------------------------------------- #


def render_episode(
    config,
    task_plan_idx: int,
    output_path: str,
    poses_dict: Dict[str, np.ndarray],
    pose_format: str,
    num_frames: Optional[int] = None,
    flush_threshold: int = 100,
    batch_size: int = 300,
) -> int:
    """Render a single episode from poses using batch parallel rendering.

    Args:
        config: OmegaConf configuration object.
        task_plan_idx: Task plan index.
        output_path: Output HDF5 file path.
        poses_dict: Dictionary of camera poses {key: (N, 7) or (N, 4, 4)}.
        pose_format: "7d" or "4x4".
        num_frames: Number of frames to sample (None=use all poses).
        flush_threshold: Buffer size before flushing to HDF5.
        batch_size: Number of poses to render simultaneously (ManiSkill batch env).

    Returns:
        Number of frames saved.
    """
    import torch
    from tqdm import tqdm

    # Setup environment
    DatasetGenEnv, plan_data_from_file = setup_maniskill_env(config)

    ms_asset_dir = os.environ.get("MS_ASSET_DIR", os.path.join(os.path.expanduser("~"), ".maniskill"))
    ms_data = os.path.join(ms_asset_dir, "data")

    task_plan_base = os.path.join(
        ms_data,
        f"scene_datasets/replica_cad_dataset/rearrange/task_plans/{config.task}/{config.subtask}/{config.split}",
    )
    task_plan_fp = os.path.join(task_plan_base, f"{config.obj}.json")

    if not os.path.exists(task_plan_fp):
        print(f"Warning: {task_plan_fp} not found. Trying 'all.json'.")
        config.obj = "all"
        task_plan_fp = os.path.join(task_plan_base, "all.json")

    spawn_data_fp = os.path.join(
        ms_data,
        f"scene_datasets/replica_cad_dataset/rearrange/spawn_data/{config.task}/{config.subtask}/{config.split}/spawn_data.pt",
    )

    if not os.path.exists(task_plan_fp):
        raise FileNotFoundError(f"Task plan file not found: {task_plan_fp}")

    print(f"Using Task Plan: {task_plan_fp}")

    plan_data = plan_data_from_file(task_plan_fp)
    task_plans = plan_data.plans

    env_kwargs = dict(
        spawn_data_fp=spawn_data_fp,
        task_plans=task_plans,
        scene_builder_cls=plan_data.dataset,
        require_build_configs_repeated_equally_across_envs=False,
    )

    # Collect all poses (7D SAPIEN format for rendering)
    # Note: Extrinsics will be obtained directly from ManiSkill after rendering
    all_poses_7d = []

    keys = sorted(poses_dict.keys())
    for key in keys:
        traj_poses = poses_dict[key]
        for t in range(traj_poses.shape[0]):
            if pose_format == "7d":
                pose_7d = traj_poses[t].astype(np.float32)
            else:
                # 4x4 extrinsic → 7D SAPIEN pose for rendering
                extrinsic_cv = traj_poses[t]
                pose_7d = extrinsic_cv_to_sapien_pose(extrinsic_cv)

            all_poses_7d.append(pose_7d)

    all_poses_7d = np.stack(all_poses_7d)  # (N, 7)
    n_total_poses = len(all_poses_7d)

    print(f"Total available poses: {n_total_poses}")

    # Sample frames if num_frames is specified
    if num_frames is not None and num_frames < n_total_poses:
        # Get episode number for seed from filtered task plans
        filtered_plans = get_filtered_task_plans(config, config.build_config_idx)
        episode_stem = Path(filtered_plans[task_plan_idx].init_config_name).stem
        parts = episode_stem.rsplit("_", 1)
        episode_num = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

        # Set seed based on episode number for reproducible but varied sampling
        np.random.seed(episode_num)

        # Random sampling instead of uniform
        indices = np.random.choice(n_total_poses, size=num_frames, replace=False)
        indices = np.sort(indices)  # Maintain temporal order

        all_poses_7d = all_poses_7d[indices]
        n_total_poses = num_frames
        print(f"Sampled {num_frames} frames randomly (seed={episode_num})")

    print(f"Total poses to render: {n_total_poses}")
    print(f"Batch size: {batch_size}")

    # Create environment with batch_size
    env = DatasetGenEnv(
        obs_mode="rgbd",
        render_mode="all",
        control_mode="pd_joint_delta_pos",
        sim_backend="gpu",
        num_envs=batch_size,
        image_width=config.image_width,
        image_height=config.image_height,
        **env_kwargs,
    )

    print(
        f"Resetting env with build_config_idx={config.build_config_idx}, "
        f"task_plan_idx={task_plan_idx}"
    )

    options = dict(
        reconfigure=True,
        build_config_idxs=[config.build_config_idx] * batch_size,
        task_plan_idxs=torch.tensor([task_plan_idx] * batch_size, dtype=torch.int32),
        init_config_idxs=[config.init_config_idx] * batch_size,
        spawn_selection_idxs=[config.spawn_selection_idx] * batch_size,
    )

    env.reset(seed=0, options=options)

    # Extract segmentation ID to name mapping
    seg_id_map = {}
    if hasattr(env, 'segmentation_id_map'):
        for seg_id, obj in env.segmentation_id_map.items():
            if hasattr(obj, 'name'):
                seg_id_map[int(seg_id)] = obj.name

    # Initialize HDF5 saver with instance_id_to_name mapping
    saver = HDF5DataSaver(
        output_path,
        flush_threshold=flush_threshold,
        instance_id_to_name=seg_id_map,
    )
    saver.open()

    # Save intrinsics on first frame
    intrinsics_saved = False
    frame_count = 0

    print("Starting batch rendering...")
    n_batches = (n_total_poses + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc="Batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_total_poses)
        actual_batch_size = batch_end - batch_start

        # Print frame progress
        print(f"Processing frames {batch_start + 1}-{batch_end} / {n_total_poses}")

        # Get poses for this batch
        batch_poses = all_poses_7d[batch_start:batch_end]  # (B, 7)

        # Pad if necessary (last batch may be smaller)
        if actual_batch_size < batch_size:
            pad_size = batch_size - actual_batch_size
            batch_poses = np.concatenate([
                batch_poses,
                np.tile(batch_poses[-1:], (pad_size, 1))
            ], axis=0)

        # Render batch
        obs = env.render_at_poses(batch_poses)

        # Get observations (GPU -> CPU)
        rgb_batch = obs["rgb"].cpu().numpy()  # (B, H, W, 3)
        depth_batch = obs["depth"].cpu().numpy()  # (B, H, W, 1)

        if "segmentation" in obs:
            seg_batch = obs["segmentation"].cpu().numpy()
            # Handle shape: (B, H, W, 1) -> (B, H, W)
            if seg_batch.ndim == 4:
                seg_batch = seg_batch.squeeze(-1)
        else:
            seg_batch = np.zeros(depth_batch.shape[:3], dtype=np.int16)

        # Save intrinsics once
        if not intrinsics_saved:
            intrinsic_cv = obs["intrinsic_cv"][0].cpu().numpy()
            saver.save_intrinsics(intrinsic_cv)
            intrinsics_saved = True

        # Save only actual frames (not padding)
        # Use ManiSkill's correct extrinsic matrices (OpenCV convention)
        extrinsic_cv_batch = obs["extrinsic_cv"][:actual_batch_size].cpu().numpy()
        for i in range(actual_batch_size):
            saver.save_frame(
                rgb_batch[i],
                depth_batch[i].squeeze(-1),
                extrinsic_cv_batch[i],
                seg_batch[i],
            )
            frame_count += 1

    saver.close()
    print(f"Saved {frame_count} frames to {output_path}")
    return frame_count


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Render HDF5 dataset from camera poses"
    )
    # Task configuration
    parser.add_argument(
        "--task", type=str, default="set_table", help="Task name (e.g., set_table)"
    )
    parser.add_argument(
        "--build_config_idx", type=int, default=37, help="Build config index"
    )
    parser.add_argument("--task_plan_idx", type=int, default=0, help="Task plan index")
    parser.add_argument(
        "--split", type=str, default="train", help="Data split (train/val/test)"
    )
    parser.add_argument("--obj", type=str, default="all", help="Object filter")
    parser.add_argument(
        "--init_config_idx",
        type=int,
        default=-1,
        help="Init config index (-1 = use default from task plan)",
    )
    parser.add_argument(
        "--spawn_selection_idx", type=int, default=18, help="Spawn selection index"
    )
    parser.add_argument("--image_height", type=int, default=512, help="Image height")
    parser.add_argument("--image_width", type=int, default=512, help="Image width")

    # Paths
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--poses_file",
        type=str,
        default="dataset/camera_params/camera_poses.npy",
        help="Path to camera poses file",
    )

    # Rendering options
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames to sample from each scene (None=use all poses)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=300,
        help="Number of poses to render simultaneously (ManiSkill batch env)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    from omegaconf import OmegaConf
    import random

    args = parse_args()

    # Create config from CLI arguments
    config = OmegaConf.create(
        {
            "task": args.task,
            "subtask": "pick",
            "build_config_idx": args.build_config_idx,
            "split": args.split,
            "obj": args.obj,
            "init_config_idx": args.init_config_idx,
            "spawn_selection_idx": args.spawn_selection_idx,
            "image_height": args.image_height,
            "image_width": args.image_width,
        }
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load poses
    if not os.path.exists(args.poses_file):
        print(f"Error: Poses file not found at {args.poses_file}")
        return

    poses_dict, pose_format = load_camera_poses(args.poses_file)
    total_poses = sum(len(p) for p in poses_dict.values())
    print(
        f"Loaded {len(poses_dict)} pose trajectories, {total_poses} total poses (format: {pose_format})"
    )
    print(f"Batch size: {args.batch_size}")
    if args.num_frames:
        print(f"Sampling {args.num_frames} frames")

    # Get episode name from task plan
    episode_name = get_episode_name(config, args.task_plan_idx)

    # Output path
    output_path = os.path.join(
        args.output_dir,
        config.task,
        episode_name,
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Skip if exists
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Skipping existing: {output_path}")
        return

    print(f"\n{'=' * 60}")
    print(f"Generating dataset for task_plan_idx={args.task_plan_idx}")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")

    render_episode(
        config=config,
        task_plan_idx=args.task_plan_idx,
        output_path=output_path,
        poses_dict=poses_dict,
        pose_format=pose_format,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
    )

    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
