import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Set, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm

from mshab.utils.array import to_tensor
from mshab.utils.dataset import ClosableDataset


def recursive_h5py_to_numpy(h5py_obs, slice=None):
    if isinstance(h5py_obs, (h5py.Group, dict)):
        return {k: recursive_h5py_to_numpy(h5py_obs[k], slice) for k in h5py_obs.keys()}
    if isinstance(h5py_obs, list):
        return [recursive_h5py_to_numpy(x, slice) for x in h5py_obs]
    if isinstance(h5py_obs, tuple):
        return tuple(recursive_h5py_to_numpy(x, slice) for x in h5py_obs)
    if slice is not None:
        return h5py_obs[slice]
    return h5py_obs[:]


class DPDataset(ClosableDataset):
    def __init__(
        self,
        data_path,
        obs_horizon: int,
        pred_horizon: int,
        control_mode: str,
        trajs_per_obj="all",
        max_image_cache_size: Union[int, str] = 0,
        truncate_trajectories_at_success: bool = True,
        allowed_uids: Optional[Set[str]] = None,
    ):
        data_path = Path(data_path)
        if data_path.is_dir():
            h5_fps = [
                data_path / fp for fp in os.listdir(data_path) if fp.endswith(".h5")
            ]
        else:
            h5_fps = [data_path]

        trajectories = dict(actions=[], observations=[], subtask_uids=[])
        num_cached = 0
        self.h5_files: List[h5py.File] = []
        for fp_num, fp in enumerate(h5_fps):
            json_fp = fp.with_suffix(".json")
            with open(json_fp, "rb") as json_f:
                json_file = json.load(json_f)

            f = h5py.File(fp, "r")
            num_uncached_this_file = 0

            if trajs_per_obj == "all":
                keys = list(f.keys())
            else:
                keys = random.sample(list(f.keys()), k=trajs_per_obj)

            for k in tqdm(keys, desc=f"hf file {fp_num}"):
                ep_num = int(k.replace("traj_", ""))
                subtask_uid = json_file["episodes"][ep_num]["subtask_uid"]

                if allowed_uids is not None and subtask_uid not in allowed_uids:
                    continue

                obs, act = f[k]["obs"], f[k]["actions"][:]

                if truncate_trajectories_at_success:
                    success: List[bool] = f[k]["success"][:].tolist()
                    if True in success:
                        success_cutoff = min(success.index(True) + 10, len(success))
                    else:
                        success_cutoff = len(act)
                    del success
                else:
                    success_cutoff = len(act)

                state_obs_list = [
                    *recursive_h5py_to_numpy(
                        obs["agent"], slice=slice(success_cutoff + 1)
                    ).values(),
                    *recursive_h5py_to_numpy(
                        obs["extra"], slice=slice(success_cutoff + 1)
                    ).values(),
                ]
                state_obs_list = [
                    x[:, None] if len(x.shape) == 1 else x for x in state_obs_list
                ]
                state_obs = torch.from_numpy(np.concatenate(state_obs_list, axis=1))
                act = torch.from_numpy(act[:success_cutoff])

                pixel_obs = dict(
                    fetch_head_rgb=obs["sensor_data"]["fetch_head"]["rgb"],
                    fetch_head_depth=obs["sensor_data"]["fetch_head"]["depth"],
                    fetch_hand_rgb=obs["sensor_data"]["fetch_hand"]["rgb"],
                    fetch_hand_depth=obs["sensor_data"]["fetch_hand"]["depth"],
                )
                if (
                    max_image_cache_size == "all"
                    or len(act) <= max_image_cache_size - num_cached
                ):
                    pixel_obs = to_tensor(
                        recursive_h5py_to_numpy(
                            pixel_obs, slice=slice(success_cutoff + 1)
                        )
                    )
                    num_cached += len(act)
                else:
                    num_uncached_this_file += len(act)

                # Camera extrinsics
                cam_pose_obs = to_tensor(
                    recursive_h5py_to_numpy(
                        dict(
                            fetch_head_pose=obs["sensor_param"]["fetch_head"][
                                "extrinsic_cv"
                            ],
                            fetch_hand_pose=obs["sensor_param"]["fetch_hand"][
                                "extrinsic_cv"
                            ],
                        ),
                        slice=slice(success_cutoff + 1),
                    ),
                    dtype=torch.float,
                )

                pixel_obs.update(**cam_pose_obs)

                trajectories["actions"].append(act)
                trajectories["observations"].append(dict(state=state_obs, **pixel_obs))
                trajectories["subtask_uids"].append(subtask_uid)

            if num_uncached_this_file == 0:
                f.close()
            else:
                self.h5_files.append(f)

        # Pre-compute all possible (traj_idx, start, end) tuples (Diffusion Policy style)
        if (
            "delta_pos" in control_mode
            or control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,)
            )
        else:
            raise NotImplementedError(f"Control Mode {control_mode} not supported")
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["observations"][traj_idx]["state"].shape[0] - 1
            total_transitions += L

            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            self.slices += [
                (
                    trajectories["subtask_uids"][traj_idx],
                    traj_idx,
                    start,
                    start + pred_horizon,
                )
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    def __getitem__(self, index):
        subtask_uid, traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                max(0, start) : start + self.obs_horizon
            ]
            if len(obs_seq[k].shape) == 4:
                obs_seq[k] = to_tensor(obs_seq[k])
            if start < 0:
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)

        act_seq = self.trajectories["actions"][traj_idx][max(0, start) : end]
        if start < 0:
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:
            gripper_action = act_seq[-1, -1]
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[0] == self.pred_horizon
        )
        return {
            "observations": obs_seq,
            "actions": act_seq,
            "subtask_uid": subtask_uid,
            "traj_idx": traj_idx,
        }

    def __len__(self):
        return len(self.slices)

    def close(self):
        for h5_file in self.h5_files:
            h5_file.close()


def build_object_map(json_file_path: str, object_names: List[str]) -> Dict[str, torch.Tensor]:
    """
    Build a label map from subtask_uid to object label.
    Returns an empty dict if no JSON file is found.
    """
    if not os.path.exists(json_file_path):
        print(f"[Warning] subtask map JSON not found: {json_file_path}")
        return {}

    with open(json_file_path, "r") as f:
        data = json.load(f)

    uid_to_label_map = {}
    for plan in data.get("plans", []):
        for subtask in plan.get("subtasks", []):
            subtask_uid = subtask["uid"]
            obj_id = subtask["obj_id"]

            found_label = None
            for i, obj_name in enumerate(object_names):
                if obj_name in obj_id:
                    found_label = i
                    break

            if found_label is None:
                raise ValueError(f"Unsupported object_id: {obj_id}")

            uid_to_label_map[subtask_uid] = torch.tensor(found_label, dtype=torch.long)

    return uid_to_label_map


def get_object_labels_batch(
    object_map: Dict[str, torch.Tensor], uids: List[str]
) -> torch.Tensor:
    """
    Return a tensor of labels for a batch of uids.
    If a uid is not found in the map, label = -1.
    """
    labels = []
    for uid in uids:
        if uid not in object_map:
            labels.append(torch.tensor(-1, dtype=torch.long))
        else:
            labels.append(object_map[uid])
    return torch.stack(labels, dim=0)


def get_episode_ids_batch(
    uid_to_episode_idx: Dict[str, int],
    subtask_uids: List[str],
) -> torch.Tensor:
    """Map a batch of subtask_uid strings to episode indices.

    Raises KeyError if any uid is not found in the mapping.
    """
    missing = [uid for uid in subtask_uids if uid not in uid_to_episode_idx]
    if missing:
        raise KeyError(f"UIDs not found in uid_to_episode_idx: {missing}")
    return torch.tensor(
        [uid_to_episode_idx[uid] for uid in subtask_uids], dtype=torch.long
    )


