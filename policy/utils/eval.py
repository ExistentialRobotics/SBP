from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import torch

from mshab.utils.array import to_tensor
from mani_skill.utils import common
from mshab.utils.logger import Logger

from policy.utils.dataset import get_object_labels_batch, get_episode_ids_batch


def _available_episode_names(map_dir: str) -> Set[str]:
    """Return set of episode names available in map_dir (both padded and unpadded)."""
    map_dir_path = Path(map_dir)
    names: Set[str] = set()
    for f in map_dir_path.glob("episode_*.pt"):
        if f.name.endswith(".dense.pt"):
            continue
        episode_name = f.stem  # e.g. "episode_0018"
        names.add(episode_name)
        # Also add unpadded form so "episode_18" matches "episode_0018"
        episode_num = episode_name.split("_", 1)[1]
        unpadded = f"episode_{int(episode_num)}"
        names.add(unpadded)
    return names



def _collect_stats(envs, device: torch.device) -> dict:
    stats = dict(
        return_per_step=(
            common.to_tensor(envs.return_queue, device=device).float().mean().item()
            / envs.max_episode_steps
        ),
        success_once=common.to_tensor(
            envs.success_once_queue, device=device
        )
        .float()
        .mean()
        .item(),
    )
    envs.reset_queues()
    return stats


def _pretty_print_stats(tag: str, stats: dict, logger: Logger, color: str):
    logger.print(
        f"{tag:<14}| Return: {stats['return_per_step']:.2f} | "
        f"Success_once: {stats['success_once']:.2f}",
        color=color,
        bold=True,
    )


def _flatten_obs(
    obs_raw: Dict[str, np.ndarray | torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    flat = {"state": to_tensor(obs_raw["state"], device=device)}

    px = obs_raw.get("pixels", {})
    for k in ("fetch_hand_rgb", "fetch_head_rgb"):
        if k in px:
            flat[k] = to_tensor(px[k], device=device)
        elif k in obs_raw:
            flat[k] = to_tensor(obs_raw[k], device=device)

    return flat


def run_eval_episode(
    eval_envs,
    eval_obs,
    agent,
    uid_to_label_map: Dict,
    device,
    uid2episode_id: Optional[Dict[str, int]] = None,
):
    """Runs one episode of evaluation."""
    max_steps = eval_envs.max_episode_steps

    plan0 = eval_envs.unwrapped.task_plan[0]
    subtask_labels = get_object_labels_batch(
        uid_to_label_map, plan0.composite_subtask_uids
    ).to(device)

    episode_ids = None
    if uid2episode_id is not None:
        episode_ids = get_episode_ids_batch(
            uid2episode_id, plan0.composite_subtask_uids
        ).to(device)

    for _ in range(max_steps):
        agent_obs = _flatten_obs(eval_obs, device)

        with torch.no_grad():
            if episode_ids is not None:
                action = agent(agent_obs, subtask_labels, episode_ids=episode_ids)
            else:
                action = agent(agent_obs, subtask_labels)

        eval_obs, _, _, _, _ = eval_envs.step(action[:, 0, :])

    return _collect_stats(eval_envs, device)
