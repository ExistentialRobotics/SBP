import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to sys.path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from mani_skill.utils import common
from mshab.envs.make import make_env
from mshab.envs.planner import plan_data_from_file
from mshab.utils.config import parse_cfg

from policy.utils.dataset import build_object_map, get_object_labels_batch, get_episode_ids_batch
from policy.utils.eval import _available_episode_names, _flatten_obs


def _load_train_module():
    from policy.train_bc import TrainConfig, get_mshab_train_cfg, build_agent
    return TrainConfig, get_mshab_train_cfg, build_agent


# --- Helpers --- #


def pad_ids(ids: List[int], num_envs: int) -> List[int]:
    if len(ids) == 0:
        return []
    return ids + [ids[-1]] * (num_envs - len(ids))


def collect_env_stats(envs, device: torch.device) -> List[dict]:
    returns = common.to_tensor(envs.return_queue, device=device).float()
    successes_once = common.to_tensor(envs.success_once_queue, device=device).float()
    successes_at_end = common.to_tensor(envs.success_at_end_queue, device=device).float()
    lengths = common.to_tensor(envs.length_queue, device=device).float()

    records = []
    num_episodes = returns.numel()
    if num_episodes == 0:
        return []

    for i in range(num_episodes):
        records.append(
            dict(
                rps=returns[i].item() / envs.max_episode_steps,
                succ_once=successes_once[i].item(),
                succ_end=successes_at_end[i].item(),
                length=lengths[i].item(),
            )
        )
    envs.reset_queues()
    return records


# --- Main --- #


def main():
    config_path = sys.argv[1]
    raw_cfg = parse_cfg(default_cfg_path=config_path)
    condition_map = OmegaConf.select(raw_cfg, "condition_map", default=True)
    TrainConfig, get_mshab_train_cfg, build_agent = _load_train_module()
    cfg: TrainConfig = get_mshab_train_cfg(raw_cfg)

    assert cfg.ckpt_path is not None, "ckpt_path must be provided (e.g. ckpt_path=data/models/final_ckpt.pt)"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Resolve paths --- #
    plan_fp = Path(cfg.eval_env.task_plan_fp).expanduser()
    spawn_fp = Path(cfg.eval_env.spawn_data_fp).expanduser()
    ckpt_path = Path(cfg.ckpt_path)

    assert plan_fp.exists(), f"Plan file not found: {plan_fp}"
    assert spawn_fp.exists(), f"Spawn data not found: {spawn_fp}"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # --- Plans metadata --- #
    plan_data = plan_data_from_file(plan_fp)
    all_plan_count = len(plan_data.plans)
    print(f"Plan file: {plan_fp.name}, Total plans: {all_plan_count}")

    # --- Filter plans by build_config_prefix --- #
    build_config_prefix = getattr(cfg, "build_config_prefix", None)
    if build_config_prefix is not None:
        plan_data.plans = [
            p for p in plan_data.plans
            if p.build_config_name.startswith(build_config_prefix)
        ]
        print(f"After build_config filter '{build_config_prefix}': {len(plan_data.plans)} plans")

    # --- Filter plans by map availability --- #
    map_dir = getattr(cfg.algo, "map_dir", None)
    if map_dir is not None:
        available = _available_episode_names(map_dir)
        before = len(plan_data.plans)
        plan_data.plans = [
            p for p in plan_data.plans
            if Path(p.init_config_name).stem in available
        ]
        print(f"After map filter: {len(plan_data.plans)}/{before} plans have maps in {map_dir}")

    assert len(plan_data.plans) > 0, "No plans remaining after filtering"

    # --- Create eval environment with filtered plans --- #
    cfg.eval_env.env_kwargs["task_plans"] = plan_data.plans
    cfg.eval_env.env_kwargs["scene_builder_cls"] = plan_data.dataset
    cfg.eval_env.task_plan_fp = None  # skip re-parsing in make_env
    env = make_env(cfg.eval_env, video_path=None)

    # --- Build label map --- #
    uid2lbl = build_object_map(str(plan_fp), cfg.algo.text_input)

    # --- Sample obs for agent init --- #
    sample_obs, _ = env.reset(seed=cfg.seed)

    # --- Build agent --- #
    build_kwargs = dict(cfg=cfg, sample_obs=sample_obs, single_act_shape=env.single_action_space.shape, device=device)
    if condition_map:
        build_kwargs["task_plan_path"] = str(plan_fp)
    agent = build_agent(**build_kwargs)

    uid_to_episode_idx = getattr(agent, 'uid_to_episode_idx', None)

    # --- Load checkpoint --- #
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt["agent"] if isinstance(ckpt, dict) and "agent" in ckpt else ckpt
    del ckpt
    from policy.agent.map_act import BCAgent
    BCAgent.strip_legacy_keys(state_dict)
    agent.load_state_dict(state_dict)
    del state_dict
    torch.cuda.empty_cache()
    agent.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # --- Evaluate in batches --- #
    all_ids = list(range(len(plan_data.plans)))
    num_envs = cfg.eval_env.num_envs
    batch_results = []

    for start in tqdm(
        range(0, len(all_ids), num_envs),
        desc="Evaluating",
    ):
        chunk = all_ids[start : start + num_envs]
        padded_ids = pad_ids(chunk, num_envs)
        obs, _ = env.reset(options={
            "task_plan_idxs": torch.tensor(padded_ids, dtype=torch.long, device=device)
        })

        # Get subtask labels for the current batch
        plan0 = env.unwrapped.task_plan[0]
        subtask_uids = plan0.composite_subtask_uids
        subtask_labels = get_object_labels_batch(uid2lbl, subtask_uids).to(device)
        assert (subtask_labels >= 0).all(), (
            f"Found unmapped UIDs in eval: "
            f"{[u for u, l in zip(subtask_uids, subtask_labels) if l < 0]}"
        )

        # Build episode_ids for map agent
        if uid_to_episode_idx is not None:
            episode_ids = get_episode_ids_batch(uid_to_episode_idx, subtask_uids).to(device)

        # Run full episode rollout
        max_steps = env.max_episode_steps
        for _ in range(max_steps):
            agent_obs = _flatten_obs(obs, device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                if uid_to_episode_idx is not None:
                    action = agent(agent_obs, subtask_labels, episode_ids=episode_ids)
                else:
                    action = agent(agent_obs, subtask_labels)
            obs, _, _, _, _ = env.step(action[:, 0, :])

        # Collect stats (only for unpadded episodes)
        batch_recs = collect_env_stats(env, device)[: len(chunk)]
        if batch_recs:
            batch_results.append((len(chunk), batch_recs))

    # --- Aggregate results --- #
    total_n_episodes = sum(n for n, _ in batch_results)
    if total_n_episodes == 0:
        print("No episodes collected. Exiting.")
        env.close()
        return

    weighted_sums: Dict[str, float] = defaultdict(float)
    for n_chunk, recs_chunk in batch_results:
        for key in recs_chunk[0].keys():
            batch_mean = np.mean([rec[key] for rec in recs_chunk])
            weighted_sums[key] += batch_mean * n_chunk

    summary = {
        "seed": cfg.seed,
        "ckpt": str(ckpt_path),
        "plan_file": plan_fp.name,
        "n_episodes": int(total_n_episodes),
        "return_per_step": float(weighted_sums["rps"] / total_n_episodes),
        "success_once": float(weighted_sums["succ_once"] / total_n_episodes),
        "success_at_end": float(weighted_sums["succ_end"] / total_n_episodes),
        "timestamp_utc": datetime.utcnow().isoformat(),
    }

    # --- Save result --- #
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    plan_tag = plan_fp.stem
    config_tag = f"_{build_config_prefix}" if build_config_prefix else ""
    out_json = out_dir / f"eval_{plan_tag}{config_tag}_seed{cfg.seed}_{timestamp}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {out_json}")
    print(json.dumps(summary, indent=2))

    env.close()


if __name__ == "__main__":
    main()
