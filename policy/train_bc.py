import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Add project root to sys.path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from dacite import from_dict
from omegaconf import OmegaConf
from torch.optim import Optimizer
from tqdm import tqdm

from policy.agent.map_act import BCAgent
from policy.utils.dataset import (
    DPDataset,
    build_object_map,
    get_object_labels_batch,
    get_episode_ids_batch,
)
from policy.utils.eval import run_eval_episode, _pretty_print_stats
from mshab.envs.make import EnvConfig, make_env
from mshab.envs.planner import plan_data_from_file
from mshab.utils.array import to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.dataset import ClosableDataLoader
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler


# --- Config --- #


@dataclass
class BCConfig:
    name: str
    lr: float
    batch_size: int
    epochs: int

    eval_freq: int
    log_freq: int
    save_freq: int
    save_backup_ckpts: bool

    data_dir_fp: Optional[str]
    max_image_cache_size: int
    num_dataload_workers: int
    trajs_per_obj: Union[str, int]
    torch_deterministic: bool

    # CLIP / Agent Settings
    clip_input_dim: int
    text_input: List[str]
    camera_intrinsics: List[float]
    bc_loss_weight: float
    num_heads: int
    num_layers_transformer: int
    num_action_layer: int
    action_pred_horizon: int
    action_temp_weights: float
    transf_input_dim: int

    # Map conditioning (optional, only used when condition_map=True)
    map_dir: Optional[str] = None
    decoder_path: Optional[str] = None
    map_level_index: int = 1

    num_eval_envs: int = field(init=False)

    def _additional_processing(self):
        assert self.name == "bc"
        try:
            self.trajs_per_obj = int(self.trajs_per_obj)
        except ValueError:
            pass
        assert isinstance(self.trajs_per_obj, int) or self.trajs_per_obj == "all"


@dataclass
class TrainConfig:
    seed: int
    eval_env: EnvConfig
    algo: BCConfig
    logger: LoggerConfig

    resume_logdir: Optional[Union[Path, str]] = None
    model_ckpt: Optional[Union[Path, int, str]] = None

    # Eval-only fields (used by policy.eval, ignored during training)
    ckpt_path: Optional[str] = None
    out_dir: str = "eval/results"
    condition_map: bool = True
    build_config_prefix: Optional[str] = None

    # Set by the entry point before constructing TrainConfig
    _passed_config_path: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        # --- Handle resuming logic ---
        if self.resume_logdir is not None:
            self.resume_logdir = Path(self.resume_logdir)
            old_config_path = self.resume_logdir / "config.yml"
            if self._passed_config_path is not None and old_config_path.absolute() == Path(self._passed_config_path).absolute():
                assert (
                    self.resume_logdir == self.logger.exp_path
                ), "if setting resume_logdir, must set logger workspace and exp_name accordingly"
            else:
                assert (
                    old_config_path.exists()
                ), f"Couldn't find old config at path {old_config_path}"

            old_cfg_raw = parse_cfg(default_cfg_path=old_config_path)
            # Prevent infinite recursion: old config must not trigger another resume
            OmegaConf.update(old_cfg_raw, "resume_logdir", None)
            old_config = get_mshab_train_cfg(old_cfg_raw)
            self.logger.workspace = old_config.logger.workspace
            self.logger.exp_path = old_config.logger.exp_path
            self.logger.log_path = old_config.logger.log_path
            self.logger.model_path = old_config.logger.model_path
            self.logger.train_video_path = old_config.logger.train_video_path
            self.logger.eval_video_path = old_config.logger.eval_video_path
            self.logger.clear_out = False

            if self.model_ckpt is None:
                self.model_ckpt = self.logger.model_path / "latest.pt"

        # --- Automatic resume from existing checkpoints ---
        if self.model_ckpt is None:
            backup_ckpts = list(self.logger.model_path.glob("*_ckpt.pt"))
            latest_backup_ckpt = None
            if backup_ckpts:
                def _epoch_num(path):
                    m = re.search(r"(\d+)_ckpt\.", path.name)
                    return int(m.group(1)) if m else -1
                latest_backup_ckpt = max(backup_ckpts, key=_epoch_num)

            latest_ckpt = self.logger.model_path / "latest.pt"

            candidate_ckpts = []
            if latest_backup_ckpt is not None:
                candidate_ckpts.append(latest_backup_ckpt)
            if latest_ckpt.exists():
                candidate_ckpts.append(latest_ckpt)

            if candidate_ckpts:
                best_ckpt = None
                best_epoch = -1
                for ckpt_path in candidate_ckpts:
                    try:
                        epoch_val = torch.load(ckpt_path, map_location="cpu")["epoch"]
                        if epoch_val > best_epoch:
                            best_epoch = epoch_val
                            best_ckpt = ckpt_path
                    except Exception as e:
                        print(f"[Warn] Failed to read epoch from {ckpt_path}: {e}")
                if best_ckpt is None:
                    best_ckpt = max(candidate_ckpts, key=lambda p: p.stat().st_mtime)

                self.model_ckpt = best_ckpt
                self.logger.clear_out = False

        if self.model_ckpt is not None:
            self.model_ckpt = Path(self.model_ckpt)
            assert self.model_ckpt.exists(), f"Could not find {self.model_ckpt}"

        self.algo.num_eval_envs = self.eval_env.num_envs
        self.algo._additional_processing()

        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]
        del self.logger.exp_cfg["resume_logdir"]
        del self.logger.exp_cfg["model_ckpt"]
        self.logger.exp_cfg.pop("_passed_config_path", None)


def get_mshab_train_cfg(cfg: dict, config_path: Optional[str] = None) -> TrainConfig:
    data = OmegaConf.to_container(cfg, resolve=True)
    data["_passed_config_path"] = config_path
    return from_dict(data_class=TrainConfig, data=data)


# --- Training --- #


def build_agent(
    cfg: TrainConfig,
    sample_obs,
    single_act_shape,
    device: torch.device,
    task_plan_path: Optional[str] = None,
) -> BCAgent:
    kwargs = dict(
        sample_obs=sample_obs,
        single_act_shape=single_act_shape,
        transf_input_dim=cfg.algo.transf_input_dim,
        clip_input_dim=cfg.algo.clip_input_dim,
        text_input=cfg.algo.text_input,
        num_heads=cfg.algo.num_heads,
        num_layers_transformer=cfg.algo.num_layers_transformer,
        num_action_layer=cfg.algo.num_action_layer,
        action_pred_horizon=cfg.algo.action_pred_horizon,
        condition_map=cfg.condition_map,
    )
    if cfg.condition_map:
        kwargs.update(
            map_dir=cfg.algo.map_dir,
            decoder_path=cfg.algo.decoder_path,
            task_plan_path=task_plan_path or cfg.eval_env.task_plan_fp,
            map_level_index=cfg.algo.map_level_index,
        )
    return BCAgent(**kwargs).to(device)


def setup_models_and_optimizer(
    cfg: TrainConfig, device: torch.device, sample_obs, single_act_shape,
    task_plan_path: Optional[str] = None,
) -> Tuple[BCAgent, Optimizer]:
    agent = build_agent(cfg, sample_obs, single_act_shape, device, task_plan_path=task_plan_path)
    params_to_optimize = filter(lambda p: p.requires_grad, agent.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize, lr=cfg.algo.lr)
    return agent, optimizer


def train_one_epoch(
    agent: BCAgent,
    optimizer: Optimizer,
    dataloader: ClosableDataLoader,
    device: torch.device,
    cfg: BCConfig,
    uid_to_label_map: Dict,
    time_weights: torch.Tensor,
    global_step: int,
    logger: Logger,
    uid_to_episode_idx: Optional[Dict[str, int]] = None,
) -> Tuple[float, int]:
    tot_loss, n_samples = 0.0, 0
    agent.train()

    for batch in tqdm(dataloader, desc="Batch", unit="batch"):
        obs = batch["observations"]
        act = batch["actions"]
        subtask_uids = batch["subtask_uid"]

        subtask_labels = get_object_labels_batch(uid_to_label_map, subtask_uids).to(
            device
        )
        assert (subtask_labels >= 0).all(), (
            f"Found unmapped UIDs: "
            f"{[u for u, l in zip(subtask_uids, subtask_labels) if l < 0]}"
        )

        episode_ids = None
        if uid_to_episode_idx is not None:
            episode_ids = get_episode_ids_batch(uid_to_episode_idx, subtask_uids).to(
                device
            )

        obs, act = to_tensor(obs, device=device, dtype="float"), to_tensor(
            act, device=device, dtype="float"
        )

        pi = agent(obs, subtask_labels, episode_ids=episode_ids)

        total_bc_loss = F.smooth_l1_loss(pi, act, reduction="none")
        total_bc_loss = total_bc_loss * time_weights
        bc_loss = total_bc_loss.mean()

        bc_loss = bc_loss * cfg.bc_loss_weight
        loss = bc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = act.size(0)
        tot_loss += loss.item() * batch_size
        n_samples += batch_size
        global_step += 1
        if logger.tensorboard:
            logger.tb_writer.add_scalar("train_iter/bc_loss", bc_loss.item(), global_step)

    return tot_loss / n_samples if n_samples > 0 else 0.0, global_step


def evaluate_agent(
    agent: BCAgent,
    eval_envs,
    fixed_plan_idxs,
    uid_to_label_map,
    logger,
    device,
    global_step,
    epoch,
    uid_to_episode_idx: Optional[Dict[str, int]] = None,
):
    agent.eval()
    eval_obs, _ = eval_envs.reset(options={"task_plan_idxs": fixed_plan_idxs})

    print("Run eval episode (single horizon)")
    eval_kwargs = dict(
        eval_envs=eval_envs,
        eval_obs=eval_obs,
        agent=agent,
        uid_to_label_map=uid_to_label_map,
        device=device,
    )
    if uid_to_episode_idx is not None:
        eval_kwargs["uid2episode_id"] = uid_to_episode_idx

    stats_single = run_eval_episode(**eval_kwargs)
    _pretty_print_stats("[Eval-Single]", stats_single, logger, color="yellow")

    logger.store(tag="eval", success_once=stats_single["success_once"])
    logger.store(tag="eval", return_per_step=stats_single["return_per_step"])
    logger.store(tag="eval", epoch=epoch)
    logger.log(global_step)

    return stats_single["success_once"]


def train(cfg: TrainConfig):
    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.algo.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_plan_fp = cfg.eval_env.task_plan_fp
    uid_to_label_map = build_object_map(task_plan_fp, cfg.algo.text_input)

    # Filter plans by build_config_prefix (same pattern as eval.py)
    build_config_prefix = getattr(cfg, "build_config_prefix", None)
    if build_config_prefix is not None:
        plan_data = plan_data_from_file(Path(task_plan_fp))
        plan_data.plans = [
            p for p in plan_data.plans
            if p.build_config_name.startswith(build_config_prefix)
        ]
        print(f"Filtered plans by '{build_config_prefix}': {len(plan_data.plans)} plans")
        cfg.eval_env.env_kwargs["task_plans"] = plan_data.plans
        cfg.eval_env.env_kwargs["scene_builder_cls"] = plan_data.dataset
        cfg.eval_env.task_plan_fp = None

    # Make eval env
    print("Making eval env...")
    eval_envs = make_env(cfg.eval_env, video_path=cfg.logger.eval_video_path)
    print("Eval env made.")

    eval_obs, _ = eval_envs.reset(seed=cfg.seed + 1_000_000)
    fixed_plan_idxs = eval_envs.unwrapped.task_plan_idxs.clone()
    eval_envs.action_space.seed(cfg.seed + 1_000_000)
    assert isinstance(eval_envs.single_action_space, gym.spaces.Box)

    agent, optimizer = setup_models_and_optimizer(
        cfg, device, eval_obs, eval_envs.unwrapped.single_action_space.shape,
        task_plan_path=task_plan_fp,
    )

    # Grab the uid→episode mapping built during agent init (map agent only)
    uid_to_episode_idx = getattr(agent, "uid_to_episode_idx", None)

    def save(save_path):
        torch.save(
            dict(
                agent=agent.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                global_step=global_step,
            ),
            save_path,
        )

    def load(load_path):
        checkpoint = torch.load(str(load_path), map_location=device)
        agent_state = BCAgent.strip_legacy_keys(checkpoint["agent"])
        agent.load_state_dict(agent_state)
        optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint.get("epoch", 0), checkpoint.get("global_step", 0)

    logger = Logger(
        logger_cfg=cfg.logger,
        save_fn=save,
    )

    start_epoch = 0
    epoch = 0
    global_step = 0
    if cfg.model_ckpt and cfg.model_ckpt.exists():
        print(f"Resuming from checkpoint: {cfg.model_ckpt}")
        resumed_epoch, resumed_global_step = load(cfg.model_ckpt)
        start_epoch = resumed_epoch + 1
        global_step = resumed_global_step
        print(f"Resumed from epoch {resumed_epoch}. Starting at epoch {start_epoch}.")

    assert eval_envs.unwrapped.control_mode == "pd_joint_delta_pos"
    bc_dataset = DPDataset(
        cfg.algo.data_dir_fp,
        obs_horizon=1,
        pred_horizon=cfg.algo.action_pred_horizon,
        control_mode=eval_envs.unwrapped.control_mode,
        trajs_per_obj=cfg.algo.trajs_per_obj,
        max_image_cache_size=cfg.algo.max_image_cache_size,
        truncate_trajectories_at_success=True,
    )
    bc_dataloader = ClosableDataLoader(
        bc_dataset,
        batch_size=cfg.algo.batch_size,
        shuffle=True,
        num_workers=cfg.algo.num_dataload_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
    )

    def check_freq(freq):
        return epoch % freq == 0

    print("Start training...")
    timer = NonOverlappingTimeProfiler()
    best_success = -1.0

    time_weights = torch.exp(
        -cfg.algo.action_temp_weights
        * torch.arange(cfg.algo.action_pred_horizon, device=device, dtype=torch.float32)
    ).view(1, -1, 1)

    for epoch in range(start_epoch, cfg.algo.epochs):
        logger.print(
            f"Epoch: {epoch}; Global step: {global_step}"
        )

        avg_loss, global_step = train_one_epoch(
            agent,
            optimizer,
            bc_dataloader,
            device,
            cfg.algo,
            uid_to_label_map,
            time_weights,
            global_step,
            logger,
            uid_to_episode_idx=uid_to_episode_idx,
        )
        timer.end(key="train")

        # Save latest checkpoint right after training (before eval)
        save(logger.model_path / "latest.pt")

        # Log
        if check_freq(cfg.algo.log_freq):
            logger.store(tag="losses", loss=avg_loss)
            if epoch > 0:
                logger.store("time", **timer.get_time_logs(epoch))
            logger.log(global_step)
            timer.end(key="log")

        # Evaluation
        if cfg.algo.eval_freq:
            if check_freq(cfg.algo.eval_freq):
                torch.cuda.empty_cache()
                success = evaluate_agent(
                    agent,
                    eval_envs,
                    fixed_plan_idxs,
                    uid_to_label_map,
                    logger,
                    device,
                    global_step,
                    epoch,
                    uid_to_episode_idx=uid_to_episode_idx,
                )
                if success > best_success:
                    best_success = success
                    save(logger.model_path / "best.pt")
                    logger.print(f"New best model (success_once={best_success:.2f})", color="green", bold=True)
                torch.cuda.empty_cache()
                timer.end(key="eval")

        # Checkpoint
        if check_freq(cfg.algo.save_freq):
            if cfg.algo.save_backup_ckpts:
                save(logger.model_path / f"{epoch}_ckpt.pt")
            timer.end(key="checkpoint")

    save(logger.model_path / "final_ckpt.pt")
    save(logger.model_path / "latest.pt")

    bc_dataloader.close()
    eval_envs.close()
    logger.close()


if __name__ == "__main__":
    _config_path = sys.argv[1]
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=_config_path), config_path=_config_path)
    train(cfg)
