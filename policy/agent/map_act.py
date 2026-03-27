import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from policy.module.transformer import ActionTransformerDecoder, TransformerEncoder
from policy.module.mlp import MLP
from mapping.models.vision_backbone import DINOv2Wrapper

logger = logging.getLogger(__name__)

_LEGACY_CHECKPOINT_KEYS = ("all_grid_feats", "all_grid_coords", "all_grid_masks")


class BCAgent(nn.Module):
    def __init__(
        self,
        sample_obs: Dict[str, torch.Tensor],
        single_act_shape: tuple,
        text_input: List[str],
        transf_input_dim: int,
        num_heads: int,
        num_layers_transformer: int,
        clip_input_dim: int,
        num_action_layer: int,
        action_pred_horizon: int,
        # --- Map conditioning ---
        condition_map: bool = True,
        # --- Map-specific parameters (only used when condition_map=True) ---
        map_dir: Optional[str] = None,
        decoder_path: Optional[str] = None,
        task_plan_path: Optional[str] = None,
        map_level_index: int = 1,
    ):
        super().__init__()

        self.condition_map = condition_map

        # --- Feature and Action Dimensions ---
        state_dim = sample_obs["state"].shape[1]
        self.action_dim = np.prod(single_act_shape)

        # --- Vision and Language Pre-trained Models ---
        # DINOv2 for visual features
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.vision_model = DINOv2Wrapper(dino_model)
        self.image_size = 224
        self.feat_size = self.image_size // self.vision_model.patch_size

        # CLIP for text embeddings
        clip_model, _, _ = open_clip.create_model_and_transforms("EVA02-L-14", "merged2b_s4b_b131k")
        tokenizer = open_clip.get_tokenizer("EVA02-L-14")

        if text_input:
            text_input = ['pick up the ' + s.replace('_', ' ') for s in text_input]

        text_tokens = tokenizer(text_input)
        with torch.no_grad():
            text_embeddings = clip_model.encode_text(text_tokens)
            self.register_buffer("text_embeddings", F.normalize(text_embeddings, dim=-1, p=2))

        del clip_model, tokenizer

        # --- Agent Modules ---
        self.text_proj = nn.Linear(clip_input_dim, transf_input_dim)

        self.transformer_encoder = TransformerEncoder(
            input_dim=transf_input_dim,
            hidden_dim=transf_input_dim * 4,
            num_layers=num_layers_transformer,
            num_heads=num_heads,
        )

        self.action_transformer = ActionTransformerDecoder(
            d_model=512,
            transf_input_dim=transf_input_dim,
            nhead=num_heads,
            num_decoder_layers=num_action_layer,
            dim_feedforward=512 * 4,
            dropout=0.1,
            action_dim=self.action_dim,
            action_pred_horizon=action_pred_horizon,
        )

        self.state_mlp_action = MLP(state_dim, transf_input_dim)

        # --- Map conditioning (only for condition_map=True) ---
        if condition_map:
            from mapping.models.latent_decoder import LatentDecoder
            from policy.module.scene_encoder import GlobalSceneEncoder
            from mapping.representations.voxel_hash_table import VoxelHashTable

            assert map_dir is not None, "map_dir required when condition_map=True"
            assert decoder_path is not None, "decoder_path required when condition_map=True"
            assert task_plan_path is not None, "task_plan_path required when condition_map=True"

            # Load frozen LatentDecoder (dims inferred from weights)
            decoder_ckpt = torch.load(decoder_path, map_location="cpu")
            decoder_state = decoder_ckpt.get("model", decoder_ckpt)

            decoder_input_dim = decoder_state["fc1.weight"].shape[1]
            decoder_hidden_dim = decoder_state["fc1.weight"].shape[0]
            decoder_output_dim = decoder_state["fc5.weight"].shape[0]

            self.grid_decoder = LatentDecoder(
                voxel_feature_dim=decoder_input_dim,
                hidden_dim=decoder_hidden_dim,
                output_dim=decoder_output_dim,
            )
            self.grid_decoder.load_state_dict(decoder_state)
            self.grid_decoder.requires_grad_(False)

            # Load all sparse maps and precompute grid features
            map_dir_path = Path(map_dir)
            map_files = sorted(
                f for f in map_dir_path.glob("episode_*.pt")
                if not f.name.endswith(".dense.pt")
            )

            self.episode_name_to_idx: Dict[str, int] = {}
            all_feats: List[torch.Tensor] = []
            all_coords: List[torch.Tensor] = []

            for idx, fpath in enumerate(map_files):
                episode_name = fpath.stem  # e.g. "episode_0018"
                self.episode_name_to_idx[episode_name] = idx
                # Also register the un-padded form so both "episode_18" and
                # "episode_0018" resolve to the same index.
                episode_num = episode_name.split("_", 1)[1]
                unpadded = f"episode_{int(episode_num)}"
                if unpadded != episode_name:
                    self.episode_name_to_idx[unpadded] = idx

                grid = VoxelHashTable.load_sparse(str(fpath), device="cpu")
                level = grid.levels[map_level_index]
                coords = level.coords.clone()  # (L, 3)
                feats = grid.query_voxel_feature(coords)  # (L, decoder_input_dim)

                all_feats.append(feats.detach())
                all_coords.append(coords.detach())
                del grid

            self.num_episodes = len(all_feats)
            max_L = max(f.shape[0] for f in all_feats)
            logger.info(
                f"Loaded {self.num_episodes} sparse maps, max_L={max_L}, "
                f"feat_dim={decoder_input_dim}"
            )

            # Pad to (num_episodes, max_L, ...) and keep on CPU (moved to GPU on-demand)
            padded_feats = torch.zeros(self.num_episodes, max_L, decoder_input_dim, dtype=torch.float16)
            padded_coords = torch.zeros(self.num_episodes, max_L, 3, dtype=torch.float16)
            padded_masks = torch.ones(self.num_episodes, max_L, dtype=torch.bool)  # True = pad

            for i, (f, c) in enumerate(zip(all_feats, all_coords)):
                L = f.shape[0]
                padded_feats[i, :L] = f.half()
                padded_coords[i, :L] = c.half()
                padded_masks[i, :L] = False

            self.all_grid_feats = padded_feats   # CPU, float16
            self.all_grid_coords = padded_coords  # CPU, float16
            self.all_grid_masks = padded_masks    # CPU, bool

            # Build subtask_uid → episode_idx mapping
            self.uid_to_episode_idx: Dict[str, int] = {}
            with open(task_plan_path, "r") as f:
                task_plan_data = json.load(f)

            for plan in task_plan_data.get("plans", []):
                init_config_name = plan.get("init_config_name", "")
                episode_name = Path(init_config_name).stem  # "episode_XXXX"
                if episode_name not in self.episode_name_to_idx:
                    continue
                ep_idx = self.episode_name_to_idx[episode_name]
                for subtask in plan.get("subtasks", []):
                    uid = subtask["uid"]
                    self.uid_to_episode_idx[uid] = ep_idx

            # Count plans whose episodes have no corresponding map file
            skipped_plans = sum(
                1 for plan in task_plan_data.get("plans", [])
                if Path(plan.get("init_config_name", "")).stem not in self.episode_name_to_idx
            )
            if skipped_plans > 0:
                logger.warning(
                    f"{skipped_plans} plan(s) have no matching map file; "
                    f"their subtask UIDs are excluded from uid_to_episode_idx"
                )

            logger.info(
                f"Built uid→episode mapping: {len(self.uid_to_episode_idx)} subtask_uids "
                f"→ {len(set(self.uid_to_episode_idx.values()))} unique episodes"
            )

            # Learnable map-conditioning modules
            self.map_dim_reducer = MLP(decoder_output_dim, transf_input_dim)
            self.scene_encoder = GlobalSceneEncoder(
                in_dim=transf_input_dim, out_dim=transf_input_dim
            )

    @staticmethod
    def strip_legacy_keys(state_dict: dict) -> dict:
        """Remove legacy buffer keys from checkpoints saved with register_buffer."""
        for k in _LEGACY_CHECKPOINT_KEYS:
            state_dict.pop(k, None)
        return state_dict

    def _process_sensor_data(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.shape[2] != 3:
            rgb = rgb.permute(0, 1, 4, 2, 3)

        B, fs, d, H, W = rgb.shape
        rgb = rgb.reshape(B * fs, d, H, W)
        rgb = F.interpolate(rgb.float() / 255.0, size=self.image_size, mode="bicubic", antialias=True)

        visfeat = self.vision_model(rgb)

        N = self.feat_size * self.feat_size
        feats = visfeat.permute(0, 2, 3, 1).reshape(B, N, -1)
        return feats

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        object_labels: torch.Tensor,
        episode_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state = observations["state"].squeeze(1)

        feats_hand = self._process_sensor_data(observations["fetch_hand_rgb"])
        feats_head = self._process_sensor_data(observations["fetch_head_rgb"])
        feats = torch.cat([feats_hand, feats_head], dim=1)

        text_emb = self.text_proj(self.text_embeddings[object_labels]).unsqueeze(1)
        state_tok = self.state_mlp_action(state).unsqueeze(1)
        visual_tok = self.transformer_encoder(visual_token=feats, use_pe=True)

        # Map conditioning
        global_tok = None
        if episode_ids is not None and hasattr(self, "grid_decoder"):
            # Cache global_tok during eval (episode_ids is constant within a rollout)
            cache_hit = (
                not self.training
                and hasattr(self, "_cached_episode_ids")
                and torch.equal(episode_ids, self._cached_episode_ids)
            )
            if cache_hit:
                global_tok = self._cached_global_tok
            else:
                device = state.device
                ids_cpu = episode_ids.cpu()
                grid_feats = self.all_grid_feats[ids_cpu].to(device=device, dtype=torch.float32)
                grid_coords = self.all_grid_coords[ids_cpu].to(device=device, dtype=torch.float32)
                grid_masks = self.all_grid_masks[ids_cpu].to(device)    # (B, max_L)

                with torch.no_grad():
                    decoded = self.grid_decoder(grid_feats)  # (B, max_L, decoder_out)

                reduced = self.map_dim_reducer(decoded)  # (B, max_L, transf_input_dim)
                pts_kv = torch.cat([grid_coords, reduced], dim=-1)  # (B, max_L, 3+D)
                global_feat = self.scene_encoder(pts_kv, pad=grid_masks)  # (B, D)
                global_tok = global_feat.unsqueeze(1)  # (B, 1, D)

                if not self.training:
                    self._cached_global_tok = global_tok
                    self._cached_episode_ids = episode_ids.clone()

        return self.action_transformer(visual_tok, state_tok, text_emb, global_tok=global_tok)
