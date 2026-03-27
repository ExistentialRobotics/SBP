#!/usr/bin/bash
#
# Train BC agent for a selected task, with optional map conditioning.
#
# Usage:
#   bash scripts/train.sh                                  # interactive menu (map conditioning on)
#   bash scripts/train.sh set_table                        # map conditioning on (default)
#   bash scripts/train.sh set_table false                  # map conditioning off
#   CONDITION_MAP=false bash scripts/train.sh set_table    # map conditioning off via env var

set -euo pipefail

# ============================================================
# Task selection
# ============================================================
TASKS=("set_table")

if [ $# -ge 1 ]; then
    TASK="$1"
else
    echo "Select a task:"
    for i in "${!TASKS[@]}"; do
        echo "  $((i+1))) ${TASKS[$i]}"
    done
    read -rp "Enter number [1-${#TASKS[@]}]: " choice
    if [[ "$choice" =~ ^[1-3]$ ]]; then
        TASK="${TASKS[$((choice-1))]}"
    else
        echo "Invalid choice: $choice"
        exit 1
    fi
fi

# Validate task name
valid=false
for t in "${TASKS[@]}"; do
    if [ "$t" = "$TASK" ]; then valid=true; break; fi
done
if [ "$valid" = false ]; then
    echo "ERROR: Unknown task '$TASK'. Must be one of: ${TASKS[*]}"
    exit 1
fi

# ============================================================
# Map conditioning selection
# ============================================================
CONDITION_MAP="${CONDITION_MAP:-true}"

if [ $# -ge 2 ]; then
    CONDITION_MAP="$2"
fi

if [ "$CONDITION_MAP" != "true" ] && [ "$CONDITION_MAP" != "false" ]; then
    echo "ERROR: CONDITION_MAP must be 'true' or 'false', got '$CONDITION_MAP'"
    exit 1
fi

# ============================================================
# Configuration (override via environment variables)
# ============================================================
SEED="${SEED:-0}"
SUBTASK="${SUBTASK:-pick}"
SPLIT="${SPLIT:-train}"
OBJ="${OBJ:-all}"
SCENES="${SCENES:-13}"

NUM_ENVS="${NUM_ENVS:-10}"
TRAJS_PER_OBJ="${TRAJS_PER_OBJ:-1000}"
MAX_IMAGE_CACHE_SIZE="${MAX_IMAGE_CACHE_SIZE:-0}"
NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS:-2}"

TENSORBOARD="${TENSORBOARD:-True}"

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${SBP_ROOT}/third_party/mshab:${SBP_ROOT}/third_party/ManiSkill:${SBP_ROOT}:${PYTHONPATH:-}"

export MS_ASSET_DIR="$SBP_ROOT/.maniskill"
MS_DATA_DIR="$MS_ASSET_DIR/data"
REARRANGE_ROOT="$MS_DATA_DIR/scene_datasets/replica_cad_dataset/rearrange"
DATASET_ROOT="$SBP_ROOT/data/demonstrations"

WORKSPACE="${WORKSPACE:-mshab_exps}"

MAP_DIR="${MAP_DIR:-$SBP_ROOT/data/models/$TASK}"
DECODER_PATH="${DECODER_PATH:-$SBP_ROOT/data/models/latent_decoder.pt}"

# ============================================================
# Derived variables
# ============================================================
# shellcheck disable=SC2001
ENV_ID="$(echo "$SUBTASK" | sed 's/\b\(.\)/\u\1/g')SubtaskTrain-v0"
GROUP="$TASK-$SUBTASK"

TASK_PLAN_FP="$REARRANGE_ROOT/task_plans/$TASK/$SUBTASK/$SPLIT/all.json"
BUILD_CONFIG="v3_sc1_staging_${SCENES}"
SPAWN_DATA_FP="$REARRANGE_ROOT/spawn_data/$TASK/$SUBTASK/$SPLIT/spawn_data.pt"
DATA_DIR_FP="$DATASET_ROOT/$TASK/$SUBTASK/$OBJ.h5"

BASE_CONFIG="policy/configs/${TASK}.yml"
if [ "$CONDITION_MAP" = "true" ]; then
    AGENT_LABEL="map"
else
    AGENT_LABEL="image"
fi
EXP_NAME="$ENV_ID/$GROUP/$TASK-$SUBTASK-$AGENT_LABEL-$SEED"
PROJECT_NAME="$TASK-$SUBTASK-$AGENT_LABEL"
RESUME_LOGDIR="$WORKSPACE/$EXP_NAME"
RESUME_CONFIG="$RESUME_LOGDIR/config.yml"

# ============================================================
# Training args
# ============================================================
args=(
    "seed=$SEED"
    "condition_map=$CONDITION_MAP"
    "eval_env.env_id=$ENV_ID"
    "eval_env.task_plan_fp=$TASK_PLAN_FP"
    "build_config_prefix=$BUILD_CONFIG"
    "eval_env.spawn_data_fp=$SPAWN_DATA_FP"
    "eval_env.frame_stack=1"
    "eval_env.make_env=True"
    "eval_env.num_envs=$NUM_ENVS"
    "eval_env.max_episode_steps=200"
    "eval_env.record_video=True"
    "eval_env.info_on_video=True"
    "eval_env.save_video_freq=5"
    "algo.trajs_per_obj=$TRAJS_PER_OBJ"
    "algo.data_dir_fp=$DATA_DIR_FP"
    "algo.max_image_cache_size=$MAX_IMAGE_CACHE_SIZE"
    "algo.num_dataload_workers=$NUM_DATALOAD_WORKERS"
    "algo.eval_freq=1"
    "algo.log_freq=1"
    "algo.save_freq=1"
    "algo.save_backup_ckpts=True"
    "logger.tensorboard=$TENSORBOARD"
    "logger.exp_name=$EXP_NAME"
    "logger.project_name=$PROJECT_NAME"
    "logger.workspace=$WORKSPACE"
)

if [ "$CONDITION_MAP" = "true" ]; then
    args+=(
        "algo.map_dir=$MAP_DIR"
        "algo.decoder_path=$DECODER_PATH"
    )
fi

TRAIN_MODULE="policy.train_bc"

# ============================================================
# Launch training
# ============================================================
echo ""
echo "============================================================"
echo "  Training: condition_map=$CONDITION_MAP | task=$TASK | seed=$SEED"
echo "============================================================"

# Skip if training already completed
if [ -f "$RESUME_LOGDIR/models/final_ckpt.pt" ]; then
    echo "ALREADY DONE: found $RESUME_LOGDIR/models/final_ckpt.pt"
    exit 0
fi

if [ -f "$RESUME_CONFIG" ] && [ -f "$RESUME_LOGDIR/models/latest.pt" ]; then
    echo "RESUMING from $RESUME_LOGDIR"
    SAPIEN_NO_DISPLAY=1 python -m $TRAIN_MODULE "$RESUME_CONFIG" \
        resume_logdir="$RESUME_LOGDIR" \
        logger.clear_out="False" \
        "logger.best_stats_cfg={eval/success_once: 1, eval/return_per_step: 1}" \
        "${args[@]}"
else
    echo "STARTING fresh"
    SAPIEN_NO_DISPLAY=1 python -m $TRAIN_MODULE "$BASE_CONFIG" \
        logger.clear_out="True" \
        "logger.best_stats_cfg={eval/success_once: 1, eval/return_per_step: 1}" \
        "${args[@]}"
fi
