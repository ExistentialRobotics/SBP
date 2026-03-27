#!/usr/bin/bash
#
# Evaluate BC agent for a selected task, with optional map conditioning.
#
# Usage:
#   bash scripts/eval.sh                                   # interactive menu (map conditioning on)
#   bash scripts/eval.sh set_table                         # map conditioning on (default)
#   bash scripts/eval.sh set_table false                   # map conditioning off
#   CONDITION_MAP=false bash scripts/eval.sh set_table     # map conditioning off via env var
#   SEED=1 bash scripts/eval.sh tidy_house                # with seed override

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
# Configuration (override via environment variables)
# ============================================================
CONDITION_MAP="${CONDITION_MAP:-true}"

# Optional 2nd argument for map conditioning
if [ $# -ge 2 ]; then
    CONDITION_MAP="$2"
fi
SEED="${SEED:-0}"
SUBTASK="${SUBTASK:-pick}"
SPLIT="${SPLIT:-train}"
SCENES="${SCENES:-13}"

NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-16}"
CKPT_NAME="${CKPT_NAME:-final_ckpt.pt}"

# Validate condition_map
if [ "$CONDITION_MAP" != "true" ] && [ "$CONDITION_MAP" != "false" ]; then
    echo "ERROR: CONDITION_MAP must be 'true' or 'false', got '$CONDITION_MAP'"
    exit 1
fi

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${SBP_ROOT}/third_party/mshab:${SBP_ROOT}/third_party/ManiSkill:${SBP_ROOT}:${PYTHONPATH:-}"

export MS_ASSET_DIR="$SBP_ROOT/.maniskill"
MS_DATA_DIR="$MS_ASSET_DIR/data"
REARRANGE_ROOT="$MS_DATA_DIR/scene_datasets/replica_cad_dataset/rearrange"

WORKSPACE="${WORKSPACE:-mshab_exps}"

if [ "$CONDITION_MAP" = "true" ]; then
    MAP_DIR="${MAP_DIR:-$SBP_ROOT/data/models/$TASK}"
    DECODER_PATH="${DECODER_PATH:-$SBP_ROOT/data/models/latent_decoder.pt}"
fi

# ============================================================
# Derived variables
# ============================================================
# shellcheck disable=SC2001
ENV_ID="$(echo "$SUBTASK" | sed 's/\b\(.\)/\u\1/g')SubtaskTrain-v0"
GROUP="$TASK-$SUBTASK"

BASE_CONFIG="policy/configs/${TASK}.yml"
if [ "$CONDITION_MAP" = "true" ]; then
    AGENT_LABEL="map"
else
    AGENT_LABEL="image"
fi
CKPT_DIR="$WORKSPACE/$ENV_ID/$GROUP/$TASK-$SUBTASK-$AGENT_LABEL-$SEED/models"

# ============================================================
# Evaluation
# ============================================================
echo ""
echo "============================================================"
echo "  Evaluation: task=$TASK | condition_map=$CONDITION_MAP | seed=$SEED"
echo "============================================================"

PLAN_FP="$REARRANGE_ROOT/task_plans/$TASK/$SUBTASK/$SPLIT/all.json"
if [ ! -f "$PLAN_FP" ]; then
    echo "Plan file not found: $PLAN_FP"
    exit 1
fi

SPAWN_DATA_FP="$REARRANGE_ROOT/spawn_data/$TASK/$SUBTASK/$SPLIT/spawn_data.pt"

for SCENE in $SCENES; do
    BUILD_CONFIG="v3_sc1_staging_${SCENE}"

    echo ""
    echo "------------------------------------------------------------"
    echo "  Eval: task=$TASK | condition_map=$CONDITION_MAP | scene=$BUILD_CONFIG | seed=$SEED"
    echo "------------------------------------------------------------"

    eval_args=(
        seed="$SEED"
        "eval_env.task_plan_fp=$PLAN_FP"
        "eval_env.spawn_data_fp=$SPAWN_DATA_FP"
        "eval_env.num_envs=$NUM_EVAL_ENVS"
        "ckpt_path=$CKPT_DIR/$CKPT_NAME"
        "out_dir=eval/results/$TASK"
        "condition_map=$CONDITION_MAP"
        "build_config_prefix=$BUILD_CONFIG"
    )

    if [ "$CONDITION_MAP" = "true" ]; then
        eval_args+=(
            "algo.map_dir=$MAP_DIR"
            "algo.decoder_path=$DECODER_PATH"
        )
    fi

    SAPIEN_NO_DISPLAY=1 python -m policy.eval "$BASE_CONFIG" "${eval_args[@]}"
done

echo ""
echo "============================================================"
echo "  Done! Results saved to eval/results/$TASK/"
echo "============================================================"
