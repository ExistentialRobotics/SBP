#!/bin/bash
set -euo pipefail

ENV_NAME="sbp"

# --- Create conda environment --- #
echo "Creating conda environment: $ENV_NAME"
conda env create -f environment.yml

# Set LD_LIBRARY_PATH to resolve SAPIEN libstdc++ compatibility
CONDA_ENV_PATH="$(conda info --envs | grep "^$ENV_NAME " | awk '{print $NF}')"
conda env config vars set LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib" -n "$ENV_NAME"
conda env config vars set MS_ASSET_DIR="$(pwd)/.maniskill" -n "$ENV_NAME"

# Activate environment (disable nounset temporarily — conda scripts use unbound vars)
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
set -u

# --- Install PyG extensions (prebuilt wheels matching PyTorch + CUDA) --- #
TORCH_VER="$(python -c "import torch; print(torch.__version__)")"
echo "Installing torch-cluster for PyTorch $TORCH_VER..."
pip install torch-cluster -f "https://data.pyg.org/whl/torch-${TORCH_VER}.html"

# --- Install submodules --- #
echo "Installing submodules..."
pip install -e third_party/ManiSkill -e third_party/mshab

# --- Download ManiSkill assets (skip if already present) --- #
export MS_ASSET_DIR="$(pwd)/.maniskill"
if [ ! -d "$MS_ASSET_DIR" ]; then
    echo "Downloading ManiSkill assets..."
    yes | python -m mani_skill.utils.download_asset ycb
    yes | python -m mani_skill.utils.download_asset ReplicaCAD
    yes | python -m mani_skill.utils.download_asset ReplicaCADRearrange
else
    echo "ManiSkill assets already exist at $MS_ASSET_DIR, skipping download."
fi

echo ""
echo "Done! Run: conda activate $ENV_NAME"
