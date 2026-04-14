#!/bin/bash
set -e

# ==============================================================================
# Proteina-Complexa - conda + pip 安装脚本
# 用法：
#   bash build_uv_env_adapted.sh              # 完整安装
#   bash build_uv_env_adapted.sh --minimal    # 跳过 ColabFold / JAX / tmol
#   bash build_uv_env_adapted.sh --clean      # 删除已有 env 重建
#   bash build_uv_env_adapted.sh --name NAME  # 自定义 env 名（默认 complexa）
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_INSTALL=true
CLEAN=false
ENV_NAME="complexa"

while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal) FULL_INSTALL=false; shift ;;
        --clean)   CLEAN=true;         shift ;;
        --name)    ENV_NAME="$2";      shift 2 ;;
        -h|--help)
            echo "Usage: bash build_uv_env_adapted.sh [--minimal] [--clean] [--name NAME]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "  Proteina-Complexa - conda + pip installer"
echo "  Project : $PROJECT_DIR"
echo "  Env     : $ENV_NAME"
echo "  Full    : $FULL_INSTALL"
echo "=============================================="

# 1. conda env
if [[ "$CLEAN" == "true" ]]; then
    conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
fi

echo "[1/8] Creating conda env '$ENV_NAME' (Python 3.12)..."
conda create -n "$ENV_NAME" python=3.12 -y

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 2. PyTorch 2.7.0 + CUDA 12.6
echo "[2/8] Installing PyTorch 2.7.0 + CUDA 12.6..."
pip install torch==2.7.0+cu126 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# 3. 项目基础依赖
echo "[3/8] Installing base dependencies (pyproject.toml)..."
pip install -e "$PROJECT_DIR"

# 4. PyTorch Geometric
echo "[4/8] Installing PyTorch Geometric..."
pip install torch_geometric torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.7.0+cu126.html

# 5. Graphein + Atomworks
echo "[5/8] Installing Graphein and Atomworks..."
pip install graphein==1.7.7 --no-deps
pip install "atomworks[ml,openbabel,dev]" || echo "Warning: atomworks install failed"

# 6. 可选：ColabFold / JAX / tmol
if [ "$FULL_INSTALL" = true ]; then
    echo "[6/8] Installing ColabFold, JAX, tmol..."

    pip install colabdesign==1.1.1 alphafold-colabfold==2.3.7
    pip install -e "$PROJECT_DIR/community_models/colabdesign"

    pip install jaxlib==0.4.29+cuda12.cudnn91 \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install "jax[cuda12]==0.4.29" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install flax==0.9.0 --no-deps

    pip install "git+https://github.com/uw-ipd/tmol.git@d8a6f7f9649d36e74440bca25246ee7c467ce490" \
        || echo "Warning: tmol install failed"
else
    echo "[6/8] Skipping optional deps"
fi

# 7. rc-foundry
echo "[7/8] Installing rc-foundry..."
pip install "rc-foundry[all]" || echo "Warning: rc-foundry install failed"

# 8. 固定 biotite 版本
echo "[8/8] Pinning biotite to 1.6.0..."
pip install biotite==1.6.0

echo ""
echo "=============================================="
echo "  Done! Activate with: conda activate $ENV_NAME"
echo "  Verify : python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
echo "=============================================="
