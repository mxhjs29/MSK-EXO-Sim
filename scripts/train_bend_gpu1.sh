#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BEND_GPU_UUID=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$BEND_GPU_UUID"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MUJOCO_GL=osmesa
export OPENBLAS_NUM_THREADS=1
export AMASS_PATH="$PWD/datasets/musclemimic-msk"
export SMPL_MODEL_PATH="$PWD/musclemimic/models/smpl"
export CONVERTED_AMASS_PATH="$PWD/outputs/fullbody_bend/reference_data/amass_cache_root_aligned"
exec .venv/bin/python fullbody/experiment.py --config-name=conf_fullbody_bend \
    hydra.run.dir=outputs/fullbody_bend/run "$@"
