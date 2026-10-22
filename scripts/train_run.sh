#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Select a physical GPU explicitly, e.g. RUN_GPU_INDEX=3 bash fullbody/train_run.sh.
RUN_GPU_UUID=$(nvidia-smi -i "${RUN_GPU_INDEX:-3}" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$RUN_GPU_UUID"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MUJOCO_GL=osmesa
export OPENBLAS_NUM_THREADS=1
export AMASS_PATH="$PWD/outputs/fullbody_run_h/reference_data/sources"
export SMPL_MODEL_PATH="$PWD/musclemimic/models/smpl"
export CONVERTED_AMASS_PATH="$PWD/outputs/fullbody_run_h/reference_data/amass_cache"
exec .venv/bin/python fullbody/experiment.py --config-name=conf_fullbody_run \
    hydra.run.dir=outputs/fullbody_run_h/run "$@"
