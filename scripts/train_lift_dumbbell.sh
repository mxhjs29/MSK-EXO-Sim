#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# GPU is the physical nvidia-smi index; default to GPU 0.
LIFT_GPU_UUID=$(nvidia-smi -i "${GPU:-0}" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$LIFT_GPU_UUID"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MUJOCO_GL=osmesa
export OPENBLAS_NUM_THREADS=1
export CONVERTED_C3D_PATH="$PWD/outputs/fullbody_lift_dumbbell/reference_data"
exec .venv/bin/python fullbody/experiment.py --config-name=conf_fullbody_lift_dumbbell \
    hydra.run.dir=outputs/fullbody_lift_dumbbell/run "$@"
