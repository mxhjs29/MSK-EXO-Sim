#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Select physical GPU 2 by UUID to avoid CUDA enumeration differences.
WALK_GPU_UUID=$(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$WALK_GPU_UUID"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Return unused JAX allocations so Warp can allocate CUDA graphs.
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MUJOCO_GL=osmesa
export OPENBLAS_NUM_THREADS=1
export AMASS_PATH="$PWD/datasets/musclemimic-msk"
export SMPL_MODEL_PATH="$PWD/musclemimic/models/smpl"
export CONVERTED_AMASS_PATH="$PWD/outputs/fullbody_walk_kit11/amass_cache"
.venv/bin/python - <<'PY'
import os
import jax
import warp as wp
wp.init()
print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
print('JAX:', jax.devices())
print('Warp:', wp.get_cuda_devices())
if not wp.get_cuda_devices():
    raise SystemExit('Warp cannot initialize CUDA; training was not started.')
PY
if [[ "${1:-}" == "--check" ]]; then
    exit 0
fi
exec .venv/bin/python fullbody/experiment.py --config-name=conf_fullbody_walk \
    hydra.run.dir=outputs/fullbody_walk_kit11/run "$@"
