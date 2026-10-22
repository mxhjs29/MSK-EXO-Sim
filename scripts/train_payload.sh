#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PAYLOAD_TASK="${1:?Usage: train_payload.sh lift|bend [payload_finetune train options]}"
shift
case "$PAYLOAD_TASK" in
  lift) PAYLOAD_GPU_UUID=${MUSCLEMIMIC_GPU_PAYLOAD_LIFT:-2}; PAYLOAD_CPUS=${MUSCLEMIMIC_CPUS_PAYLOAD_LIFT:-$(.venv/bin/python -c 'from musclemimic.runner.jobs.resources import cpu_set_for; print(cpu_set_for("payload_lift"))')} ;;
  bend) PAYLOAD_GPU_UUID=${MUSCLEMIMIC_GPU_PAYLOAD_BOX:-3}; PAYLOAD_CPUS=${MUSCLEMIMIC_CPUS_PAYLOAD_BOX:-$(.venv/bin/python -c 'from musclemimic.runner.jobs.resources import cpu_set_for; print(cpu_set_for("payload_box"))')} ;;
  *) echo "Task must be lift or bend" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="$PAYLOAD_GPU_UUID"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=osmesa
export OPENBLAS_NUM_THREADS=1
# Use the original PPO environment: stage2's older cuDNN is incompatible with JAX.
exec taskset -c "$PAYLOAD_CPUS" .venv/bin/python -u -m musclemimic.experiments.payload_finetune train --task "$PAYLOAD_TASK" "$@"
