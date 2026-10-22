#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../scripts/train_walk_gpu2.sh" "$@"
