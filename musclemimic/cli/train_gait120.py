from musclemimic.utils.paths import config_root
"""Gait120 PPO entrypoint with durable, machine-readable progress records."""
import json
from datetime import datetime, timezone
from pathlib import Path

import hydra
import jax
import numpy as np
from hydra.core.hydra_config import HydraConfig
from musclemimic.runner.engine import run_experiment
from musclemimic.runner.logging import ExperimentHooks


class RecordHooks(ExperimentHooks):
    def build_video_recorder(self, result_dir, config):
        # Separate final CPU evaluator records actual muscle states and video.
        return None

    def enrich_log(self, log_dict, metrics_dict, env):
        path = Path(HydraConfig.get().runtime.output_dir)
        record = {k: np.asarray(v).tolist() for k, v in metrics_dict.items()
                  if not k.startswith("_")}
        record["recorded_at"] = datetime.now(timezone.utc).isoformat()
        with (path / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        tmp = path / "progress.json.tmp"
        tmp.write_text(json.dumps(record, indent=2, allow_nan=False))
        tmp.replace(path / "progress.json")


@hydra.main(version_base=None, config_path=str(config_root()/"experiments/fullbody"), config_name="conf_gait120_emg")
def main(config):
    jax.config.update("jax_default_matmul_precision", "high")
    run_experiment(config, RecordHooks())


if __name__ == "__main__":
    main()
