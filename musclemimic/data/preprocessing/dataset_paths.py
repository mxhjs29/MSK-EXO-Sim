"""Paths shared by the relocated dataset tools (independent of cwd)."""
import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "pyproject.toml").is_file() and (p / "musclemimic").is_dir())
DATASET = ROOT / "datasets" / "musclemimic-msk"
SCRIPTS = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "datasets" / "output"
CONFIG = SCRIPTS / "ik_config.json"
SMPL_MODELS = ROOT / "musclemimic" / "models" / "smpl"
sys.path.insert(0, str(ROOT))
# Explicit user overrides retain precedence over the local default.
os.environ.setdefault("MUSCLEMIMIC_SMPL_MODEL_PATH", str(SMPL_MODELS))
