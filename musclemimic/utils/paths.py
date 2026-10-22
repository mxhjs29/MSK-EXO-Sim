"""Resolve workspace data and maintained configuration independently of cwd."""
import os
import sys
from pathlib import Path


def workspace_root() -> Path:
    override = os.environ.get("MUSCLEMIMIC_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for path in Path(__file__).resolve().parents:
        if (path / "pyproject.toml").is_file() and (path / "models").is_dir():
            return path
    return Path.cwd()


def config_root() -> Path:
    override = os.environ.get("MUSCLEMIMIC_CONFIG_DIR")
    candidates = [Path(override)] if override else [
        workspace_root() / "configs", Path(sys.prefix) / "share/musclemimic/configs"
    ]
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError("Configuration directory missing; set MUSCLEMIMIC_CONFIG_DIR")
