"""Small, explicit registry for Assist MSK retargeting targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from .msk_manifest import MskManifest


PROFILE_VERSION = "assist-msk-v1"
SUPPORTED_MSK_KEYS = ("myofullbody", "myoarms", "myolegs")


@dataclass(frozen=True)
class MskRetargetProfile:
    source_msk_key: str
    target_msk_key: str
    version: str
    ik_config: Path | None


def load_msk_retarget_profile(msk_key: str, manifest: MskManifest | None = None) -> MskRetargetProfile:
    if msk_key not in SUPPORTED_MSK_KEYS:
        raise ValueError(f"Unsupported msk_key {msk_key!r}; choose from {SUPPORTED_MSK_KEYS}")
    ik_config = None
    if msk_key == "myofullbody":
        ik_config = Path(__file__).parents[2] / "loco_mujoco" / "smpl" / "gmr_configs" / "smplh_to_myofullbody.json"
        if not ik_config.is_file():
            raise FileNotFoundError(ik_config)
    if manifest is not None and manifest.msk_key != msk_key:
        raise ValueError(f"Profile {msk_key} does not match manifest {manifest.msk_key}")
    return MskRetargetProfile("myofullbody", msk_key, PROFILE_VERSION, ik_config)


def load_fullbody_solver_config():
    """Solver parameters, separated from environment-name dispatch."""
    root = Path(__file__).parents[2] / "loco_mujoco" / "smpl" / "robot_confs"
    return OmegaConf.merge(OmegaConf.load(root / "defaults.yaml"), OmegaConf.load(root / "MyoFullBody.yaml"))
