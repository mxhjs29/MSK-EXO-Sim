"""Retargeting utilities and visualization entry points."""
from .msk_manifest import MskManifest, build_msk_manifest, manifest_from_model
from .msk_profiles import MskRetargetProfile, SUPPORTED_MSK_KEYS, load_msk_retarget_profile
from .msk_adapter import adapt_msk_trajectory_to_composed_model


def retarget_dataset_to_assist_msk(*args, **kwargs):
    """Lazy import keeps optional Torch/GMR initialization out of model setup."""
    from .msk_retarget import retarget_dataset_to_assist_msk as implementation

    return implementation(*args, **kwargs)

__all__ = [
    "MskManifest",
    "MskRetargetProfile",
    "SUPPORTED_MSK_KEYS",
    "build_msk_manifest",
    "manifest_from_model",
    "load_msk_retarget_profile",
    "retarget_dataset_to_assist_msk",
    "adapt_msk_trajectory_to_composed_model",
]
