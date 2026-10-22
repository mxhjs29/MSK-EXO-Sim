"""MuscleMimic utilities - custom implementations"""

from .display import (
    detect_headless_environment,
    setup_headless_rendering,
    setup_headless_rendering_if_needed,
)
from .logging import TimestepTracker, setup_logger
from .metrics import MetricsHandler, QuantityContainer, ValidationSummary
from .model import count_actor_critic_params, count_params, count_params_by_path, count_trainable_params
from .utd import compute_utd

__all__ = [
    "MetricsHandler",
    "QuantityContainer",
    "TimestepTracker",
    "ValidationSummary",
    # MuscleMimic custom utilities
    "compute_utd",
    "count_actor_critic_params",
    "count_params",
    "count_params_by_path",
    "count_trainable_params",
    "detect_headless_environment",
    # LocoMuJoCo utilities
    "set_all_caches",
    "set_amass_path",
    "set_c3d_model_path",
    "set_converted_amass_path",
    "set_converted_c3d_path",
    "set_converted_lafan1_path",
    "set_lafan1_path",
    "set_moshpp_assets_path",
    "set_smpl_model_path",
    "setup_headless_rendering",
    "setup_headless_rendering_if_needed",
    "setup_logger",
]


def set_all_caches(*args, **kwargs):
    from loco_mujoco.utils import set_all_caches as _set_all_caches

    return _set_all_caches(*args, **kwargs)


def set_amass_path(*args, **kwargs):
    from loco_mujoco.utils import set_amass_path as _set_amass_path

    return _set_amass_path(*args, **kwargs)


def set_c3d_model_path(*args, **kwargs):
    from loco_mujoco.utils import set_c3d_model_path as _set_c3d_model_path

    return _set_c3d_model_path(*args, **kwargs)


def set_converted_amass_path(*args, **kwargs):
    from loco_mujoco.utils import set_converted_amass_path as _set_converted_amass_path

    return _set_converted_amass_path(*args, **kwargs)


def set_converted_c3d_path(*args, **kwargs):
    from loco_mujoco.utils import set_converted_c3d_path as _set_converted_c3d_path

    return _set_converted_c3d_path(*args, **kwargs)


def set_converted_lafan1_path(*args, **kwargs):
    from loco_mujoco.utils import set_converted_lafan1_path as _set_converted_lafan1_path

    return _set_converted_lafan1_path(*args, **kwargs)


def set_lafan1_path(*args, **kwargs):
    from loco_mujoco.utils import set_lafan1_path as _set_lafan1_path

    return _set_lafan1_path(*args, **kwargs)


def set_moshpp_assets_path(*args, **kwargs):
    from loco_mujoco.utils import set_moshpp_assets_path as _set_moshpp_assets_path

    return _set_moshpp_assets_path(*args, **kwargs)


def set_smpl_model_path(*args, **kwargs):
    from loco_mujoco.utils import set_smpl_model_path as _set_smpl_model_path

    return _set_smpl_model_path(*args, **kwargs)
