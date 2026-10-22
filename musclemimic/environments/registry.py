"""Expose the existing environment registry without a second source of truth."""
from musclemimic.environments.base import LocoEnv


def registered_environments():
    from musclemimic import environments  # register maintained families
    return dict(LocoEnv.registered_envs)
