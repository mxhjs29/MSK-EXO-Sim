"""Policy ownership is separate from physical entity composition."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    entity_id: str
    policy_id: str
    actuator_indices: tuple[int, ...]
    controller_type: str = "ppo"
    trainable: bool = True
    observation_indices: tuple[int, ...] = ()
    critic_indices: tuple[int, ...] = ()
    control_period: int = 1
