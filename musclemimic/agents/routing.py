"""Validate action ownership before advancing a shared physical world."""
import numpy as np


class ActionRouter:
    def __init__(self, agents, nu):
        self.agents = tuple(agents)
        self.nu = nu
        identifiers = [a.agent_id for a in self.agents]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Duplicate agent_id")
        owned = [i for a in self.agents for i in a.actuator_indices]
        if len(set(owned)) != len(owned) or any(i < 0 or i >= nu for i in owned):
            raise ValueError("Actuator ownership overlaps or is out of range")
        if any(a.control_period != 1 for a in self.agents):
            raise ValueError("This synchronous router requires control_period=1")

    def route(self, actions):
        if set(actions) != {a.agent_id for a in self.agents}:
            raise ValueError("Provide exactly one command for every agent")
        ctrl = np.zeros(self.nu)
        for agent in self.agents:
            value = np.asarray(actions[agent.agent_id])
            if value.shape != (len(agent.actuator_indices),) or not np.isfinite(value).all():
                raise ValueError(f"Invalid action for {agent.agent_id}")
            ctrl[list(agent.actuator_indices)] = value
        return ctrl
