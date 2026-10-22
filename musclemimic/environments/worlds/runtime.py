"""CPU shared-world primitive; learning and task termination live above this layer."""
import mujoco
import numpy as np
from musclemimic.agents.routing import ActionRouter


class WorldRuntime:
    def __init__(self, model, agents, n_substeps=1):
        if not isinstance(n_substeps, int) or n_substeps < 1:
            raise ValueError("Positive integer n_substeps required")
        self.model = model
        self.data = mujoco.MjData(model)
        self.router = ActionRouter(agents, model.nu)
        self.n_substeps = n_substeps

    def reset(self, qpos=None, qvel=None):
        for value, size in ((qpos, self.model.nq), (qvel, self.model.nv)):
            if value is not None and (np.shape(value) != (size,) or not np.isfinite(value).all()):
                raise ValueError("Invalid reset state")
        mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def step(self, actions):
        self.data.ctrl[:] = self.router.route(actions)
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        return self.data
