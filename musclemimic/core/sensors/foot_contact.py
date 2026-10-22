"""Measured normal foot forces, sampled without changing the live solver state."""
import mujoco
import numpy as np

CONTACT_NAMES = ("r_foot", "r_toes", "l_foot", "l_toes")


class FootContactSensor:
    def __init__(self, model, names=CONTACT_NAMES):
        self.model = model
        ids = [model.sensor(n).id for n in names]
        if any(model.sensor_type[i] != mujoco.mjtSensor.mjSENS_TOUCH or model.sensor_dim[i] != 1 for i in ids):
            raise ValueError("Expected scalar MuJoCo touch sensors")
        self.addresses = model.sensor_adr[ids].copy()
        self.shadow = mujoco.MjData(model)
        self.state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))

    def values(self, data):
        forces = np.asarray(data.sensordata[self.addresses], np.float32).copy()
        if not np.isfinite(forces).all() or (forces < -1e-6).any():
            raise FloatingPointError("Invalid foot contact force")
        return np.maximum(forces, 0)

    def sample(self, data):
        mujoco.mj_getState(self.model, data, self.state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_setState(self.model, self.shadow, self.state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.shadow)
        return self.values(self.shadow)
