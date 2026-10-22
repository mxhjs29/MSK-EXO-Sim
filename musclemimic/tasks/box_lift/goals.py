import numpy as np
from musclemimic.tasks.dumbbell_lift.goals import PayloadGoalTrajMimic

class BoxFrictionGoal(PayloadGoalTrajMimic):
    """Append object state without moving any of the original PPO inputs."""
    def _init_from_mj(self, env, model, data, current_obs_size):
        super()._init_from_mj(env, model, data, current_obs_size)
        self.box_body = model.body('payload').id
        self.box_q = int(model.jnt_qposadr[model.joint('payload_free').id])
        self.box_v = int(model.jnt_dofadr[model.joint('payload_free').id])
        self.palm_sites = np.array([model.site('box_palm_'+s).id for s in ['r', 'l']])

    @property
    def dim(self):
        return super().dim + 19

    def get_obs_and_update_state(self, env, model, data, carry, backend):
        obs, carry = super().get_obs_and_update_state(env, model, data, carry, backend)
        a, v = self.box_q, self.box_v
        extra = backend.concatenate((data.qpos[a:a+3]-data.qpos[:3],
            data.qpos[a+3:a+7], data.qvel[v:v+6],
            (data.site_xpos[self.palm_sites]-data.qpos[a:a+3]).reshape(-1)))
        return backend.concatenate((obs, extra)), carry

BoxFrictionGoal.register()
