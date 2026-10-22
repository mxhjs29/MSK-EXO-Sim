import numpy as np
from musclemimic.core.terminal_state_handler.enhanced_fullbody import MeanRelativeSiteDeviationWithRootTerminalStateHandler

class BoxFrictionTerminal(MeanRelativeSiteDeviationWithRootTerminalStateHandler):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.box_q = int(env._model.jnt_qposadr[env._model.joint('payload_free').id])
        self.depth_adr=np.array([env._model.sensor_adr[env._model.sensor('box_contact_depth_'+s).id] for s in ['r','l']])

    def _is_absorbing_compat(self, env, obs, info, data, carry, backend):
        done, carry = super()._is_absorbing_compat(env, obs, info, data, carry, backend)
        ref = env.th.get_current_traj_data(carry, backend).qpos
        a = self.box_q
        drop = data.qpos[a+2] < ref[a+2] - .15
        penetration=backend.max(-data.sensordata[self.depth_adr],initial=0.)
        return done | drop | (penetration>.003), carry

BoxFrictionTerminal.register()
