"""A free box lifted by compressive hand contacts and Coulomb friction only."""
import mujoco
import numpy as np

from .myofullbody_payload import MyoFullBodyPayloadEnv, PayloadGoalTrajMimic, PayloadMimicReward
from .myofullbody_assist import MyoFullBodyAssistEnv, MjxMyoFullBodyAssistEnv
from musclemimic.core.terminal_state_handler.enhanced_fullbody import MeanRelativeSiteDeviationWithRootTerminalStateHandler


from musclemimic.tasks.box_lift.goals import BoxFrictionGoal


from musclemimic.rewards.tasks.box_friction import BoxFrictionReward


from musclemimic.tasks.box_lift.terminal import BoxFrictionTerminal


class MyoFullBodyBoxFrictionEnv(MyoFullBodyPayloadEnv):
    def __init__(self, box_half_size=(.12, .10, .12), hand_friction=.8,
                 box_support_pos=(0., 0., .1), box_support_half_size=(.20, .20, .1), **kwargs):
        self.box_half_size = np.asarray(box_half_size, float)
        self.hand_friction = float(hand_friction)
        self.box_support_pos = np.asarray(box_support_pos, float)
        self.box_support_half_size = np.asarray(box_support_half_size, float)
        kwargs['payload_kind'] = 'box'
        kwargs.setdefault('model_option_conf',dict(iterations=20,ls_iterations=8,cone=1,
            disableflags=int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)))
        super().__init__(**kwargs)
        self.box_hand_bodies = []
        for side in ['r', 'l']:
            root = self._model.body('lunate_'+side).id
            ids = []
            for b in range(self._model.nbody):
                ancestor = b
                while ancestor and ancestor != root:
                    ancestor = int(self._model.body_parentid[ancestor])
                if ancestor == root:
                    ids.append(b)
            self.box_hand_bodies.append(ids)

    def _apply_spec_changes(self, spec):
        from musclemimic.scenes.objects.box import add_friction_box
        return add_friction_box(MyoFullBodyAssistEnv._apply_spec_changes(self, spec), self)


class MjxMyoFullBodyBoxFrictionEnv(MyoFullBodyBoxFrictionEnv, MjxMyoFullBodyAssistEnv):
    mjx_enabled = True


BoxFrictionGoal.register()
BoxFrictionReward.register()
BoxFrictionTerminal.register()
MyoFullBodyBoxFrictionEnv.register()
MjxMyoFullBodyBoxFrictionEnv.register()
