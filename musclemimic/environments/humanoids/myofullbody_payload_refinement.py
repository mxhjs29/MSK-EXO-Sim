"""Loaded tracking refinements; sensor feedback never applies forces to the box."""
import mujoco
import numpy as np

from .myofullbody_payload import PayloadMimicReward
from .myofullbody_box_friction import (MyoFullBodyBoxFrictionEnv, BoxFrictionGoal,
                                      BoxFrictionReward, BoxFrictionTerminal)
from .myofullbody_assist import MjxMyoFullBodyAssistEnv

CONTACT_SLOTS = 24


from musclemimic.core.sensors.hand_contact import add_contact_feedback


from musclemimic.core.sensors.hand_contact import ContactReadout


from musclemimic.rewards.tasks.dumbbell_lift import LiftRefinementReward


from musclemimic.tasks.box_lift.refinement_goals import BoxRefinementGoal


from musclemimic.rewards.tasks.box_lift import BoxRefinementReward


from musclemimic.rewards.tasks.box_lift import BoxPoseRefinementReward


from musclemimic.tasks.box_lift.refinement_terminal import BoxRefinementTerminal


class MyoFullBodyBoxRefinementEnv(MyoFullBodyBoxFrictionEnv):
    def _apply_spec_changes(self,spec):
        spec=super()._apply_spec_changes(spec);add_contact_feedback(spec);return spec


class MjxMyoFullBodyBoxRefinementEnv(MyoFullBodyBoxRefinementEnv,MjxMyoFullBodyAssistEnv):
    mjx_enabled=True


for cls in [LiftRefinementReward,BoxRefinementGoal,BoxRefinementReward,BoxPoseRefinementReward,BoxRefinementTerminal,
            MyoFullBodyBoxRefinementEnv,MjxMyoFullBodyBoxRefinementEnv]:cls.register()
