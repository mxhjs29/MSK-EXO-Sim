"""Stage-one ideal-grip payloads; original muscle action/observation order retained.

Dumbbell is a rigid hand child: its full mass/inertia load the wrist and arm.
Box is a free rigid body connected at two hand sites, with no world support.
Neither mode claims learned finger grasping; payload/ground collision is real.
"""
import mujoco
import numpy as np
import jax.numpy as jnp
from .myofullbody_assist import MyoFullBodyAssistEnv, MjxMyoFullBodyAssistEnv
from musclemimic.core.goals.trajectory import GoalTrajMimic
from musclemimic.rewards.common.imitation import MimicReward


from musclemimic.tasks.dumbbell_lift.goals import PayloadGoalTrajMimic


from musclemimic.rewards.tasks.payload import PayloadMimicReward


class MyoFullBodyPayloadEnv(MyoFullBodyAssistEnv):
    def __init__(self,payload_kind='dumbbell',payload_mass_kg=.25,payload_grip_span_m=.21,
                 payload_mount_offset=(0.,0.,0.),payload_finger_posture=None,**kwargs):
        if payload_kind not in {'dumbbell','box'}:raise ValueError(payload_kind)
        if payload_mass_kg<=0:raise ValueError('Use positive physical payload mass')
        self.payload_kind=payload_kind
        self.payload_mass_kg=float(payload_mass_kg)
        self.payload_grip_span_m=float(payload_grip_span_m)
        self.payload_mount_offset=np.asarray(payload_mount_offset,float)
        self.payload_finger_posture=dict(payload_finger_posture or {})
        super().__init__(**kwargs)
        self.payload_body_id=self._model.body('payload').id
        self.payload_geom_ids=np.array([i for i in range(self._model.ngeom) if self._model.geom(i).name.startswith('payload_')])

    def _apply_spec_changes(self, spec):
        from musclemimic.scenes.objects.payload import add_payload
        return add_payload(super()._apply_spec_changes(spec), self)

    def set_sim_state_from_traj_data(self,data,traj_data,carry):
        # The base class recenters human root XY; the free box must receive
        # the identical translation, including random starts midway through a lift.
        if self.payload_kind=='box':
            q=traj_data.qpos.copy()
            adr=int(self._model.jnt_qposadr[self._model.joint('payload_free').id])
            q[adr:adr+2]-=self.th.get_init_traj_data(carry,np).qpos[:2]
            traj_data=traj_data.replace(qpos=q)
        return super().set_sim_state_from_traj_data(data,traj_data,carry)

    def mjx_set_sim_state_from_traj_data(self,data,traj_data,carry):
        if self.payload_kind=='box':
            adr=int(self._model.jnt_qposadr[self._model.joint('payload_free').id])
            q=traj_data.qpos.at[adr:adr+2].add(-self.th.get_init_traj_data(carry,jnp).qpos[:2])
            traj_data=traj_data.replace(qpos=q)
        return super().mjx_set_sim_state_from_traj_data(data,traj_data,carry)


class MjxMyoFullBodyPayloadEnv(MyoFullBodyPayloadEnv,MjxMyoFullBodyAssistEnv):
    mjx_enabled=True


PayloadGoalTrajMimic.register()
PayloadMimicReward.register()
