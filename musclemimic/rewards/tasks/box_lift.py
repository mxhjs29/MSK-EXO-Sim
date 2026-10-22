from musclemimic.rewards.tasks.box_friction import BoxFrictionReward
from musclemimic.core.sensors.hand_contact import ContactReadout

class BoxRefinementReward(BoxFrictionReward):
    def __init__(self,env,**kwargs):
        super().__init__(env,**kwargs)
        self.contact_readout=ContactReadout(env._model)
        self.weight=env.payload_mass_kg*9.81
        self.squeeze_target=1.5*self.weight/(2*env.hand_friction)
        self.box_v=int(env._model.jnt_dofadr[env._model.joint('payload_free').id])

    def __call__(self,state,action,next_state,absorbing,info,env,model,data,carry,backend):
        reward,carry,terms=super().__call__(state,action,next_state,absorbing,info,env,model,data,carry,backend)
        f=self.contact_readout(data,backend)
        pressure=backend.clip(f['normal']/self.squeeze_target,0.,1.)
        bilateral=backend.sqrt(pressure[0]*pressure[1])
        squeeze=.75*bilateral
        ref=env.th.get_current_traj_data(carry,backend)
        origin=env.th.get_init_traj_data(carry,backend).qpos[:2]
        a=self.box_qadr;target=ref.qpos[a:a+3]-backend.concatenate((origin,backend.zeros(1)))
        position=backend.sum((data.qpos[a:a+3]-target)**2)
        velocity=backend.sum((data.qvel[self.box_v:self.box_v+3]-ref.qvel[self.box_v:self.box_v+3])**2)
        tracking=1.0/(1.+position/.08**2)+.25/(1.+velocity/.5**2)
        grip_error=backend.sum((data.site_xpos[self.hands]-data.site_xpos[self.grips])**2,axis=-1)
        capture=.3*backend.min(1./(1.+grip_error/.05**2))
        lifted=backend.clip((target[2]-(env.box_support_pos[2]+env.box_support_half_size[2]+env.box_half_size[2]))/.04,0.,1.)
        support=.3*lifted*bilateral*backend.clip(backend.sum(f['friction_z'])/self.weight,0.,1.)
        other=backend.linalg.norm(f['other'])/self.weight
        penalty=.8*backend.minimum(other,3.)+.4*backend.clip(f['other'][2]/self.weight,0.,3.)
        excess=.05*backend.sum(backend.clip(f['normal']/self.squeeze_target-3.,0.,5.)**2)
        reward=reward+squeeze+tracking+capture+support-penalty-excess
        return reward,carry,dict(terms,reward_squeeze=squeeze,reward_box_dense_tracking=tracking,
            reward_grip_capture=capture,reward_friction_support=support,penalty_nonhand_contact=penalty,
            penalty_excess_squeeze=excess,reward_total=reward)

class BoxPoseRefinementReward(BoxRefinementReward):
    """Add a smooth orientation target to the friction-grip box objective."""
    def __call__(self,state,action,next_state,absorbing,info,env,model,data,carry,backend):
        reward,carry,terms=super().__call__(state,action,next_state,absorbing,info,env,model,data,carry,backend)
        ref=env.th.get_current_traj_data(carry,backend)
        a=self.box_qadr
        actual=data.qpos[a+3:a+7]
        target=ref.qpos[a+3:a+7]
        actual=actual/backend.maximum(backend.linalg.norm(actual),1e-8)
        target=target/backend.maximum(backend.linalg.norm(target),1e-8)
        angle=2.*backend.arccos(backend.clip(backend.abs(backend.sum(actual*target)),0.,1.))
        orientation=.8/(1.+(angle/.40)**2)
        reward=reward+orientation
        return reward,carry,dict(terms,reward_box_orientation=orientation,error_box_orientation_rad=angle,
            reward_total=reward)

BoxRefinementReward.register()

BoxPoseRefinementReward.register()
