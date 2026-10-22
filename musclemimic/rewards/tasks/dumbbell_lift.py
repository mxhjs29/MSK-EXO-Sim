from musclemimic.rewards.tasks.payload import PayloadMimicReward

class LiftRefinementReward(PayloadMimicReward):
    def __init__(self,env,**kwargs):
        super().__init__(env,**kwargs)
        self.elbow_dof=int(env._model.jnt_dofadr[env._model.joint('elbow_flexion_r').id])

    def __call__(self,state,action,next_state,absorbing,info,env,model,data,carry,backend):
        reward,carry,terms=super().__call__(state,action,next_state,absorbing,info,env,model,data,carry,backend)
        ref=env.th.get_current_traj_data(carry,backend)
        error=data.qpos[self.elbow_adr]-ref.qpos[self.elbow_adr]
        velocity_error=data.qvel[self.elbow_dof]-ref.qvel[self.elbow_dof]
        phase=carry.traj_state.subtraj_step_no/(env.th.len_trajectory(carry.traj_state.traj_no)-1)
        late=1.+.5*backend.clip((phase-.45)/.15,0.,1.)
        broad=.7*late/(1.+(error/.35)**2)
        velocity=.1/(1.+(velocity_error/2.)**2)
        reward=reward+broad+velocity
        return reward,carry,dict(terms,reward_elbow_broad=broad,reward_elbow_velocity=velocity,reward_total=reward)

LiftRefinementReward.register()
