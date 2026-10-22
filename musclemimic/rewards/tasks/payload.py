import numpy as np
from musclemimic.rewards.common.imitation import MimicReward

class PayloadMimicReward(MimicReward):
    """For the box, reward bilateral alignment; rigid dumbbell needs no fake grasp bonus."""
    def __init__(self, env, payload_grip_weight=.05, payload_joint_weight=.3, **kwargs):
        self.payload_grip_weight=payload_grip_weight
        self.payload_joint_weight=payload_joint_weight
        self.payload_kind=env.payload_kind
        self.elbow_adr=int(env._model.jnt_qposadr[env._model.joint('elbow_flexion_r').id])
        if self.payload_kind=='box':
            self.hands=np.array([env._model.site('S_grasp_'+s).id for s in ['r','l']])
            self.grips=np.array([env._model.site('payload_grip_'+s).id for s in ['r','l']])
            self.box_qadr=int(env._model.jnt_qposadr[env._model.joint('payload_free').id])
        self.finger_adr=np.array([env._model.jnt_qposadr[env._model.joint(n).id] for n in env.payload_finger_posture],int)
        self.finger_target=np.array(list(env.payload_finger_posture.values()))
        super().__init__(env, **kwargs)

    def __call__(self,state,action,next_state,absorbing,info,env,model,data,carry,backend):
        reward,carry,terms=super().__call__(state,action,next_state,absorbing,info,env,model,data,carry,backend)
        if self.payload_kind=='dumbbell':
            ref=env.th.get_current_traj_data(carry,backend).qpos
            error=(data.qpos[self.elbow_adr]-ref[self.elbow_adr])**2
            bonus=self.payload_joint_weight*backend.exp(-error/.20**2)
            reward=reward+bonus
            terms=dict(terms,reward_payload_joint=bonus,reward_total=reward)
        if self.payload_kind=='box':
            error=backend.sum((data.site_xpos[self.hands]-data.site_xpos[self.grips])**2,axis=-1)
            bonus=self.payload_grip_weight*backend.exp(-backend.mean(error)/(.025**2))
            reward=reward+bonus
            terms=dict(terms,reward_payload_grip=bonus,payload_grip_error_m=backend.sqrt(backend.mean(error)))
            terms['reward_total']=reward
            ref=env.th.get_current_traj_data(carry,backend).qpos
            origin=env.th.get_init_traj_data(carry,backend).qpos[:3]
            a=self.box_qadr
            target=ref[a:a+3]-backend.array([origin[0],origin[1],0.])
            position_error=backend.sum((data.qpos[a:a+3]-target)**2)
            quat_dot=backend.sum(data.qpos[a+3:a+7]*ref[a+3:a+7])
            tracking=.1*backend.exp(-position_error/.10**2)+.1*backend.exp(-(1-quat_dot**2)/.2)
            reward=reward+tracking
            terms.update(reward_payload_tracking=tracking,reward_total=reward)
        if len(self.finger_adr):
            error=backend.mean((data.qpos[self.finger_adr]-backend.asarray(self.finger_target))**2)
            bonus=.1*backend.exp(-error/.5**2)
            reward=reward+bonus
            terms=dict(terms,reward_grip_posture=bonus,reward_total=reward)
        return reward,carry,terms

PayloadMimicReward.register()
