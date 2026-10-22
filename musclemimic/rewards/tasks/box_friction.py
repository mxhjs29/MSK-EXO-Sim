import numpy as np
from musclemimic.rewards.tasks.payload import PayloadMimicReward

class BoxFrictionReward(PayloadMimicReward):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.hands = np.array([env._model.site('box_palm_'+s).id for s in ['r', 'l']])
        self.palms = np.array([env._model.body('lunate_'+s).id for s in ['r', 'l']])
        self.box_body = env._model.body('payload').id

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend):
        reward, carry, terms = super().__call__(state, action, next_state, absorbing, info, env, model, data, carry, backend)
        axis = data.xmat[self.box_body].reshape(3, 3)[:, 0]
        # Both anatomical local z axes should point from the left face to the right face.
        z = data.xmat[self.palms].reshape(2, 3, 3)[:, :, 2]
        align = backend.mean(backend.sum(z * (-axis), axis=-1))
        bonus = .15 * backend.exp(-4 * (1-align))
        reward = reward + bonus
        return reward, carry, dict(terms, reward_palm_alignment=bonus, reward_total=reward)

BoxFrictionReward.register()
