import numpy as np
from musclemimic.rewards.common.imitation import MimicReward

def synergy_score(activation, target, scale, weights, backend=np):
    """Bounded phase target reward: zero is not automatically optimal as in cone loss."""
    errors = (activation-target)/backend.asarray(scale)
    return backend.sum(backend.asarray(weights)*backend.exp(-.5*errors**2))/backend.sum(backend.asarray(weights))

class Gait120SynergyReward(MimicReward):
    def __init__(self, env, synergy_targets, synergy_weight=0., synergy_warmup_steps=10, **kwargs):
        super().__init__(env, **kwargs)
        target = np.load(synergy_targets, allow_pickle=False)
        self.targets = target['activation_proxy'].astype(np.float32)
        self.scales = target['scale'].astype(np.float32)
        self.weights = target['group_weights'].astype(np.float32)
        self.indices = np.array([env._model.actuator_actadr[env._model.actuator(str(n)).id]
                                 for n in target['group_names']], dtype=np.int32)
        self.weight = float(synergy_weight)
        self.warmup = max(1, int(synergy_warmup_steps))
        if len(self.indices) != 12 or np.any(self.indices < 0):
            raise ValueError('Expected 12 valid muscle activation addresses')

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend):
        reward, carry, terms = super().__call__(state, action, next_state, absorbing, info, env, model, data, carry, backend)
        # All on-line validation uses the same reference; held-out evaluation turns this term off.
        index = backend.clip(carry.traj_state.subtraj_step_no, 0, len(self.targets)-1)
        target = backend.asarray(self.targets)[index]
        value = synergy_score(data.act[self.indices], target, self.scales, self.weights, backend)
        ramp = backend.minimum((carry.cur_step_in_episode+1)/self.warmup, 1.)
        bonus = self.weight*ramp*value
        reward += bonus
        return reward, carry, dict(terms, reward_synergy=bonus, synergy_similarity=value,
                                   synergy_activation_rmse=backend.sqrt(backend.mean((data.act[self.indices]-target)**2)),
                                   reward_total=reward)

Gait120SynergyReward.register()
