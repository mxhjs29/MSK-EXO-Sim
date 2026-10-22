"""Composition for new rewards; legacy task subclasses preserve their exact order."""


class RewardComposer:
    """Evaluate common reward once, then stateless task contributions.

    Each extra takes the same positional transition arguments with the updated
    carry and returns (contribution, metrics). Clipping belongs to the common
    reward, matching existing task rewards that add bonuses after base clipping.
    """
    def __init__(self, common, extras=()):
        self.common, self.extras = common, tuple(extras)

    def __getattr__(self, name):
        return getattr(self.common, name)

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend):
        reward, carry, terms = self.common(state, action, next_state, absorbing, info, env, model, data, carry, backend)
        terms = dict(terms)
        for extra in self.extras:
            value, metrics = extra(state, action, next_state, absorbing, info, env, model, data, carry, backend)
            if set(metrics) & set(terms):
                raise ValueError("Task reward metric names must be unique")
            reward = reward + value
            terms.update(metrics)
        terms["reward_total"] = reward
        return reward, carry, terms
