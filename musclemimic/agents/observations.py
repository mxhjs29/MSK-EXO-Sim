"""Explicit actor/critic observation views for future policy grouping."""
def observation_views(observation, agent):
    actor = observation[list(agent.observation_indices)] if agent.observation_indices else observation
    critic = observation[list(agent.critic_indices)] if agent.critic_indices else actor
    return actor, critic
