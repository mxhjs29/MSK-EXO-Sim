现有单智能体接口由 environments/base.py 与 core/wrappers 保持兼容。后续 Parallel/JAX 多智能体适配应在此实现；不得逐 agent 重复推进同一个物理世界。
