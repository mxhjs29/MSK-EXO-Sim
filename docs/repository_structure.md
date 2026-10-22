# 公共仓库结构与维护边界

本仓库维护仿真组件组合、肌骨 PPO 训练、任务奖励、数据处理和可视化。`repository_scope.json` 与 `scripts/check_repository_scope.py` 定义并检查发布范围。

```text
models/                         三个模型包，保留资源与注册表的内部布局
  myo_sim/                      肌骨模型
  assist_sim/                   外骨骼、机器人、连接配置
  myoassist_terrains/            地形、仓库场景、纹理
loco_mujoco/                     项目修改过的框架、轨迹与重映射基础
musclemimic/
  scenes/                       场景描述、组合、对象构建、编译后所有权
  agents/                       agent/policy 标识、动作归属、观测视图
  environments/                 现有环境与共享 CPU 世界接口
  core/                         MJX、wrapper、真实接触传感器
  tasks/                        模仿、步态、哑铃、箱子的目标与终止条件
  rewards/common/               各任务共享的 MimicReward
  rewards/tasks/                哑铃、摩擦箱子、Gait120 附加奖励
  controllers/                  物理力矩接口、通用波形、可选独立募集
  algorithms/                   原 PPO、MLP/ResNet/MoE；多智能体算法扩展位置
  rl_core/                      rollout buffer 与现有训练数据结构
  data/                         HF 下载、版本/校验、轨迹、预处理和传感记录
  retargeting/                  MSK/GMR/C3D/Gait120 重映射
  runner/workflows/             原 experiments 的训练与评估流程
  runner/jobs/                  后台监督、资源配置
  runner/evaluation/            checkpoint 评估入口
  visualization/                viewer、C3D/轨迹展示
  cli/                          训练、场景编译等入口
configs/                        场景、任务、奖励、算法、数据、完整实验组合
scripts/                        启动脚本、GPU smoke、发布检查与导出
examples/                       小型组合与共享世界示例
tests/                         单元与回归检查、小型合成样例
docs/                          结构、协作、迁移说明
fullbody/                       全身训练旧命令的兼容入口和配置链接
```

`fullbody/`、`musclemimic/experiments/`、旧 viewer 路径只承担兼容职责。新代码应直接使用规范位置，不在兼容入口添加业务逻辑。兼容模块指向同一模块对象，旧 monkeypatch 与序列化导入仍可找到对象。

## 场景、任务、控制器与策略

`SceneSpec` 选择物理实体、地形与对象，`build_scene()` 编译模型并给出 `ComponentOwnership`。现有 `build_msk_device_terrain_spec()` 仍可使用。物理组合不决定策略网络或奖励。

当前场景构建器支持一个 MSK、至多一个设备/机器人和一个负载对象；沿用原模型命名与物理参数。组合不支持的条件会显式报错。`AttachmentSpec` 的 `model_defined` 表示使用已有构建器的连接语义，不是任意连接求解器。哑铃是右手刚性子体，箱子是依靠双手碰撞和摩擦的自由物体；历史理想约束箱子有独立 `ideal_box` 名称。

```bash
.venv/bin/python -m musclemimic.cli.compose configs/scenes/fullbody_hippo.yaml
.venv/bin/python -m musclemimic.cli.compose configs/scenes/fullbody_dumbbell.yaml
.venv/bin/python -m musclemimic.cli.compose configs/scenes/fullbody_box.yaml
.venv/bin/python -m musclemimic.cli.train_fullbody --config-name=conf_fullbody_assist
.venv/bin/python -m musclemimic.runner.evaluation.fullbody --help
```

从仓库根目录执行训练。数据集和预训练 checkpoint 需要另行取得并配置路径；普通代码导入和合成数据 smoke 不要求这些文件。`musclemimic-train`、`musclemimic-evaluate`、`musclemimic-compose` 是重新安装公共包后的对应命令。训练默认配置已选择维护中的 Assist 环境。

## 基础奖励与任务额外奖励

公共奖励实现为 `rewards/common/imitation.py`；公共权重由 `configs/rewards/common/fullbody.yaml` 管理，完整实验通过 Hydra 继承。任务额外奖励为 `rewards/tasks/` 下的独立类，`configs/rewards/tasks/` 提供选择和参数模板。

迁移保留既有运算顺序：基础跟踪与惩罚先按原公式合并、裁剪并更新 carry；哑铃后段肘关节跟踪、箱子夹紧/位置/速度/姿态/非手部接触等额外项随后叠加。没有借迁移改变激活惩罚、终止门槛或观测维数。新增任务可使用 `RewardComposer`，其中公共奖励只计算一次，附加项返回增量与独立指标。

`controllers/muscle_recruitment/` 是显式选择的独立控制器，与 PPO 奖励分开；默认 PPO 不自动插入募集层。`core/sensors/` 负责真实接触测量，不包含任何预测网络。

## 多智能体扩展边界

已实现 `AgentSpec`、冲突检查与动作路由，以及接收所有 agent 动作后仅推进一次的 CPU `WorldRuntime`；已有测试验证共享推进、原子重置和所有权冲突。物理实体与 agent 不等同：被动箱子、地形、左右手、GPU 中的多个并行世界都不会自动变成额外 agent。

当前 PPO 仍为单策略训练。IPPO/MAPPO、多人体场景组合、Parallel API、JAX/Warp 多 agent rollout 尚未实现。扩展时需要按 policy 分组的参数、归一化、观测历史和 checkpoint，以及每 agent 的 reward/termination/mask。现有 `[T,B]` rollout 不能只加一个目录就当作多智能体实现。

## 本地内容

```text
local/assets/smpl/              Git LFS 管理的 SMPL/MANO 模型资源（上传例外）
local/configs/resources.json    本机 GPU UUID 分配
outputs/、datasets/、converted_datasets/、wandb/、.venv*/
```

其中 `local/assets/smpl/` 按用户要求纳入 Git 上传范围，PKL/NPZ 模型使用 Git LFS；`musclemimic/models/smpl` 兼容链接也纳入 Git。其余 local 内容、数据集与输出仍被忽略。模型资源继续排除于公共 wheel/sdist，Git 克隆后通过兼容链接访问。

GPU 可用 `MUSCLEMIMIC_GPU_<ROLE>` 覆盖，或写入 `local/configs/resources.json`。本机 UUID 不写进公共 Python 源码。数据集通过 `configs/datasets/manifest.yaml` 描述 HF 仓库与固定版本；没有自动上传或下载数据。

## 协作与发布

每个分支围绕一个任务、控制器或模型组件改动；奖励变化与纯迁移分开评审。模型包内部 XML/mesh/连接 YAML 保留原结构，禁止随手合并成一个巨大模型文件。公共接口改变时保留兼容入口并加行为回归。

```bash
make test
make lint
.venv/bin/python scripts/check_repository_scope.py
uv build --out-dir outputs/public_dist
.venv/bin/python scripts/check_repository_scope.py outputs/public_dist/*
.venv/bin/python scripts/export_public_tree.py /tmp/musclemimic-public
```

导出工具同时包含未提交的新源码，避免只导出旧 HEAD 而丢失负载任务；目标必须为空。上传模型前应满足目录内 LICENSE.txt 的第三方分发授权要求；许可原文随文件保留。新机器使用 Git LFS 拉取模型，例如 `git lfs pull`。发布前仍要检查差异和仓库 URL。当前 origin 为 `mxhjs29/human-robot-sim-field`，已按用户授权推送。Contributor 可按 CONTRIBUTING.md 的模块边界提交 PR，账号/团队分工需另行配置。

实际迁移证据见 [2026-10-13 迁移与验证](repository_migration_20261013.md)。

人体模型通过统一 Assist 环境的 `msk_key` 选择。`myoarms` 使用当前双臂肌骨模型；重映射与训练复用公共入口，任务需匹配对应的上肢参考轨迹、观测、奖励和终止条件。
