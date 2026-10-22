# MSK + Robot/Device + Terrains 接入 MuscleMimic 训练框架

## 1. 目标与范围

本文说明如何把 `myo_sim`、`assist_sim`、后续机器人模型与
`myoassist_terrains` 生成的组合模型接入 MuscleMimic。组合形式至少包括：

```text
MSK + Terrains
MSK + Device + Terrains
MSK + Robot + Terrains
```

其中 Robot 可以是人形机器人、机械臂或其他独立机器人，不应假设它穿戴在
MSK 上，也不应强制它具有 Device 的人体连接点、绑带、同构关节或 equality。
下文在描述通用接口时以 `robot` 表示所有非 MSK 主动组件；已有穿戴式
Device 是 Robot 的一种来源和组装方式，而不是通用组合流程的结构前提。

计划分为两个阶段：

1. 单智能体阶段：只训练 MSK 的肌肉控制策略。Device 参与质量、惯量、碰撞、关节和约束动力学，但 Device actuator 不属于策略动作。
2. 多智能体阶段：MSK 和 Device 各自使用独立策略，在同一个 MuJoCo/MJX 世界中同步控制并同时训练。

当前重点应先完成第一阶段。第二阶段涉及环境接口、PPO rollout、网络、优化器、奖励和 checkpoint 的系统性改造。

### 1.1 不可破坏的设计约束

1. MSK 的权威模型来源是 `myo_sim`。当 `myo_sim` 与 MuscleMimic 中的模型、
   body、joint、actuator、sensor 或 site 名称冲突时，以当前 `myo_sim` 为准；
   MuscleMimic 的轨迹、配置和代码通过显式 alias/migration 表适配，不反向修改
   `myo_sim` 名称来迎合旧模型。
2. 组合器只负责构建一个共享物理世界，不假设 Robot 与 MSK 的拓扑关系。
3. MSK 与 Robot 的 observation/action 必须按组件归属分别生成，不能在最终
   `MjSpec` 上用“除 root 外全部 joint”或“全部 actuator”代替归属判断。
4. 传给现有 MuscleMimic 单智能体训练器的主 observation/action 始终是 MSK
   observation/action；Robot observation 只能通过显式配置作为可选附加输入。
5. `device_key=None`（以及 CLI 的空字符串）是合法配置，必须能构建
   `MSK + Terrains`，不创建空 Device、伪 actuator 或伪 observation。

## 2. 当前代码路径

### 2.1 MSK 模型来源

当前 `assist_sim` 使用 `myo_sim` 的动态组装模型：

```python
import myo_sim

spec = myo_sim.load_spec("myofullbody")
```

对应模型定义位于：

```text
models/myo_sim/myo_sim/models/
models/myo_sim/myo_sim/build/compose.py
```

`myofullbody` 是运行时构建的 `MjSpec`，不是一个长期维护的单一静态 XML。

### 2.2 Device 组装

`assist_sim.load_combined(msk_key, device_key)` 会：

1. 调用 `myo_sim` 构建 MSK `MjSpec`；
2. 根据 Device YAML 添加 body、joint、actuator、sensor、contact 和 equality；
3. 编译为 `MjModel`；
4. 可选择导出组合 XML。

Device 配置位于：

```text
models/assist_sim/assist_sim/models/<Device>/<Level>config.yaml
```

### 2.3 Terrain 组装

`models/compose_viewer.py` 当前把 `myoassist_terrains` 生成的 terrain asset 和 worldbody geom 注入组合 XML，然后编译模型。其实际拼接顺序应作为训练 builder 的参考实现：

1. 有 Device 时调用 `load_combined(msk_key, device_key, export_xml=...)`；
2. Device 为空时调用 `load_msk(msk_key, export_xml=...)`；
3. 解析基础 XML，并把其中相对 asset 路径绝对化；
4. 用 `build_terrain(config_from_dict(...), output_dir=...)` 生成 terrain；
5. 把 terrain 的 `<asset>` 子项和 `<worldbody>` 子项注入基础模型；
6. 明确设置注入 terrain geom 的 `contype=1`、`conaffinity=1`；
7. 最后只编译一次最终模型。

该文件面向 viewer。训练环境不应依赖 viewer 模块，组装代码需要提取为独立、无 GUI 的 builder，由训练和 viewer 共同调用。

### 2.4 MuscleMimic 环境与训练入口

全身环境：

```text
musclemimic/environments/humanoids/myofullbody.py
```

训练入口：

```text
fullbody/experiment.py
fullbody/conf_fullbody.yaml
musclemimic/runner/engine.py
musclemimic/algorithms/ppo/
```

当前 runner 使用一个环境观测、一个扁平动作向量、一套 Actor-Critic 和一个 PPO optimizer。

## 3. 当前模型与 MyoFullBody 环境的差异

### 3.1 统一模型来源

当前组合环境的 MSK 来自 `myo_sim.load_spec(msk_key)`，并由统一场景入口
装配可选设备与地形。`MyoFullBody` 是 Assist 环境复用的实现基类；
模型规格由 Assist builder 提供。

### 3.2 当前模型尺寸变化

本地检查结果如下：

| 模型 | `nq` | `nv` | `nu` | 默认状态观测维度 | 默认动作维度 |
|---|---:|---:|---:|---:|---:|
| 当前 `myo_sim myofullbody` | 129 | 128 | 416 | 259 | 416 |
| `myofullbody + HipExoskeleton_L1 + flat` | 131 | 130 | 418 | 263 | 418 |

HipExoskeleton 增加两个 hinge joint 和两个 motor actuator。旧策略和 checkpoint 的输入、输出维度不能直接假设不变。

### 3.3 根节点命名和物理根 body

当前模型中：

```text
free joint 名称：root
free joint 所属 body：Full Body
人体 pelvis body：pelvis
上身参考 body：torso
```

`MyoFullBody.root_free_joint_xml_name == "root"` 可以继续使用。

`root_body_name == "pelvis"` 当前用于地形观测中心、domain randomization 等人体语义，不能仅因为 free joint 位于 `Full Body` 就机械改名。若某个模块需要自由根本身，应显式使用 root joint 的 `jnt_bodyid`，不要复用语义上的 `root_body_name`。

### 3.4 手部名称统一与 finger 删除解耦

当前环境只在 `disable_fingers=True` 的删除分支中手写了一组左手 actuator 名称，
例如：

```text
FDS5_left
```

当前 `myo_sim` 模型实际使用：

```text
FDS5_l
```

现有代码会删除 `FDS5_tendon_l`，但保留引用该 tendon 的 `FDS5_l` actuator，随后编译失败：

```text
transmission target 'FDS5_tendon_l' not found in actuator FDS5_l
```

手部命名统一不能依赖 `disable_fingers`。模型载入后，无论该选项是 `true` 还是
`false`，环境内部、observation/action spec、轨迹映射、checkpoint metadata 和
日志都必须使用 `myo_sim` 的规范名称，例如左手统一使用 `_l`，不能继续生成或
保存 `_left` 名称。

处理顺序固定为：

1. 从 `myo_sim.load_spec(msk_key)` 获取权威的 hand joint、tendon、actuator 名称；
2. 对旧 MuscleMimic 配置或轨迹中的名称执行一次输入迁移，如
   `FDS5_left -> FDS5_l`；
3. 用迁移后的规范名称构造组件 metadata 和 observation/action spec；
4. 仅当 `disable_fingers=True` 时，根据规范名称和实际引用关系删除 finger
   actuator、tendon、joint 及相关 observation/action；
5. `disable_fingers=False` 时保留完整手部，但仍只暴露 `myo_sim` 规范名称。

alias 表只允许用于读取旧配置、旧轨迹或旧 checkpoint，且应集中维护，例如：

```python
LEGACY_HAND_NAME_ALIASES = {
    "FDS5_left": "FDS5_l",
    # 其余映射由 myo_sim 的实际 hand 元素清单补齐。
}
```

新生成的模型、缓存和 checkpoint 不得写回旧名称。删除逻辑也不要长期维护一份
独立的 muscle 名称列表；应从 `myo_sim` hand 组件清单或 actuator 到 tendon/joint
的引用关系推导删除集合，先删除 actuator，再删除不再被引用的 tendon 和 joint。

### 3.5 muscle observation 错误地遍历所有 actuator

当前 `_get_observation_specification()` 对所有 actuator 添加 muscle length、velocity、force、excitation 和 activation 观测。

组合模型中 Device motor 不是 muscle actuator。尤其 `ActuatorActivation` 读取 `data.act`，而 motor 没有 muscle activation state。开启 activation 观测后，组合模型在 Reset 时会发生越界：

```text
IndexError: index 416 is out of bounds for axis 0 with size 416
```

应先筛选 muscle actuator：

```python
muscle_actuators = [
    actuator
    for actuator in spec.actuators
    if actuator.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
]
```

所有 `muscle_*` 观测只遍历 `muscle_actuators`。Device motor 的 joint、velocity、force、ctrl 等状态使用独立命名的观测项。

### 3.6 默认动作会自动包含 Device actuator

当前 `_get_action_specification()` 返回全部 actuator 名称。组合 HipExoskeleton 后，动作会从 416 维变成 418 维：

```text
416 个 MSK muscle actuator
2 个 Device motor actuator
```

这不满足“只训练 MSK”的目标。单智能体阶段必须把动作列表限制为 muscle actuator。

### 3.7 足底 touch sensor

环境期望以下四个 sensor 名称：

```text
r_foot
r_toes
l_foot
l_toes
```

基础 `myofullbody` 提供这些名称。Anatomics、Humotech、STRIDE 等设备会删除人体原传感器，再以同名 sensor 绑定设备鞋底 site，因此环境层可以保持名称稳定。

环境仍应在构造观测前检查 sensor 是否存在。对于可选传感器可以跳过；对于训练必需传感器应立即给出明确错误。

不能依赖 `sensordata` 的固定索引，因为 Device sensor 会改变最终排列。必须按 sensor 名称解析地址。

### 3.8 mimic site 添加不是幂等的

环境目前无条件给 body 添加 mimic site。若传入的组合 spec 已经包含这些 site，可能出现重名或重复定义。

添加前应检查：

```python
body = spec.body(body_name)
if body is None:
    raise ValueError(...)

if spec.site(site_name) is None:
    body.add_site(...)
```

当前 `myo_sim myofullbody` 仍包含现有 `body2sites_for_mimic` 所需的全部人体 body，因此映射本身暂时可用。

### 3.9 keyframe、qpos 与训练 Reset

纯 MSK 轨迹有 129 个 qpos；HipExoskeleton 组合模型有 131 个 qpos。不能把旧轨迹数组按位置直接复制到新模型。

`compose_viewer.py` 中对 `model.qpos0`、`model.key_qpos` 和 `data.qpos` 的修改只作用于该次编译出的 viewer 模型。它不自动进入训练环境所使用的源 `MjSpec` 或导出 XML。

Device 初值应在以下位置之一按 joint 名称定义：

1. Device YAML 的 `keyframe_overrides`；
2. 组合 builder 在编译前修改 keyframe；
3. 训练用 initial-state handler 在每次 Reset 后设置 Device qpos/qvel。

推荐优先使用 Device YAML 的 `keyframe_overrides`，同时在 initial-state handler 中保留校验。

### 3.10 Terrain 与 MJX/Warp

Terrain geom 可以进入 CPU MuJoCo 模型，但训练前仍需验证：

- 所选 terrain 类型是否受当前 MJX/Warp backend 支持；
- hfield、mesh、contact pair 和 equality 的 backend 行为；
- `nconmax`、`njmax` 是否足够；
- Device 与 MSK 自碰撞是否导致接触数激增；
- JAX 非 Warp 分支当前会关闭大量非 floor/ground geom 的碰撞，不适合直接用于复杂 terrain；
- 策略是否需要 terrain height map、坡度或局部几何观测。

## 4. 第一阶段：只训练 MSK，Device 不训练

### 4.1 期望控制结构

```text
MSK policy action
      |
      v
416 个 muscle actuator --------+
                               |
Device 固定/规则控制 ----------+--> 同一个 MuJoCo/MJX physics step
                               |
Terrain contact ---------------+
```

Device body、joint、mass、inertia、contact、sensor 和 equality 都保留。Device actuator 不进入 MSK policy 的 action space。

### 4.2 提取无 GUI 的组合 builder

新增共享模块，例如：

```text
musclemimic/models/composed_spec.py
```

建议接口不要只返回裸 `MjSpec`，还要返回组件归属。否则模型完成拼接后无法可靠
区分同类型的 MSK actuator 与机器人 muscle actuator，也无法支持名称前缀不同的
人形机器人和机械臂：

```python
@dataclass(frozen=True)
class ComponentNames:
    joints: tuple[str, ...] = ()
    actuators: tuple[str, ...] = ()
    sensors: tuple[str, ...] = ()
    bodies: tuple[str, ...] = ()

@dataclass(frozen=True)
class CompositionResult:
    spec: mujoco.MjSpec
    components: Mapping[str, ComponentNames]  # "msk", "robot", "terrain"

def build_msk_device_terrain_spec(
    msk_key: str,
    device_key: str | None = None,
    terrain: dict | None = None,
) -> CompositionResult:
    ...
```

职责：

1. 从 `myo_sim.load_spec(msk_key)` 获取 MSK，并在拼接前记录 MSK 元素名称；
2. `device_key is not None` 时使用 `assist_sim` 应用 Device 配置，并把新增元素
   记录为 `robot` 归属；为空时跳过该步骤；
3. 按 `compose_viewer.py` 的 asset 处理和 XML 注入顺序加入 Terrain；
4. 返回未编译的最终 `MjSpec` 和不可变的组件归属 metadata；
5. 不启动 viewer，不修改运行时 `MjData`。

后续接入独立机器人时，新增 `robot_builder`/`robot_key` 输入即可。它只需要把
机器人插入 worldbody，并返回自己的元素清单；组合器不得自动建立 MSK-Robot
连接。穿戴式 Device 的 attachment/equality 仍由它自己的 YAML/builder 定义。

归属应在每个组件加入前后通过稳定 ID/显式 builder 返回值记录，不能依赖
`exo_`、`robot_` 等名称前缀。若最终 MJCF 要求全局名称唯一，组合器应对非 MSK
组件使用确定性 namespace，并保存原名到最终名的映射。MSK 名称保持 `myo_sim`
原名；发生冲突时重命名 Robot/Device 侧并同步修正其引用。

如果 `assist_sim` 暂时只能导出 XML，可沿用 `compose_viewer.py` 的临时 XML、
asset 路径绝对化、terrain 注入流程，再用最终 XML 创建 `MjSpec`；最终应提供直接
返回组合 `MjSpec` 的公开 API，避免重复序列化和资源路径问题。训练 builder 与
viewer 必须调用同一个组合函数，viewer 只额外负责创建 `MjData` 和启动 GUI。

### 4.3 新建组合训练环境类

建议新增：

```text
MyoFullBodyAssistEnv
MjxMyoFullBodyAssistEnv
```

它们继承 `MyoFullBody`，接收：

```python
msk_key="myofullbody"
device_key="HipExoskeleton_L1"  # 也可以为 None
terrain={"terrain": "flat"}
device_control_mode="zero"
```

构造流程：

```python
composition = build_msk_device_terrain_spec(msk_key, device_key, terrain)
self._components = composition.components
super().__init__(spec=composition.spec, ...)
```

不要把 Device/Terrain 条件直接写死在 `MyoFullBody` 中，以便基础环境继续支持纯 MSK。
`device_key=None` 时仍执行 terrain 注入，环境退化为 `MSK + Terrains`；这一路径
必须纳入常规测试，而不是仅作为异常 fallback。

### 4.4 分别定义 MSK 与 Robot 动作空间

动作归属先来自 `CompositionResult.components`，再在各自集合内做类型校验：

```python
def build_action_specs(result: CompositionResult) -> tuple[list[str], list[str]]:
    msk_names = set(result.components["msk"].actuators)
    robot_names = set(result.components.get("robot", ComponentNames()).actuators)
    msk_actions = [
        actuator.name for actuator in result.spec.actuators
        if actuator.name in msk_names
        and actuator.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    ]
    # Robot 可以使用 motor、position、velocity 或 muscle actuator。
    robot_actions = [
        actuator.name for actuator in result.spec.actuators
        if actuator.name in robot_names
    ]
    return msk_actions, robot_actions
```

HipExoskeleton 组合环境中，MSK policy 动作维度应保持 416，而不是 418。
Robot action spec 即使第一阶段不训练也必须生成并校验；`device_key=None` 时它是
空列表。这样后续切换为人形机器人或机械臂时不需要改变共享 physics 接口。

Device actuator 的处理模式应显式配置：

- `zero`：Device motor `ctrl=0`，表现为无主动助力；
- `constant`：固定控制值；
- `prescribed`：按时间、相位或人体状态运行规则控制器；
- `pretrained`：加载冻结的 Device policy，只训练 MSK policy。

当前阶段推荐先使用 `zero` 验证模型和训练链路，再使用 `prescribed` 或冻结策略。未进入 `actuation_spec` 的 actuator 默认不会被 MSK action 写入，但 Reset 时仍应显式初始化其 `ctrl`，避免残留状态。

### 4.5 分别定义 MSK 与 Robot 观测空间

MSK actor 最少保留：

- 人体 root pose/velocity；
- 全部人体 joint position/velocity；
- muscle length、velocity、force、excitation、activation；
- 四个足底 touch sensor；
- mimic goal/phase。

不能让 Device/Robot joint 被当前 `q_all_pos`、`dq_all_vel` 自动包含。应由组件
metadata 分别生成两套 observation spec：

```python
msk_joint_names = tuple(
    name for name in result.components["msk"].joints
    if name != root_free_joint_xml_name
)
robot_joint_names = result.components.get("robot", ComponentNames()).joints

msk_observation_spec = build_current_musclemimic_observations(
    spec=result.spec,
    joint_names=msk_joint_names,
    actuator_names=msk_muscle_actuator_names,
)
robot_observation_spec = [
    ObservationType.JointPosArray("robot_qpos", robot_joint_names),
    ObservationType.JointVelArray("robot_qvel", robot_joint_names),
] if robot_joint_names else []
```

MSK observation 保持当前 MuscleMimic 的语义和顺序：MSK free-root、MSK joint
position/velocity、可选 muscle state、足底 touch，以及 mimic goal/phase。所有
muscle observation 仅遍历归属于 MSK 且 `dyntype == mjDYN_MUSCLE` 的 actuator。

Robot observation 第一版只包含 Robot 各关节的角度与角速度。free joint、ball
joint 不能假设“一关节一标量”：实现应使用 MuJoCo 的 qpos/qvel address 和宽度
生成稳定切片；后续再扩展 motor、力矩、接触和任务观测。

传给现有 MuscleMimic runner 的输入按如下规则组装：

```python
policy_observation_spec = list(msk_observation_spec)
if include_robot_observations:
    policy_observation_spec.extend(robot_observation_spec)

actuation_spec = msk_action_spec
```

即 Robot observation 是可选附加输入，默认关闭；Robot action 永远不传入当前
MSK policy。环境可在 `info` 或结构化内部状态中保留 `robot_observation`，供记录、
规则控制器或未来多智能体 runner 使用。必须分别保存两套名称、顺序、shape 和
normalization metadata，不能只保存拼接后的总维度。

两种策略：

1. 兼容旧 checkpoint：`include_robot_observations=false`，MSK actor 只接收旧 MSK 观测；Robot 状态只记录或不使用。
2. 重新训练：`include_robot_observations=true`，在 MSK observation 后追加 Robot joint angle/velocity；checkpoint 必须记录新 schema。

如果要研究人在外骨骼下的适应性，推荐第二种。

### 4.6 Reset 与轨迹映射

Reset 必须按 joint 名称组装最终状态：

```text
轨迹中的 MSK joint qpos/qvel -> 组合模型同名 joint
Device joint                 -> Device 初始值
新增自由度                   -> qpos0 或配置值
```

HipExoskeleton 当前目标初值：

```text
exo_right_hip_flexion = -0.4
exo_left_hip_flexion  =  0.4
```

不应根据 qpos 数组尾部位置写入，应使用 `mj_name2id` 和 `jnt_qposadr`。

轨迹缓存需要记录模型签名，至少包含：

```text
msk_key
device_key
joint name/order
site name/order
nq/nv
```

模型签名变化时应重建缓存，不能复用仅凭文件名命中的旧缓存。

### 4.7 Reward 与 termination

第一阶段继续使用人体模仿奖励：

- MSK qpos/qvel tracking；
- 人体 mimic site position/orientation/velocity tracking；
- root tracking；
- muscle activation/energy penalty；
- action-rate penalty。

不要把 Device 新增 qpos 自动放进原轨迹误差，除非轨迹中有对应 Device 目标。

可增加以下 Device 相关环境代价，但它们仍反馈给 MSK policy：

- Device-human 接口约束力；
- Device 关节限位；
- Device 与人体非法碰撞；
- Device 被动阻力或固定控制器功率；
- 跌倒、异常接触和数值不稳定。

termination 的健康高度、姿态和 site 偏差仍应基于 MSK body/site，不应使用 Device body 替代。

### 4.8 配置建议

新增配置而不覆盖纯 MSK 配置，例如：

```yaml
experiment:
  env_params:
    env_name: MjxMyoFullBodyAssistEnv
    msk_key: myofullbody
    device_key: HipExoskeleton_L1
    terrain:
      terrain: flat
    device_control_mode: zero
    include_robot_observations: false
    disable_fingers: false  # 只控制是否保留手指，不影响手部名称规范
    enable_touch_sensor_observations: true
```

### 4.9 第一阶段验证顺序

1. 分别构建 `MSK + flat`、`MSK + Device + flat`，CPU 编译最终 `MjSpec`；
2. 检查所有必需元素名称，并确认 MSK 名称以 `myo_sim` 为准；
3. 检查 MSK/Robot 元素集合无交集且完整覆盖预期元素；
4. 检查 MSK 动作只包含 MSK muscle actuator，Robot 动作不进入 MSK action；
5. 分别在 `disable_fingers=false/true` 下检查手部只使用 `myo_sim` 规范名称，
   且不存在 `_left` 等旧名称泄漏；
6. 检查 MSK observation schema 与当前设置一致，Robot schema 仅含 joint qpos/qvel；
7. 分别测试 `include_robot_observations=false/true` 的 policy observation shape；
8. 执行 CPU `reset()` 和若干 `step()`；
9. 验证 Device actuator ctrl 符合配置模式；
10. 验证轨迹 joint-name 映射、旧手部名称迁移和 Device 初值；
11. 检查观测和 reward 全部为有限值；
12. 在 CPU viewer 中检查接触、约束和姿态；
13. 单环境 MJX/Warp reset/step；
14. 小规模并行 rollout；
15. 最后扩大 `num_envs` 开始 PPO 训练。

每次构建建议打印并保存：

```text
nq, nv, nu, na, nbody, nsensor, neq
MSK/Robot joint names/count
MSK/Robot action names/count
MSK/Robot observation names/shape
policy observation shape and include_robot_observations
contact/constraint capacity
```

## 5. 第二阶段：MSK 与 Device 多智能体同时训练

### 5.1 为什么不能只把动作拼接后称为多智能体

现有 PPO 可以让一个 actor 输出：

```text
[MSK action, Device action]
```

这仍是单智能体联合策略：只有一套观测、一套 reward、一套 Actor-Critic 和一套 optimizer。它适合作为联合控制基线，但不能独立训练、冻结、恢复或部署 Device 策略。

真正的双智能体训练需要：

```text
MSK actor    -> MSK muscle action
Device actor -> Device motor action
                   |
                   v
             shared physics step
                   |
          两套 observation/reward
```

### 5.2 环境接口

建议环境内部仍维护一个 `MjModel/MjxModel` 和一份共享 `Data`，对外提供多智能体结构：

```python
observations = {
    "msk": msk_observation,
    "device": device_observation,
}

actions = {
    "msk": msk_action,
    "device": device_action,
}

rewards = {
    "msk": msk_reward,
    "device": device_reward,
}
```

在 JAX/JIT 内部，为避免动态 Python dict，可以使用固定 dataclass、NamedTuple 或固定数组结构：

```python
@struct.dataclass
class MultiAgentAction:
    msk: jax.Array
    device: jax.Array
```

环境 step 流程：

1. 分别预处理两套 action；
2. 按预计算的 actuator index 写入同一个 `data.ctrl`；
3. 只调用一次共享 physics step；
4. 从同一个新状态构造两套 observation；
5. 计算两套 reward；
6. 使用共享 episode done，必要时附加 agent-specific truncation mask。

### 5.3 动作索引

环境初始化时建立固定索引：

```python
msk_action_indices = actuator IDs where dyntype == mjDYN_MUSCLE
device_action_indices = actuator IDs declared by Device config
```

不要通过名称是否包含 `exo` 推断生产逻辑。组合 builder 应把 actuator 所属角色作为 metadata 传给环境，或者由 Device 配置明确给出 actuator 列表。

控制写入：

```python
ctrl = data.ctrl
ctrl = ctrl.at[msk_action_indices].set(msk_ctrl)
ctrl = ctrl.at[device_action_indices].set(device_ctrl)
data = data.replace(ctrl=ctrl)
```

两个策略动作都建议在策略侧使用 `[-1, 1]`，再分别映射到各自 actuator 的实际 `ctrlrange`。

### 5.4 两套观测

MSK actor 可观察：

- 人体本体感觉；
- muscle state；
- 足底接触；
- mimic goal 和 phase；
- Device joint/interaction force 的必要子集；
- terrain 局部信息。

Device actor 可观察：

- Device joint position/velocity；
- motor torque、ctrl、功率；
- Device-human attachment force；
- 人体 hip/knee/ankle 状态；
- root velocity、步态 phase、足底接触；
- terrain 局部信息；
- 上一步 Device action。

应为两套观测分别维护 spec、索引和 normalization statistics，不能把同一个全局 RunningMeanStd 无条件用于两者。

### 5.5 Reward 设计

MSK reward 继续关注：

- 动作/站立/行走轨迹跟踪；
- 稳定性；
- muscle effort、activation 和动作平滑；
- 跌倒惩罚。

Device reward 可关注：

- 保持或改善人体跟踪质量；
- 降低人体 muscle effort/代谢代理量；
- 限制 motor torque、功率和峰值；
- 平滑助力；
- 限制 attachment force 和不舒适力；
- 避免关节限位和非法碰撞。

如果 Device 只奖励降低肌肉用力，它可能通过破坏动作或支撑方式投机。Device reward 必须同时包含任务完成、稳定性和安全约束。

可采用：

```text
r_msk    = tracking - muscle_effort - fall_cost
r_device = tracking - muscle_effort - motor_power - interface_force
```

也可以让两者共享团队 reward，再添加各自代价。共享 reward 更合作，但 credit assignment 更困难。

### 5.6 推荐训练算法

第一版可使用 Independent PPO：

- 两个 actor；
- 两个 critic；
- 两个 optimizer/train state；
- 两份 PPO loss；
- 同一批共享环境 transition。

由于两套策略同时更新会造成环境非平稳，稳定版本推荐 MAPPO 风格的集中训练、分散执行：

- MSK actor 只读 MSK actor observation；
- Device actor 只读 Device actor observation；
- critic 在训练时读取完整共享状态和两侧必要信息；
- 部署时两个 actor 可以独立运行。

### 5.7 PPO runner 必须修改的部分

当前以下结构都是单智能体：

- `PPOAgentConf`；
- `ActorCritic`；
- `PPOAgentState`；
- rollout transition；
- action sampling 和 log-prob；
- GAE/advantage；
- PPO minibatch loss；
- optimizer state；
- checkpoint；
- inference 和 validation。

多智能体版本至少需要：

```python
MultiAgentConf(
    msk=AgentConf(...),
    device=AgentConf(...),
)

MultiAgentTrainState(
    msk=TrainState(...),
    device=TrainState(...),
)
```

rollout 需要保存：

```text
msk_obs, msk_action, msk_log_prob, msk_value, msk_reward
device_obs, device_action, device_log_prob, device_value, device_reward
shared_done, shared_state/critic_obs
```

分别计算两侧 GAE 和 PPO loss，然后在一个 update 周期中更新两套参数。

### 5.8 Checkpoint 与独立部署

checkpoint 应按 agent 分层：

```text
checkpoint/
  shared_metadata
  msk/
    actor
    critic
    optimizer
    obs_normalizer
  device/
    actor
    critic
    optimizer
    obs_normalizer
```

metadata 至少记录：

```text
MSK/Device/Terrain 配置
两侧 observation 名称和顺序
两侧 actuator 名称和顺序
模型签名
控制频率
reward 配置
```

这样可以单独加载、冻结或部署 Device actor，也能把已训练 MSK actor 与不同 Device actor 组合测试。

### 5.9 推荐课程训练顺序

直接从随机初始化同时训练两侧策略通常不稳定。建议：

1. 纯 MSK 环境训练或加载已有 MSK policy；
2. 组合 Device，冻结 MSK policy，训练 Device policy；
3. 冻结 Device，微调 MSK 适应 Device；
4. 降低学习率，联合训练两侧；
5. 逐步增加 terrain 难度和 domain randomization；
6. 分别评估无 Device、固定 Device、训练 Device 三种条件。

这套流程可以减少两个随机策略同时探索造成的非平稳性，并保留单独部署两侧策略的能力。

## 6. 建议实施顺序

### 第一批：让组合环境可靠进入单智能体训练

1. 按 `compose_viewer.py` 提取共享组合 builder，并支持 `device_key=None`；
2. 新增 `MyoFullBodyAssistEnv`；
3. 统一 `MyoFullBody` 默认模型来源到 `myo_sim`；
4. builder 返回 MSK/Robot/Terrain 组件归属和重命名映射；
5. 统一手部名称到 `myo_sim`，并将名称迁移与 `disable_fingers` 删除逻辑解耦；
6. 分别构造 MSK/Robot observation spec；
7. 分别构造 MSK/Robot action spec，runner 只接收 MSK action；
8. 增加 `include_robot_observations` 可选开关；
9. Device control mode；
10. joint-name Reset 与轨迹映射；
11. CPU 和单环境 MJX 测试；
12. 小规模 PPO smoke test。

### 第二批：提高训练可用性

1. Device/terrain observation；
2. Device 约束力、功率和碰撞指标；
3. 模型签名与轨迹缓存失效机制；
4. backend 接触容量自动配置；
5. 多 terrain curriculum；
6. checkpoint 兼容策略。

### 第三批：多智能体训练

1. 多智能体环境数据结构；
2. 两套 observation/action normalization；
3. 双 actor/critic 网络；
4. 多智能体 rollout 与 GAE；
5. 两套 PPO loss 和 optimizer；
6. 分层 checkpoint/inference；
7. 冻结、交替训练和联合训练配置；
8. MAPPO 集中式 critic；
9. 多智能体验证与消融实验。

## 7. 完成标准

单智能体阶段完成时应满足：

- 训练模型来自统一的 `myo_sim + assist_sim + myoassist_terrains` builder；
- 支持不含 Device 的 `MSK + Terrains`；
- 组合接口不假设 Robot 为穿戴式，并能扩展独立人形机器人和机械臂；
- 名称冲突时保持 `myo_sim` 名称，其他组件确定性重命名并保存映射；
- 无论 `disable_fingers` 取值如何，手部名称均使用 `myo_sim` 规范；旧名称只在
  输入迁移层出现，不写入新模型、缓存或 checkpoint；
- Device 和 terrain 完整参与动力学；
- policy action 只包含 MSK muscle actuator；
- MSK/Robot observation 和 action 各自具有独立、可复现的 schema；
- Robot observation 默认不进入 policy，配置启用后才追加；
- Device actuator 使用明确的固定、规则或冻结策略；
- Reset 按名称正确恢复 MSK 轨迹和 Device 初值；
- touch sensor、mimic site 和 reward 均按名称解析；
- CPU 与目标 MJX/Warp backend 均可 reset/step；
- 旧轨迹和 checkpoint 的兼容策略明确；
- 小规模 PPO 能持续 rollout，观测、动作、reward 和 loss 无 NaN/Inf。

多智能体阶段完成时还应满足：

- MSK 与 Device 有独立 actor、optimizer、观测、动作和 checkpoint；
- 两侧 action 在同一个 physics step 前同步写入；
- reward 和 GAE 分别计算；
- 支持冻结任一策略、交替训练和联合训练；
- Device actor 可以从训练 checkpoint 单独导出和部署；
- 训练与验证记录两侧独立指标及共享任务指标。

## 8. 第一阶段 `myolegs` 单智能体 PPO 小规模验收（2026-09-10）

本次验收只覆盖第一阶段的最小可运行闭环：`myolegs` MSK、无 Device、flat terrain、KIT 走路轨迹，CPU JAX 后端。未处理整个 KIT 数据集，仅使用一条轨迹：

```text
KIT/11/WalkingStraightForwards01_poses
```

### 8.1 环境与轨迹检查清单

- [x] `MyoFullBodyAssistEnv(msk_key="myolegs", device_key=None, terrain={"terrain":"flat"})` 可构建；模型维度为 `nq=35, nv=34, nu=80`。
- [x] `myolegs` Assist 只注册目标模型中存在的 pelvis、躯干、头部和左右髋/膝/踝/趾跟踪点，不再强制要求上肢 body。
- [x] 走路缓存已生成并使用 Assist 语义 site：
  `/home/chenshuo/.musclemimic/caches/AMASS/msk/myolegs/a04410e825ed10b4/assist-msk-v1/gmr/KIT/11/WalkingStraightForwards01_poses.npz`。
- [x] 缓存形状为 `qpos=(500,35)`、`qvel=(500,34)`、稀疏跟踪 site 为 `11` 个；`msk_key=myolegs`、fingerprint=`a04410e825ed10b4`、profile=`assist-msk-v1`。
- [x] 真实 `ImitationFactory` 加载缓存后 reset/step 通过：观测维度 `379`，reward 为有限值（示例 `0.9953487`），无 NaN/Inf。

### 8.2 PPO smoke 清单

执行脚本：

```bash
AMASS_PATH=/home/chenshuo/workspace/musclemimic/dataset \
PYTORCH_JIT=0 JAX_PLATFORMS=cpu MUJOCO_GL=egl \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python scripts/minimal_ppo_myolegs_smoke.py
```

配置为 `num_envs=2`、`num_steps=2`、`total_timesteps=8`、单 minibatch、单 update、`[32]` 隐藏层、关闭 validation/checkpoint 保存和 wandb。验收结果：

- [x] PPO agent 初始化成功，输入维度 `379`，动作维度 `80`，参数量 `27,073`。
- [x] rollout 与 GAE/反向更新完成 `2` 次 update，最终 global timestep=`8`。
- [x] 日志未报告 observation、action、reward 或 loss 的 NaN/Inf。
- [x] 该 smoke 脚本可复现第一阶段单智能体最小训练闭环；它不是正式训练配置，正式训练需恢复更大的 rollout、网络和 validation 设置。

### 8.3 GPU 并行 smoke 验收（2026-09-10）

机器检测到 `NVIDIA GeForce GTX 1650 4GB`。原始环境中的 JAX 为 CPU 版，已在项目 `.venv` 安装与 `jax==0.7.2` 匹配的 CUDA 12 插件及运行库。运行时需要将 pip CUDA 库加入动态库搜索路径，并保留 CPU 平台供调试回调使用：

```bash
NVLIB=$PWD/.venv/lib/python3.11/site-packages/nvidia
LD_LIBRARY_PATH=$(find "$NVLIB" -mindepth 2 -maxdepth 2 -type d -printf '%p:')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
AMASS_PATH=/home/chenshuo/workspace/musclemimic/dataset \
PYTORCH_JIT=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
MUSCLEMIMIC_SMOKE_NUM_ENVS=8 MUSCLEMIMIC_SMOKE_NUM_STEPS=4 MUSCLEMIMIC_SMOKE_TOTAL_TIMESTEPS=32 \
.venv/bin/python scripts/minimal_ppo_myolegs_smoke.py
```

- [x] `jax.devices()` 返回 `CudaDevice(id=0)`，矩阵计算结果位于 CUDA device 且有限。
- [x] 8 个并行 `myolegs` 环境完成 4 步 rollout，PPO 完成 1 次更新，最终 global timestep=`32`。
- [x] GPU 测试保持 `myolegs`、无 Device、flat terrain、KIT 走路轨迹不变。
- [x] 训练日志未报告 observation、action、reward 或 loss 的 NaN/Inf。
- [x] 首次仅设置 `JAX_PLATFORMS=cuda` 时，调试 callback 因找不到 CPU device 失败；使用 `cuda,cpu` 后测试通过。该限制已记录，非模型或轨迹错误。

### 8.4 `myofullbody` / `myoarms` GPU 并行 PPO 验收（2026-09-10）

在同一 GPU 配置下分别执行：

```bash
for key in myofullbody myoarms; do
  NVLIB=$PWD/.venv/lib/python3.11/site-packages/nvidia
  LD_LIBRARY_PATH=$(find "$NVLIB" -mindepth 2 -maxdepth 2 -type d -printf '%p:')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  AMASS_PATH=/home/chenshuo/workspace/musclemimic/dataset \
  PYTORCH_JIT=0 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  MUSCLEMIMIC_SMOKE_MSK_KEY=$key MUSCLEMIMIC_SMOKE_NUM_ENVS=4 \
  MUSCLEMIMIC_SMOKE_NUM_STEPS=2 MUSCLEMIMIC_SMOKE_TOTAL_TIMESTEPS=16 \
  .venv/bin/python scripts/minimal_ppo_myolegs_smoke.py
done
```

- [x] `myofullbody`：`nq=93`、`nu=354`，输入观测维度 `755`，4 个并行环境完成 2 个 PPO update，global timestep=`16`，参数量 `60,453`。
- [x] `myoarms`：关闭手指后的 `nq=40`、`nu=64`，输入观测维度 `329`，4 个并行环境完成 2 个 PPO update，global timestep=`16`，参数量 `23,329`。
- [x] 两个测试均为 `device_key=None`、flat terrain、KIT `WalkingStraightForwards01_poses`，JAX 默认计算后端为 GPU。
- [x] 两个测试的 rollout、PPO 更新和 checkpoint 写入均成功，未出现 NaN/Inf。
- [x] 为支持 `myoarms`，补充了无 free-root joint 的 observation、GoalTrajMimic 和 TrajInitialStateHandler 路径；其走路缓存为 `qpos=(500,40)`、`qvel=(500,40)`。

### 8.5 外骨骼零助力 GPU PPO 验收（2026-09-10）

使用 `scripts/minimal_ppo_gpu_smoke.py` 做两组最小 PPO 验证。脚本现在支持通过
`MUSCLEMIMIC_SMOKE_MSK_KEY`、`MUSCLEMIMIC_SMOKE_DEVICE_KEY` 和
`MUSCLEMIMIC_SMOKE_TERRAIN` 选择组合，并强制设置：

```yaml
device_control_mode: zero
device_control_value: 0.0
terrain: {terrain: flat}
```

| 组合 | 运行时模型维度 | policy action | observation | PPO 结果 | 外骨骼 motor |
|---|---:|---:|---:|---|---|
| `myoarms + ElbowExoskeleton_L1 + flat` | `nq=42, nv=42, nu=66` | 64 | 653 | 2 updates，global timestep=8，成功 | 2 个，均为 0 |
| `myofullbody + HipExoskeleton_L1 + flat` | `nq=95, nv=94, nu=356` | 354 | 2529 | 2 updates，global timestep=8，成功 | 2 个，均为 0 |

- [x] 两组均使用 Warp CUDA 后端，设备为 `NVIDIA GeForce GTX 1650`。
- [x] 两组均完成 rollout、PPO 梯度更新和 checkpoint 写入，进程退出码为 0。
- [x] `robot_ctrl` 检查确认外骨骼执行器控制值全部为 0，未产生助力。
- [x] `myoarms` 使用 `NoTerminalStateHandler`，避免复用要求 `pelvis_mimic` 的全身终止处理器。
