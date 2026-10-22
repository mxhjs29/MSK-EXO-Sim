# MSK Assist 真实数据重映射修改方案

## 1. 目标

本方案覆盖“原始数据下载 → 批量重映射 → 缓存 → 训练加载”的完整流程，将
AMASS、KIT、C3D 等人体动作数据适配到 `MyoFullBodyAssistEnv` 的 MSK 组件。

1. 使用当前 Assist 环境及其 MSK 组件作为重映射权威目标；
2. 重映射目标由 `msk_key` 决定，模型直接来自 `myo_sim`；
3. 同一个 MSK 的重映射结果不受 `device_key` 是否为空、Device 类型、Robot 类型
   或 Terrain 类型影响；
4. `myofullbody`、`myoarms`、`myolegs` 使用同一批处理入口；其中
   `myoarms/myolegs` 采用已确认的方案 A，以 Assist 的 `myofullbody` MSK 轨迹
   作为中间表示，再执行模型间映射；
5. XML、名称和模型拓扑均来自当前 `myo_sim`；
6. `myo_sim` 是模型名称的唯一权威来源；
7. 新缓存继续通过当前 `AMASSDatasetConf`、`ImitationFactory` 和
   `TrajectoryHandler` 自动加载，训练侧的使用方式不变。

本文只描述重映射与轨迹装载改造。Robot/Device 自身策略轨迹不属于本阶段范围。

## 2. 确定的端到端流程

### 2.1 下载原始数据

数据输入必须是未经目标模型重映射的 AMASS/KIT 源文件，而不是
`amathislab/demo_dataset` 或 `musclemimic-retargeted` 中已有的 qpos 缓存。

准备步骤：

1. 从 AMASS 官方站点申请并下载所需数据集，例如 KIT；
2. 下载许可匹配的 SMPL-H/MANO 模型；
3. 保持 AMASS 原始目录层级，不重命名 motion 相对路径；
4. 设置原始数据、SMPL 模型和输出缓存目录：

```bash
musclemimic-set-amass-path --path /path/to/amass
musclemimic-set-smpl-model-path --path /path/to/smpl
musclemimic-set-all-caches --path /path/to/converted_datasets
```

批处理启动前必须抽查源 NPZ 至少包含姿态、位移、人体形状和帧率信息。下载过程
本身不执行重映射。

### 2.2 批量重映射

统一入口建议改为：

```bash
python scripts/retarget_dataset.py \
  --model MyoFullBodyAssistEnv \
  --msk-key myofullbody \
  --retargeting-method gmr \
  --dataset KIT_KINESIS_TRAINING_MOTIONS \
  --workers 4
```

`--msk-key` 第一阶段必须支持：

```text
myofullbody
myoarms
myolegs
```

批处理仍采用独立 CPU worker 和逐 motion 缓存。单个动作失败时记录失败清单，其他
动作继续处理。`--clear-cache`、`--workers`、`--dataset` 和
`--retargeting-method` 的使用习惯保持不变。

三类目标的确定流程为：

```text
msk_key=myofullbody:
SMPL → Assist MSK:myofullbody → site/body cache

msk_key=myoarms:
SMPL → Assist MSK:myofullbody → Assist MSK:myoarms → site/body cache

msk_key=myolegs:
SMPL → Assist MSK:myofullbody → Assist MSK:myolegs → site/body cache
```

这里的中间 `myofullbody` 是由 `myo_sim` 生成、属于 Assist 模型体系的 MSK 轨迹。

### 2.3 缓存与训练使用

批处理输出新的、已经适配 `myo_sim` 名称和 topology 的轨迹缓存。用户侧仍按当前
方式选择数据集：

```yaml
experiment:
  task_factory:
    name: ImitationFactory
    params:
      amass_dataset_conf:
        dataset_group: KIT_KINESIS_TRAINING_MOTIONS
        trajectory_cache_type: sparse
```

训练仍通过：

```bash
python fullbody/experiment.py --config-name=conf_fullbody_assist
```

`ImitationFactory` 内部根据环境的 `msk_key` 解析正确缓存；用户不需要手动传缓存
文件，也不需要因为 Device/Robot 改变数据集配置。

## 3. 总体架构

重映射流程从“按环境类名选择机器人配置”改为“先生成统一全身人体表示，再按目标
MSK 组件标识选择 profile”：

```text
人体数据（AMASS / KIT / C3D）
              │
              ▼
       Source motion normalization
              │
              ▼
 RetargetProfile(myofullbody)
              │
              ▼
myo_sim.load_spec("myofullbody") ── 生成权威 source manifest
              │
              ▼
求解 Assist MSK:myofullbody 轨迹（可复用的第一阶段缓存）
              │
              ├── target=myofullbody：直接作为最终 MSK 轨迹
              ├── target=myoarms：fullbody-to-arms profile + target manifest
              └── target=myolegs：fullbody-to-legs profile + target manifest
              │
              ▼
目标 MSKTrajectory + 目标 MSK model fingerprint + site/body cache
              │
              ▼
MyoFullBodyAssistEnv 的 composition.components["msk"]
              │
              ├── Device=None
              ├── 穿戴式 Device
              ├── 人形 Robot
              └── 独立机械臂
              │
              ▼
补齐非 MSK 状态并装载到最终组合模型
```

这里的 `MyoFullBodyAssistEnv` 是组合环境的当前类名。后续若改成更通用的
`MyoAssistEnv`，重映射接口仍然不依赖环境类名，只依赖 `msk_key` 和组件 manifest。

### 3.1 三个不同的模型视图

必须明确区分：

- **Fullbody intermediate view**：由 `myo_sim.load_spec("myofullbody")` 构造，作为
  所有源动作的统一人体表示；它也是 `msk_key=myofullbody` 的目标视图；
- **Target MSK view**：由 `myo_sim.load_spec(msk_key)` 构造。对 `myoarms/myolegs`
  执行从 fullbody intermediate view 到该视图的模型间映射，并重新生成运动学缓存；
- **Runtime composed view**：由 MSK、可选 Robot/Device 和 Terrain 组成，用于仿真
  与训练。

上述 intermediate/target MSK view 是 Assist 组合器在装配 Robot/Device 之前
保留的 MSK 组件视图。字符串 `myofullbody` 表示 `myo_sim` 的 `msk_key`，
也用于中间轨迹命名空间。

## 4. 模型与缓存来源

重映射模型直接通过 `myo_sim.load_spec(msk_key)` 构建，训练模型由 Assist
组合入口构建。缓存使用 `msk_key + fingerprint + profile version` 命名空间，
按关节名称映射到运行时模型。

## 5. 新增核心数据结构

建议增加模块：

```text
musclemimic/retargeting/msk_manifest.py
musclemimic/retargeting/msk_profiles.py
musclemimic/retargeting/msk_retarget.py
musclemimic/retargeting/msk_trajectory_adapter.py
```

### 5.1 `MskModelManifest`

从 `myo_sim.load_spec(msk_key)` 和编译后的 `MjModel` 动态生成：

```python
@dataclass(frozen=True)
class MskModelManifest:
    msk_key: str
    model_fingerprint: str
    joint_names: tuple[str, ...]
    qpos_indices: Mapping[str, tuple[int, ...]]
    qvel_indices: Mapping[str, tuple[int, ...]]
    body_names: tuple[str, ...]
    site_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    muscle_actuator_names: tuple[str, ...]
    root_joint_name: str | None
    root_body_name: str | None
```

所有索引必须根据 MuJoCo 的 `jnt_qposadr`、`jnt_dofadr`、joint type 和对象 ID
计算，不能假设每个 joint 都是一维，也不能依赖旧 XML 的排列。

### 5.2 `MskRetargetProfile`

每个 `msk_key` 对应一个显式 profile：

```python
@dataclass(frozen=True)
class MskRetargetProfile:
    source_msk_key: str
    target_msk_key: str
    source_skeleton: str
    target_sites: tuple[SiteTarget, ...]
    tracked_joint_names: tuple[str, ...]
    root_mode: str
    ignored_source_joints: tuple[str, ...]
    default_joint_source: str
    ik_config: str | None
```

profile 中的目标 joint/body/site 名称必须在启动时与 `myo_sim` manifest 校验。
名称不存在时立即报错，不允许静默跳过。

### 5.3 `MskTrajectory`

轨迹缓存只保存 MSK 数据和必要的 mimic cache：

```text
qpos_msk
qvel_msk
split_points
frequency
joint_names
site_names
site_xpos / site_xmat
必要的 sparse velocity cache
msk_key
myo_sim model_fingerprint
retarget_profile_version
source_dataset_id
```

缓存中不得保存 Robot/Device joint，也不得以 `device_key` 建目录。

## 6. 新的重映射入口

建议提供统一 API：

```python
trajectory = retarget_dataset_to_assist_msk(
    dataset_paths=paths,
    msk_key=config.experiment.env_params.msk_key,
    method="gmr",                 # 或 smpl
    profile_name=None,             # 默认等于 msk_key
    clear_cache=False,
)
```

内部按方案 A 固定执行：

1. 始终构造 `myo_sim.load_spec("myofullbody")`，生成并校验全身 manifest；
2. 将源人体数据标准化，并求解 Assist `myofullbody` 的 qpos/qvel；
3. 写入或复用按源 motion 索引的第一阶段 `myofullbody` 缓存；
4. 若目标是 `myofullbody`，直接生成其 site/body cache 并返回；
5. 若目标是 `myoarms` 或 `myolegs`，再构造目标 spec、manifest 和模型间 profile；
6. 按规范名称和语义映射将全身轨迹转换到目标模型，不按固定 qpos 列裁剪；
7. 在目标模型上执行必要的 IK/root/contact 修正；
8. 从目标 qpos 用 `mujoco.mj_differentiatePos` 重新计算 qvel；
9. 在目标模型上重新生成 full 或 sparse site/body cache；
10. 写入目标 MSK 缓存并返回 `MskTrajectory`。

不能再执行：

```python
env_name = env.__class__.__name__
load_robot_conf_file(env_name)
```

应改为：

```python
target_msk_key = env.msk_key
source_manifest = build_msk_manifest("myofullbody")
target_manifest = build_msk_manifest(target_msk_key)
profile = load_msk_retarget_profile(
    source_msk_key="myofullbody",
    target_msk_key=target_msk_key,
)
```

## 7. 直接装载到 Assist 环境

`ImitationFactory.get_amass_traj()` 应读取 `env.msk_key`，调用新的 MSK 重映射入口，
然后通过组件感知 adapter 装载到组合环境：

```python
runtime_traj = adapt_msk_trajectory_to_composed_model(
    msk_traj,
    model=env._model,
    msk_component=env.composition.components["msk"],
    robot_component=env.composition.components["robot"],
)
```

adapter 的规则如下：

1. MSK joint 按规范名称复制到最终组合模型；
2. 校验每个轨迹 joint 都属于 `components["msk"]`；
3. 最终模型中的 MSK joint 必须全部有明确来源；
4. Robot/Device joint 不参与人体重映射，初始化为最终模型 `qpos0`；
5. Robot/Device `qvel` 初始化为零；
6. Robot/Device 后续状态由其 controller 和共享物理动力学更新；
7. Terrain 不进入 trajectory joint/body/site 列表；
8. mimic cache 只包含当前 profile 声明的 MSK site；
9. 不允许使用“最终模型中不在轨迹里的 joint 全部补零”这种模糊规则；必须根据
   组件归属决定是 `MSK 缺失错误` 还是 `Robot 使用 qpos0`。

这样同一份 `myofullbody` MSK 轨迹可以装载到以下所有环境：

```text
myofullbody + flat
myofullbody + HipExoskeleton_L1 + flat
myofullbody + ElbowExoskeleton_L1 + flat
myofullbody + humanoid_robot + terrain
myofullbody + independent_robot_arm + terrain
```

## 8. `myofullbody` 的重映射流程

`msk_key=myofullbody` 时，不再创建 `MyoFullBody` 环境。流程为：

1. 从 `myo_sim.load_spec("myofullbody")` 构造 target model；
2. 从 `myo_sim` 模型解析 pelvis、torso、head、左右肩/肘/手、髋/膝/踝/趾等
   profile 所需目标 site；
3. 将 AMASS/SMPL-H 对应人体关节映射到这些 target site；
4. 在独立 MSK target model 上进行 SMPL IK 或 GMR；
5. 输出完整 `myofullbody` MSK qpos/qvel 和 mimic site cache；
6. 装载时只写入 Assist 环境的 `components["msk"]`；
7. Robot/Device joint 使用最终组合模型的 `qpos0`。

`disable_fingers` 只影响运行时 MSK manifest 的最终暴露集合：

- 保留手指时，轨迹可包含手指 joint；源数据没有的手指使用 `myo_sim` qpos0；
- 禁用手指时，adapter 按当前 MSK manifest 删除相应轨迹字段；
- 两种情况下名称都使用 `myo_sim` 的 `_l`、`_r` 等规范形式。

## 9. `myoarms` 的三阶段重映射流程

`myoarms` 采用已确认的方案 A。它复用第一阶段的 Assist 全身 MSK 轨迹作为人体
中间表示，每一级目标模型都来自
`myo_sim`/Assist 组件体系：

```text
SMPL
→ Assist MSK:myofullbody
→ Assist MSK:myoarms
→ myoarms site/body cache
```

具体流程：

1. 第一阶段生成或读取同一源 motion 的 `myofullbody` MSK 缓存；
2. 分别从 `myo_sim.load_spec("myofullbody")` 和
   `myo_sim.load_spec("myoarms")` 生成 source/target manifest；
3. 第二阶段使用规范 joint/site 名称建立 fullbody-to-arms 映射；
4. 保留 torso/root reference 以及左右 shoulder、elbow、wrist/hand 目标；
5. 髋、膝、踝等下肢状态仅作为第一阶段人体姿态信息，不写入最终 `myoarms` 轨迹；
6. 在 `myoarms` target model 上执行 forward kinematics/必要的 IK 修正，不能按固定
   qpos 列号裁剪；
7. 手指源数据缺失时使用 `myo_sim` qpos0；`disable_fingers` 只按规范名称过滤；
8. 从最终 `myoarms` qpos 序列重新计算 qvel；
9. 重新生成 `myoarms` 专属 site/body cache；
10. 装载到 `myoarms + Device/Robot + Terrain` 时只写 MSK 组件。

建议 `myoarms` profile 至少跟踪：

```text
torso/root reference
left/right shoulder
left/right elbow
left/right hand or wrist
```

具体 MuJoCo site 名称不在代码中手写为旧 MuscleMimic 名称，而是在 profile 中声明
语义角色，再校验其目标名称确实存在于当前 `myo_sim` 模型。

如果训练动作需要桌面、工具或外部物体坐标，应作为 task object trajectory 单独
装载，不能混入 MSK joint trajectory。

## 10. `myolegs` 的三阶段重映射流程

`myolegs` 同样采用方案 A：

```text
SMPL
→ Assist MSK:myofullbody
→ Assist MSK:myolegs
→ myolegs site/body cache
```

具体流程：

1. 第一阶段生成或读取同一源 motion 的 `myofullbody` MSK 缓存；
2. 从 `myo_sim` 生成 fullbody source manifest 和 myolegs target manifest；
3. 第二阶段按规范名称保留 root/pelvis、左右 hip、knee、ankle、toe 目标；
4. shoulder、elbow、wrist 等上肢状态不写入最终 `myolegs` 轨迹；
5. 根据 `myolegs` 实际 root joint 类型执行 root frame 转换，不能直接复制固定 qpos
   切片；
6. 在 `myolegs` target model 上执行 forward kinematics/必要的 IK 修正；
7. 使用 target model 的 floor/contact geom 重新计算足底落地偏移；
8. 从最终 `myolegs` qpos 用 `mj_differentiatePos` 重新计算 qvel；
9. 重新生成 pelvis、hip、knee、ankle、toe/foot 的专属 site/body cache；
10. 装载到组合环境后，外骨骼或鞋类 Device joint 从 `qpos0` 开始，由 Device
    controller 和约束动力学跟随人体。

`myolegs`、`myolegs26` 和由 `assist_sim` 派生的 `myolegs22` 不得共用隐式固定列
映射。它们必须分别拥有 profile 和 fingerprint：

```text
myolegs    → myo_sim 原生 80-muscle leg topology
myolegs26  → myo_sim 原生 26-muscle topology
myolegs22  → assist_sim 从 myo_sim myolegs26 显式派生的 topology
```

其中 `myolegs22` 不是当前 `myo_sim` 注册模型，若继续支持，应把“来源 key、派生操作、
派生后 fingerprint”写入 manifest，不能假装它等同于 `myolegs26`。

## 11. 名称冲突与迁移规则

### 11.1 权威名称来源

以下对象名称全部以 `myo_sim` 当前生成模型为准：

```text
joint
body
site
tendon
actuator
sensor
keyframe 中的 joint layout
```

代码不得维护一份与 `myo_sim` 并行演化的“标准名称全集”。profile 只保存重映射所需
的少量语义映射，并在运行时校验。

### 11.2 旧名称只读迁移

旧 MuscleMimic 数据若包含 `_left`、`_right` 等旧名称，必须经过集中 alias 层：

```python
canonical_name = legacy_aliases.get(old_name, old_name)
```

规则：

1. alias 只用于读取旧轨迹、旧配置和旧 checkpoint；
2. alias 结果必须存在于目标 `myo_sim` manifest；
3. 多个旧名称映射到同一规范名称时必须报冲突；
4. 新缓存、日志和 checkpoint 只写规范名称；
5. 不允许为了兼容旧缓存而重命名 `myo_sim` 模型对象。

## 12. 缓存设计

缓存 key 建议包含：

```text
source dataset identity
source file checksum
msk_key
myo_sim model fingerprint
retarget method and solver version
retarget profile version
target frequency
finger mode（仅当它改变 target joint topology）
```

缓存 key 不得包含：

```text
MyoFullBodyAssistEnv 类名
device_key
robot_key
terrain 配置
num_envs
nconmax / njmax
```

建议目录：

```text
converted_motion/
  msk/
    myofullbody/<fingerprint>/<profile-version>/...
    myoarms/<fingerprint>/<profile-version>/...
    myolegs/<fingerprint>/<profile-version>/...
```

其中 `myofullbody` 目录是可复用的第一阶段缓存，也是全身训练的最终缓存；
`myoarms/myolegs` 目录保存第二阶段的最终结果，并在 metadata 中记录所依赖的
`myofullbody` fingerprint 和 profile version。两阶段缓存均以同一个原始 motion ID
索引，因此一次全身求解可以被 arms 和 legs 批处理复用。

“使用方法与当前流程相同”指训练侧接口和数据集配置不变，并不承诺旧缓存文件与新
格式二进制兼容。loader 应自动解析当前环境的 `msk_key` 和 fingerprint：命中新缓存
时直接加载；只发现旧缓存或 fingerprint 不匹配时给出明确的批量转换命令，不得把旧
缓存当作原始数据再次重映射。

fingerprint 至少覆盖 joint 名称、joint type、qpos/dof layout、目标 site 名称与所属
body。仅使用 `msk_key` 不足以检测 `myo_sim` 模型升级。

## 13. 建议修改的文件

### 13.1 `loco_mujoco/smpl/retargeting.py`

- 新 MSK 流程统一从 `myo_sim` 构建模型；
- 将 GMR XML/spec 输入改为可传入的 `myo_sim` target spec/model；
- 将 `env_to_gmr_robot` 改为 profile registry，不再识别环境类名；
- `load_robot_conf_file(env_name)` 在 MSK 路径中替换为
  `load_msk_retarget_profile(msk_key)`；
- 缓存路径从 `env_name` 改为 `msk_key + fingerprint`。

### 13.2 `loco_mujoco/task_factories/imitation_factory.py`

- 从环境读取 `msk_key` 和 MSK component manifest；
- 调用 `retarget_dataset_to_assist_msk()`；
- 调用组件感知 trajectory adapter；
- 禁止 Assist 环境再进入 `MyoFullBody.yaml` 查找路径。

### 13.3 `musclemimic/models/composed_spec.py`

- 扩展 `ComponentNames`，保存 joint type、qpos/qvel 地址或稳定的 manifest 引用；
- 暴露组合前 MSK target spec 的 fingerprint；
- 保证 Device 有无变化时 MSK manifest 相同，否则构造阶段立即报错。

### 13.4 `musclemimic/environments/humanoids/myofullbody_assist.py`

- 暴露 `msk_key`、`msk_manifest` 和 `robot_manifest`；
- 提供 `load_msk_trajectory()`，明确执行 MSK 写入和 Robot `qpos0` 初始化；
- reward、goal、terminal handler 只消费 profile 中声明的 MSK mimic sites；
- 后续可将类重命名为更通用的 Assist 环境，但不得影响缓存。

### 13.5 配置文件

建议新增：

```yaml
experiment:
  env_params:
    msk_key: myofullbody
  task_factory:
    params:
      retarget:
        method: gmr
        profile: ${experiment.env_params.msk_key}
        cache_type: sparse
        clear_cache: false
```

`device_key` 保留在 `env_params`，但不传给 retargeting API。

### 13.6 `scripts/retarget_dataset.py`

- `--model` 支持 `MyoFullBodyAssistEnv`，`--msk-key` 限定为注册表中的有效 key；
- 扫描原始数据集并保留稳定的 source motion ID；
- `myofullbody` 执行第一阶段，`myoarms/myolegs` 自动复用或补做第一阶段后执行
  第二阶段；
- worker 只接收可序列化参数，每个 worker 独立构造 MuJoCo model；
- 输出成功、跳过、缓存命中和失败清单，进程退出码能反映批次是否有失败；
- `--clear-cache` 默认只清目标 key；显式增加 `--clear-intermediate-cache` 才清理可被
  其他目标复用的全身中间缓存。

## 14. 错误处理要求

以下情况必须在训练开始前失败：

- profile 中的 target joint/site 不存在于 `myo_sim` manifest；
- MSK trajectory 缺少最终 MSK 模型必需 joint，且 profile 未声明默认值；
- 轨迹 fingerprint 与当前 MSK fingerprint 不一致；
- 轨迹字段误包含 Robot/Device joint；
- 同名 joint 在 MSK 和 Robot 组件中发生冲突且组合器未完成显式重命名；
- source-to-target 映射存在重复目标；
- qpos/qvel 宽度与 manifest 不一致；
- `myoarms` profile 引用了下肢目标，或 `myolegs` profile 引用了上肢目标；
- 旧名称 alias 后仍不存在于 `myo_sim`；
- `myoarms/myolegs` 缓存记录的 fullbody fingerprint 与当前第一阶段不一致。

不得静默补零 MSK 必需 joint，也不得因某个 site 缺失而自动降低 reward 维度。

## 15. 验收方案

### 15.1 模型与名称验收

对 `myofullbody`、`myoarms`、`myolegs` 分别验证：

1. manifest 的 joint/body/site/actuator 均来自 `myo_sim`；
2. 新主流程统一使用当前模型提供方；
3. 旧名称输入可迁移，新缓存中不存在旧名称；
4. 指纹在模型不变时稳定，在 topology/name 变化时改变。

### 15.2 Device 无关性验收

对同一数据片段分别运行：

```text
MSK + no device
MSK + Device A
MSK + Device B
MSK + independent robot
```

必须满足：

- 只执行一次 MSK retarget；
- 命中同一 MSK cache；
- MSK qpos/qvel 数值一致；
- Device/Robot joint 初始值等于各自最终模型 qpos0；
- policy action 仍只包含 MSK muscle actuator。

### 15.3 数据与动力学验收

- reset 后 MSK qpos 与目标帧一致；
- Robot/Device qpos 等于 qpos0，qvel 为零；
- 连续帧 qvel 与 `mj_differentiatePos` 一致；
- mimic site cache 与在 target model 上重新 `mj_forward` 的结果一致；
- 轨迹末端、随机起点和 autoreset 可在 JAX jit 下运行；
- CPU MuJoCo 与 GPU Warp reset/step 均为有限值。

### 15.4 最小真实数据矩阵

至少选择一段包含上肢和下肢运动的 KIT/AMASS 动作，执行：

| MSK | Device | 重点检查 |
|---|---|---|
| `myofullbody` | None | 全身 joint/site 映射 |
| `myofullbody` | HipExoskeleton | 与上一行复用缓存，Device 使用 qpos0 |
| `myoarms` | None | 下肢源关节被忽略，双臂轨迹连续 |
| `myoarms` | ElbowExoskeleton | 复用 myoarms 缓存，肘部约束稳定 |
| `myolegs` | None | root、髋膝踝趾和地面偏移 |
| `myolegs` | leg Device | 复用 myolegs 缓存，Device 不进入 retarget |

还必须验证：同一个源 motion 先批处理 `myoarms`、再批处理 `myolegs` 时，第二次运行
命中完全相同的 `myofullbody` 第一阶段缓存；最终 arms/legs cache 的 fingerprint、
joint layout 和 site/body cache 则各自独立。

最后对每个 MSK 至少完成一次 GPU 最小 PPO：2 个以上环境、2 个 rollout step、
2 个 PPO update，并检查 reward、loss、参数和 observation 中无 NaN/Inf。

## 16. 实施顺序

1. 建立 `MskModelManifest`、fingerprint 和权威名称校验；
2. 模型来源统一为 `myo_sim`；
3. 实现 `SMPL → Assist myofullbody` 第一阶段及可复用缓存；
4. 实现 `myofullbody → myoarms/myolegs` 模型间 profiles 和第二阶段缓存；
5. 实现统一批处理脚本、失败清单与两层缓存生命周期；
6. 实现组件感知 trajectory adapter；
7. 修改 `ImitationFactory` 和 Assist 环境装载接口，同时保持训练配置用法不变；
8. 完成名称迁移、错误检查和单元测试；
9. 下载一段真实 KIT/AMASS 数据执行完整批处理矩阵验收；
10. 最后执行真实数据 GPU 最小 PPO 验收。

完成上述修改后，重映射的身份将由：

```text
环境类名 + MuscleMimic 旧 XML
```

彻底改为：

```text
msk_key + myo_sim topology fingerprint + retarget profile
```

从而保证真实数据直接映射到 Assist 环境的 MSK 部分，并与任何可选 Robot/Device
和 Terrain 解耦。

## 17. 2026-09-10 实施验收清单

本节记录本轮实际执行结果；`[x]` 表示已通过，`[ ]` 表示尚未通过或被外部资源阻塞。

### 17.1 代码改造

- [x] 新增 `MskManifest`，从 `myo_sim.load_spec(msk_key)` 编译模型并按 MuJoCo
  joint type、qpos/qvel address 生成布局；fingerprint 覆盖 joint、body、site 和
  site 所属 body。
- [x] 新增显式 MSK profile registry 和 `assist-msk-v1` profile version。
- [x] 新增 `retarget_dataset_to_assist_msk()` 统一入口；`myofullbody` 不再由环境类名
  决定缓存身份。
- [x] Assist GMR 目标 XML 由 `myo_sim` 派生的有效模型临时导出；新主流程不再导入或
  统一使用 `myo_sim` 的模型构建入口。
- [x] 缓存命名空间改为
  `msk/<msk_key>/<fingerprint>/<profile-version>/<method>/...`，并写入、校验
  `msk_key`、模型 fingerprint 和 profile version metadata。
- [x] 新增组件感知 adapter：MSK joint 按名称复制，Robot/Device qpos 使用最终模型
  `qpos0`，Robot/Device qvel 使用零；缺失 MSK joint 或混入非 MSK joint 会报错。
- [x] `ImitationFactory` 对具有 `msk_key` 的 Assist 环境调用新入口，Device/Robot
  不参与源动作重映射。
- [x] `MyoFullBodyAssistEnv` 暴露有效运行时 `msk_manifest` 和 `robot_manifest`。
- [x] 批处理脚本支持 `MyoFullBodyAssistEnv`、`--msk-key` 和精确的 `--motions`，因此
  可以只处理指定轨迹。
- [x] GMR 改为延迟导入，并在批处理 worker 中禁用不稳定的 TorchScript 编译；修复
  当前 Python 3.11/PyTorch 2.14 组合导入 GMR 时的段错误。
- [x] `myofullbody -> myoarms/myolegs` 第二阶段按规范 joint 名称映射；手指源关节缺失时
  保持 target qpos0，不使用固定 qpos 列裁剪；目标 qvel 在 target model 上重新由
  `mj_differentiatePos` 计算。

### 17.2 自动化测试

- [x] manifest 指纹稳定、free joint qpos/qvel 宽度正确。
- [x] 未实现的 arms/legs profile fail-fast。
- [x] adapter 的 MSK 名称映射、Robot qpos0 和 Robot 零 qvel。
- [x] 既有 Assist Stage 1 组合、观测、动作及 reset 测试。
- [x] 执行命令：

  ```bash
  .venv/bin/pytest -q \
    tests/unit/test_assist_msk_retargeting.py \
    tests/test_fullbody_assist_stage1.py
  ```

  结果：`8 passed`。

### 17.3 两条 KIT 真实轨迹批处理

选择的源动作：

1. 走路：`KIT/11/WalkingStraightForwards01_poses`；
2. 弯腰搬物近似动作：`KIT/441/castBox01_poses`（KIT 文件名为抛/搬箱动作）。

实际执行命令：

```bash
AMASS_PATH=/home/chenshuo/workspace/musclemimic/dataset \
MUSCLEMIMIC_CONVERTED_AMASS_PATH=/home/chenshuo/workspace/musclemimic/converted_datasets \
MUSCLEMIMIC_SMPL_MODEL_PATH=/home/chenshuo/workspace/musclemimic/musclemimic/models/smpl \
.venv/bin/python scripts/retarget_dataset.py \
  --model MyoFullBodyAssistEnv \
  --msk-key myofullbody \
  --motions \
    KIT/11/WalkingStraightForwards01_poses \
    KIT/441/castBox01_poses \
  --retargeting-method gmr \
  --workers 1 \
  --gmr-no-offset-to-ground
```

- [x] 两个源 NPZ 均存在，且输入含 `poses/trans/betas/gender/mocap_framerate`。
- [x] 脚本只调度上述两条轨迹，没有展开整个 KIT group。
- [x] 两条轨迹均进入 Assist `myofullbody` 新流程，有效模型 fingerprint 为
  `d3d43907060afb35`。
- [x] SMPL-H/MANO 模型完整，派生的 `SMPLH_NEUTRAL.pkl` 可由 `SMPLH_Parser` 加载。
- [x] 人体形状拟合完成：最终 loss `0.003334 m`、scale `1.0492`，第二条轨迹复用
  同一 `myofullbody_shape.pkl`。
- [x] 走路轨迹 GMR 求解成功：168 个 GMR 帧，重采样为 500 个 100 Hz 帧。
- [x] 搬箱轨迹 GMR 求解成功：362 个 GMR 帧，重采样为 1080 个 100 Hz 帧。
- [x] 两条最终 `.npz` 均写入
  `converted_datasets/msk/myofullbody/d3d43907060afb35/assist-msk-v1/gmr/KIT/...`。
- [x] 两个缓存均为 `qpos=(T,93)`、`qvel=(T,92)`，metadata 中的 `msk_key`、
  fingerprint、profile version 均正确。
- [x] sparse cache 已持久化：`site_xpos=(T,17,3)`、`site_xmat=(T,17,9)`、
  `cvel_parent=(T,17,6)`、`subtree_com_root=(T,3)`。
- [x] qpos、qvel 及全部 sparse cache 无 NaN/Inf。
- [x] 按中心差分规则用 `mujoco.mj_differentiatePos` 重算 qvel：走路最大绝对误差
  `4.62e-7`，搬箱最大绝对误差 `2.38e-7`。
- [x] 再次同时提交两条轨迹时均命中已有缓存并成功装载，结果 `2/2`、成功率 `100%`。

### 17.4 最终验收结论

- [x] 本轮要求的两条 `myofullbody` KIT 轨迹已完成真实批量重映射和缓存验收。
- [x] 批处理仅处理一个走路和一个弯腰搬物近似动作，没有处理整个 KIT 数据集。
- [x] 轨迹目标为 Assist MSK，缓存身份由
  `myofullbody + d3d43907060afb35 + assist-msk-v1` 决定。
- [x] 单元与 Stage 1 回归测试结果为 `8 passed`。
- [x] 本轮随后完成了 `myoarms/myolegs` 各一条轨迹的二阶段 MSK 验证（详见 17.5）；
  Device 组合回放和 GPU PPO 仍属于第 15 节未执行的扩展验收。

### 17.5 `myoarms/myolegs` 二阶段最小真实验收

本轮分别使用一条已有 Assist fullbody 缓存验证目标组件映射：

| 目标 MSK | 源轨迹 | 结果 | fingerprint | qpos/qvel |
|---|---|---|---|---|
| `myoarms` | `KIT/11/WalkingStraightForwards01_poses` | 成功 | `9e0744a30e17745b` | `(500,76)/(500,76)` |
| `myolegs` | `KIT/441/castBox01_poses` | 成功 | `a04410e825ed10b4` | `(1080,35)/(1080,34)` |

执行命令分别为：

```bash
AMASS_PATH=/home/chenshuo/workspace/musclemimic/dataset \
MUSCLEMIMIC_CONVERTED_AMASS_PATH=/home/chenshuo/workspace/musclemimic/converted_datasets \
MUSCLEMIMIC_SMPL_MODEL_PATH=/home/chenshuo/workspace/musclemimic/musclemimic/models/smpl \
.venv/bin/python scripts/retarget_dataset.py \
  --model MyoFullBodyAssistEnv --msk-key myoarms \
  --motions KIT/11/WalkingStraightForwards01_poses \
  --retargeting-method gmr --workers 1 --gmr-no-offset-to-ground

AMASS_PATH=/home/chenshuo/workspace/musclemimic/dataset \
MUSCLEMIMIC_CONVERTED_AMASS_PATH=/home/chenshuo/workspace/musclemimic/converted_datasets \
MUSCLEMIMIC_SMPL_MODEL_PATH=/home/chenshuo/workspace/musclemimic/musclemimic/models/smpl \
.venv/bin/python scripts/retarget_dataset.py \
  --model MyoFullBodyAssistEnv --msk-key myolegs \
  --motions KIT/441/castBox01_poses \
  --retargeting-method gmr --workers 1 --gmr-no-offset-to-ground
```

- [x] 两条二阶段命令均为 `1/1` 成功，且复用了对应的 `myofullbody` 缓存。
- [x] `myoarms` 目标 site cache 为 `(500,1660,3)`，`myolegs` 为 `(1080,1060,3)`；
  qpos、qvel、site cache 均无 NaN/Inf。
- [x] 目标模型 qvel 重新用 `mj_differentiatePos` 中心差分计算，最大误差分别为
  `4.62e-7`（arms）和 `1.19e-7`（legs）。
- [x] 目标缓存写入：
  `converted_datasets/msk/myoarms/9e0744a30e17745b/assist-msk-v1/gmr/...`；
  `converted_datasets/msk/myolegs/a04410e825ed10b4/assist-msk-v1/gmr/...`。
- [ ] 尚未验证 arms/legs 与 Device/Robot 组合环境的运行时回放；本节只验收 MSK 目标视图
  的二阶段重映射和缓存生成。

### 17.6 `myolegs` 走路轨迹补充验收

- [x] 输入：`KIT/11/WalkingStraightForwards01_poses`。
- [x] 处理命令成功：`MyoFullBodyAssistEnv --msk-key myolegs`，结果 `1/1`。
- [x] 目标缓存：
  `converted_datasets/msk/myolegs/a04410e825ed10b4/assist-msk-v1/gmr/KIT/11/WalkingStraightForwards01_poses.npz`。
- [x] 输出尺寸：`qpos=(500,35)`、`qvel=(500,34)`、`site_xpos=(500,1060,3)`、
  `site_xmat=(500,1060,9)`。
- [x] qpos、qvel、site cache 全部为有限值，metadata 中 `msk_key=myolegs`、
  fingerprint=`a04410e825ed10b4`、profile=`assist-msk-v1`。

## 18. 当前环境入口

公共环境注册使用 `MyoFullBodyAssistEnv` / `MjxMyoFullBodyAssistEnv`。
`msk_key` 选择 `myofullbody`、`myoarms` 或下肢模型；`myofullbody.py` 是
Assist 环境复用的规格、观测和动作实现基类，必须保留。

重映射脚本、可视化入口和 Web viewer 使用统一 Assist 入口；
`msk_metrics.load_model()` 编译与运行时一致的 `myo_sim + Assist + terrain` 模型。
