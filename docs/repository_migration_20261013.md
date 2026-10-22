# 2026-10-13 目录迁移与验证

本次在当前工作区实施结构调整，保留迁移前已经存在的未提交修改和新增任务。迁移阶段未改变 origin、当时尚未推送；后续发布状态见文末。

## 已实施的迁移

| 原位置/职责 | 当前维护位置 |
|---|---|
| `musclemimic/models/composed_spec.py` | `musclemimic/scenes/composition.py`，原入口为兼容模块 |
| 环境文件中的哑铃、箱子 MJCF 构建 | `musclemimic/scenes/objects/` |
| `core/reward/trajectory_based.py` | `rewards/common/imitation.py` |
| 环境/实验脚本中的额外奖励 | `rewards/tasks/{payload,dumbbell_lift,box_friction,box_lift,gait120}.py` |
| 负载目标、终止条件 | `tasks/dumbbell_lift/`、`tasks/box_lift/` |
| 手部接触读数 | `core/sensors/hand_contact.py` |
| 通用物理工具 | `controllers/`、`data/trajectory.py`、`runner/frozen_policy.py`、`core/sensors/foot_contact.py` |
| `musclemimic/experiments/` | `musclemimic/runner/workflows/`，原入口保留兼容 |
| 训练与评估的 Python 主程序 | `cli/`、`runner/evaluation/`、`runner/jobs/`、`runner/workflows/` |
| 训练与评估的配置 | `configs/experiments/`，原配置路径为链接 |
| 公共基础奖励参数 | `configs/rewards/common/`，通过 Hydra 合成 |
| 旧 viewer/web_viewer | `visualization/viewer/`、`visualization/web_viewer/`，旧 Python 入口兼容 |
| 数据目录与 outputs 中有用的预处理源码 | `data/preprocessing/`，原始数据和输出仍保留本地 |
| 许可模型与历史研究文档 | `local/assets/smpl/`、`local/docs/plan/` |

新增场景描述/所有权、共享 CPU 世界与 agent 动作路由，供后续多智能体开发使用；当前没有新增多智能体 PPO 算法。模型包内部资源结构和 LocoMuJoCo 框架保留。

后台作业的本机 GPU UUID、CPU 绑核放入 `local/configs/resources.json`。现有本机分配保留，其他机器可使用环境变量或本地配置覆盖。shell 启动器由 `scripts/` 维护，旧路径继续转发。

## 验证范围与结果

机器可读结果和日志位于 `outputs/repository_migration_20261013/`，主报告为其中的 `report.html`。

- 24 组迁移前后确定性数组逐元素完全一致：全身、MyoLegs26、MyoArms、Hippo 的状态/激活/传感器，以及哑铃/箱子的观测、状态和奖励。
- 15 份训练配置行为等价；14 份合成结果完全相同，Gait120 的单一本机绝对路径改为指向同一文件的项目相对路径。
- 9 组物理组合编译成功，覆盖平地、斜坡、仓库、Hippo、肘外骨骼、独立 Go2 机器人、上下肢模型、2 kg 右手刚性哑铃与 5 kg 摩擦自由箱。场景检查确认箱子没有 `payload_hold` 约束且手部碰撞启用。
- 既有哑铃/箱子 PPO checkpoint 恢复后各推进 3 个控制步，观测和动作维数维持 2808/416、2838/416。
- walk/run/bend/lift 的旧冻结 PPO 均完成加载、reset、动作推理和 snapshot/restore；无操作 substep observer 与原始 env.step 的结果逐元素一致。
- GPU1 的小批 Warp PPO 完成 2 次更新（8 个环境步）并写出 checkpoint。此次只验证训练链路，不重新开展长期性能训练。
- 公共单元/回归检查：506 passed、19 skipped、20 deselected。
- 无 datasets、outputs 的导出副本完成 59 项检查，1 项因没有真实数据跳过。
- 公共 wheel/sdist 均构建成功，内容检查没有许可资源、数据集或 outputs；安装后的 wheel 通过 Hydra 公共奖励组、GMR 配置资源与场景编译检查。
- `make lint`、`uv lock --check`、源码语法检查通过。

这些验证证明迁移覆盖范围内的行为与依赖保持有效；小批训练和短 checkpoint 回放不能替代长期训练收敛验证。

## 原有问题与边界

迁移前 `tests/unit`（包含 integration）的基线为 509 passed、22 skipped、13 failed、11 errors。10 个失败来自奖励测试的无约束 MagicMock：它虚构了 `_msk_joint_names`，误触发 MSK 子集逻辑。已修正测试夹具为显式 `None`，生产奖励算法未变。

迁移时另有 14 个标记为 integration 的环境测试失败或报错。默认 `make test` 使用 `not integration` 范围；确定性比较、真实 checkpoint 回放和 GPU smoke 使用维护中的 Assist/负载环境。

场景 YAML 是物理组合入口；现有 PPO 仍通过完整 experiment 配置创建环境。任务/agent/算法等配置模板用于明确职责，不表示所有模板已经可任意互换。新的 CPU shared world 尚未接入多智能体 JAX/Warp learner。

## 回溯与复现

- `source_before.tar.gz`：迁移前源码备份，不包含数据集、outputs、运行环境及 SMPL 许可模型。
- `working_tree_before.patch`、`git_status_before.txt`：原有工作区修改记录。
- `source_hashes_before.json`、`preservation.json`：原文件校验及去向检查。
- `moves.json`：主要文件/符号迁移清单。
- `physics_before.npz`、`physics_after.npz`、`physics_comparison.json`：确定性数值证据。
- `config_equivalence.json`、`scenes.json`、`checkpoints.json`、`frozen_policy.json`、`wm_checkpoint.json`：功能检查。
- `public_tests.xml`、`local_wm.xml`、`clean_checkout.xml`：测试记录。
- `dist/`：已检查的公共构建产物；三个模型包仍作为独立依赖保留在仓库。

备份用于对照和逐项恢复，不应直接覆盖整个工作区，否则会覆盖后续改动。长期训练数据和输出从未搬移或删除。

## 后续调整：SMPL 资源纳入 Git

按用户后续要求，`local/assets/smpl/` 的 25 个资源文件及 `musclemimic/models/smpl` 兼容链接已加入 Git 上传范围；其中 8 个 PKL/NPZ 模型文件采用 Git LFS，合计资源约 590 MiB。数据集、outputs 和其他 local 内容仍忽略。Python wheel/sdist 继续排除这批资源；前述构建检查是 Python 发行包结果。许可文件随资源保留；用户已确认授权，后续完成 Git LFS 上传。

## 正式发布补充

GitHub 部署与 contributor 扩展说明使用统一 Assist 训练流程及当前全身任务配置。HF 使用已有 `mxhjs29/musclemimic-msk` 私有数据集，只补传 `Gait120_101_to_110/` 原始文件、说明与逐文件 SHA256 清单；已有仓库目录保持不变，不追加重映射结果。

## 发布确认

2026-10-13 已推送至现有 GitHub origin 的 main，包含全身训练代码、README、公共配置、测试与授权 SMPL/MANO Git LFS 资源；不含数据集或 outputs。

HF 仅补传既有 `mxhjs29/musclemimic-msk` 的 Gait120 原始来源文件：1,329 个、102,194,713 字节，另附说明和哈希清单。固定 revision 为 `2d6ce93d8d743083e822479f3ab27b6ac35dfc70`，其他原有目录校验未变。远端原文件哈希已逐项检查，并抽样下载了 TRC/MAT 验证字节内容。本次最终公共测试为 510 passed、19 skipped、20 deselected；合成全身 GPU PPO 完成两次更新。
