# 使用 H 训练跑步策略

配置：`fullbody/conf_fullbody_run.yaml`。已删除 A–G 的生成视频、截取源文件及重映射缓存；原始数据集保留。

## 参考轨迹

- H：`KIT/183/walking_run05_poses`，原始第 65–435 帧，即 0.65–4.36 秒（右端不含）。
- 截取源文件：`outputs/fullbody_run_h/reference_data/sources/KIT/183/walking_run05_crop_65_436_poses.npz`。
- 重映射：365 帧、100 Hz、3.65 秒，包含起步与停下。
- 模型：全身 `myofullbody`，无外骨骼；删除手指关节和对应肌肉，保留腕关节。
- 视频：`outputs/fullbody_run_h/reference_data/H/reference.mp4`，参考回放，不是策略训练结果。

## 启动训练

在项目根目录执行，按实际分配设置物理 GPU 编号：

```bash
RUN_GPU_INDEX=3 bash fullbody/train_run.sh
```

脚本将 `AMASS_PATH` 指向截取源文件目录，`CONVERTED_AMASS_PATH` 指向已生成的缓存。

- 4096 个环境，每轮每环境 64 步；总上限 10000 轮（2,621,440,000 步）。
- actor/critic 为 5 层、每层 1024 单元，其余 PPO、奖励及终止配置沿用 walk。
- 每 100 轮验证及保存 checkpoint；验证和视频长度为 371 步。
- 训练随机参考帧起步；验证从第一帧开始，同一条 H 动作，使用均值动作。
- 输出：`outputs/fullbody_run_h/`；run_id 为 `fullbody_run_h`，自动续训只读取该实验的 checkpoint。

## 重建参考

```bash
.venv/bin/python fullbody/prepare_run.py --motion KIT/183/walking_run05_poses --label H --start 0.65 --end 4.36 --description '向前跑，含起步和停下' --record
.venv/bin/python fullbody/build_run_gallery.py
```
