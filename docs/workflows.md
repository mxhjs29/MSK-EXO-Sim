# 全身模型工作流

部署、数据集下载和最小验证见 [README](../README.md)。人体模型由 `models/myo_sim` 提供，通过 Assist 环境的 `msk_key` 选择；当前 walk/run/bend/lift 与负载任务配置使用全身模型。

1. `uv sync --extra dev --extra cuda` 安装环境。
2. 先运行 CPU 场景编译与 GPU 合成轨迹 PPO smoke。
3. 获取 Hugging Face 数据集访问权限并下载原始数据：
   `.venv/bin/python -m musclemimic.cli.download_data --family all --destination datasets`。
4. 需要动作重映射时增加 `smpl`、`gmr` 可选依赖，并配置 AMASS/SMPL/converted 路径。
5. 使用 `scripts/retarget_dataset.py` 将原始动作重映射至全身肌骨模型。
6. 使用 `configs/experiments/fullbody/` 配置启动 `musclemimic.cli.train_fullbody`；先少量环境验证，再扩大规模。
7. 使用 `musclemimic.runner.evaluation.fullbody` 评估、录制或导出训练结果。

原始数据不会直接成为 MuJoCo 的 qpos/qvel 参考轨迹；必须经过本项目重映射。此次 HF 补传只增加 Gait120 原始文件；下载器不选择仓库已有的重映射缓存，也不下载 PPO checkpoint。训练结果写入本地 `outputs/`。
