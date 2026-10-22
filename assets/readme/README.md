# README 模型与场景配图

本目录的 12 张 PNG 直接由项目中的 MuJoCo 模型与场景生成，分辨率为 960 × 720。肌骨模型使用 `models/myo_sim`；设备使用 `models/assist_sim`；地形使用 `models/myoassist_terrains`，通过公共 `build_msk_device_terrain_spec()` 构建。

图片为静态结构预览，不运行策略、不推进动力学，也不读取动作数据或 checkpoint。肌骨图显示骨骼与肌腱；穿戴设备图降低人体显示透明度以突出设备；Franka 图隐藏人体，并设置静态机械臂姿态。所有调整仅应用于渲染内存，不修改模型资源或训练配置。

坡地图使用 15°；楼梯使用单个 4 m × 4 m 拼块，默认 6 级台阶、每级 0.15 m，镜像下行。其他参数、相机、模型维度与图片 SHA256 见 [manifest.json](manifest.json)。

从仓库根目录复现（需要已安装项目、Pillow，以及系统 OSMesa 库）：

```bash
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  .venv/bin/python scripts/render_readme_gallery.py

# 单独重绘某个设备或地形
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/render_readme_gallery.py --only Hippo_L1 warehouse
```

也可按本机 OpenGL 配置设置 `MUJOCO_GL=egl`。图片与模型遵循仓库及对应模型资源的许可和来源声明；本目录没有引入外部图片。
