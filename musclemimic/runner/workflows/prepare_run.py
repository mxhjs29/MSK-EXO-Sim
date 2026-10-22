from musclemimic.utils.paths import workspace_root, config_root
"""Retarget the running reference on CPU and optionally record its replay."""
import os
from pathlib import Path
import sys
import argparse
import json

ROOT = workspace_root()
OUTPUT = ROOT / "outputs/fullbody_run_h/reference_data"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["AMASS_PATH"] = str(ROOT / "datasets/musclemimic-msk")
os.environ["SMPL_MODEL_PATH"] = str(ROOT / "musclemimic/models/smpl")
os.environ["CONVERTED_AMASS_PATH"] = str(OUTPUT / "amass_cache")
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from loco_mujoco.task_factories import TaskFactory
from musclemimic.environments.humanoids.myofullbody import MyoFullBody
import numpy as np


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="Record the reference, not a trained policy.")
    parser.add_argument("--motion", required=True, help="AMASS relative path without .npz")
    parser.add_argument("--label", required=True, help="Candidate label for output files")
    parser.add_argument("--start", type=float, help="Crop source interval start in seconds")
    parser.add_argument("--end", type=float, help="Crop source interval end in seconds")
    parser.add_argument("--description", default="Straight running reference")
    parser.add_argument("--output-root", type=Path, default=OUTPUT, help="Reference/cache output directory")
    parser.add_argument("--render-size", type=int, default=480, help="Square size of each camera view")
    parser.add_argument("--render-stride", type=int, default=4, help="Render every N reference frames")
    parser.add_argument("--align-root", action="store_true", help="Align initial pelvis before IK; use an isolated cache")
    parser.add_argument("--include-fingers", action="store_true", help="Keep all finger joints and muscle actuators")
    parser.add_argument("--config-name", default="conf_fullbody_run", help="Training configuration to use for the reference")
    args = parser.parse_args()
    if args.render_size < 16 or args.render_size % 16 or args.render_stride < 1:
        parser.error("render-size must be a positive multiple of 16; render-stride must be positive")
    OUTPUT = args.output_root.resolve()
    cache_root = OUTPUT / ("amass_cache_root_aligned" if args.align_root else "amass_cache")
    os.environ["CONVERTED_AMASS_PATH"] = str(cache_root)
    source_path = ROOT / "datasets/musclemimic-msk" / (args.motion + ".npz")
    with np.load(source_path) as src:
        source = {key: src[key] for key in src.files}
    source_fps = float(source["mocap_framerate"])
    source_frames = len(source["trans"])
    start_frame, end_frame = 0, source_frames
    retarget_motion = args.motion
    if args.start is not None or args.end is not None:
        start_frame = round((args.start or 0) * source_fps)
        end_frame = round(args.end * source_fps) if args.end is not None else source_frames
        assert 0 <= start_frame < end_frame <= source_frames
        source = {k: (v[start_frame:end_frame] if k in ("poses", "trans", "dmpls") else v)
                  for k, v in source.items()}
        retarget_motion = args.motion.removesuffix("_poses") + f"_crop_{start_frame}_{end_frame}_poses"
        staged_root = OUTPUT / "sources"
        staged_path = staged_root / (retarget_motion + ".npz")
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(staged_path, **source)
        os.environ["AMASS_PATH"] = str(staged_root)
    with initialize_config_dir(config_dir=str(config_root()/"experiments/fullbody"), version_base=None):
        config = compose(config_name=args.config_name, overrides=[
            f"experiment.task_factory.params.amass_dataset_conf.rel_dataset_path=[{retarget_motion}]",
            f"experiment.validation.amass_dataset_conf.rel_dataset_path=[{retarget_motion}]",
        ])
    if args.align_root:
        OmegaConf.update(config, "experiment.task_factory.params.amass_dataset_conf.gmr_config.initialize_root_from_target",
                         True, force_add=True)
    params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    if args.include_fingers:
        params["disable_fingers"] = False
    params.update(env_name="MyoFullBodyAssistEnv", mjx_backend="jax")
    params.pop("num_envs", None)
    params["th_params"] = dict(random_start=False, fixed_start_conf=[0, 0], start_from_random_step=False)
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(**params, **OmegaConf.to_container(config.experiment.task_factory.params, resolve=True))
    traj = env.th.traj
    model = env._model
    assert traj.data.qpos.shape[-1] == model.nq
    assert traj.data.qvel.shape[-1] == model.nv
    assert np.isfinite(traj.data.qpos).all() and np.isfinite(traj.data.qvel).all()
    names = {model.joint(i).name for i in range(model.njnt)}
    finger_names = {f"{stem}_{side}" for stem in MyoFullBody._FINGER_JOINT_STEMS for side in ("l", "r")}
    disable_fingers = params.get("disable_fingers", True)
    if disable_fingers:
        assert not names.intersection(finger_names), "Finger joints remain in the model"
    else:
        # The myo_sim asset leaves four right-hand joint names unsuffixed.
        for stem in MyoFullBody._FINGER_JOINT_STEMS:
            assert f"{stem}_l" in names, f"Missing left finger joint: {stem}"
            assert f"{stem}_r" in names or stem in names, f"Missing right finger joint: {stem}"
    frames = len(traj.data.qpos)
    speed = np.linalg.norm(np.diff(source["trans"][:, :2], axis=0), axis=1) * float(source["mocap_framerate"])
    report = dict(source=args.motion, label=args.label, mean_speed_m_s=float(speed.mean()),
                  retarget_source=retarget_motion, description=args.description,
                  source_start_s=start_frame / source_fps, source_end_s=end_frame / source_fps,
                  source_duration_s=(end_frame - start_frame) / source_fps,
                  nq=model.nq, nv=model.nv, actuators=model.nu,
                  frames=frames, fps=float(traj.info.frequency), duration_s=frames / float(traj.info.frequency),
                  disable_fingers=disable_fingers, joint_names=sorted(names),
                  initialize_root_from_target=args.align_root, cache_root=str(cache_root))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidate_dir = OUTPUT / args.label
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "reference_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if args.record:
        # Sample the reference at its true frame rate for normal-speed playback.
        import mujoco
        import imageio.v2 as imageio
        data = mujoco.MjData(model)
        camera = mujoco.MjvCamera()
        camera.distance = 3.5
        camera.elevation = -12
        renderer = mujoco.Renderer(model, height=args.render_size, width=args.render_size)
        with imageio.get_writer(str(candidate_dir / "reference.mp4"), fps=float(traj.info.frequency) / args.render_stride, codec="libx264") as writer:
            for frame in range(0, frames, args.render_stride):
                data.qpos[:] = traj.data.qpos[frame]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.subtree_com[0]
                camera.lookat[2] = 0.9
                views = []
                for angle in (90, 150):
                    camera.azimuth = angle
                    renderer.update_scene(data, camera=camera)
                    views.append(renderer.render().copy())
                writer.append_data(np.concatenate(views, axis=1))
        renderer.close()
    env.stop()


if __name__ == "__main__":
    main()
