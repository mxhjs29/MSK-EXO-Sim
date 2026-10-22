"""Replay saved bend/lift evaluations with muscles and a moving reference overlay.

Run this before export_readme_demos.py. Requires the local evaluation NPZs and
their original configurations; no policy inference or simulation stepping occurs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LP_NUM_THREADS", "2")

import imageio.v2 as imageio
import mujoco
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from musclemimic.runner.workflows.payload_refinement import make_cpu

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "bend": ("payload_refinement_v4", "pose_1kg_1"),
    "lift": ("payload_refinement_v3", "refine_2kg_1"),
}
REFERENCE_RGBA = (0.471, 0.38, 0.812, 0.35)


def add_reference(model, reference, option, scene, payload_body_id):
    """Add only purple human geometry; keep the measured object's view unobscured."""
    begin = scene.ngeom
    mujoco.mjv_addGeoms(model, reference, option, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
    visible = 0
    for index in range(begin, scene.ngeom):
        geom = scene.geoms[index]
        if (
            geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
            and 0 <= geom.objid < model.ngeom
            and model.geom_bodyid[geom.objid] not in (0, payload_body_id)
        ):
            geom.rgba[:] = REFERENCE_RGBA
            geom.matid = -1
            geom.texcoord = 0
            visible += 1
        else:
            geom.rgba[3] = 0
    assert visible > 0, "Reference human geometry must be visible"


def render(task, output):
    version, label = SOURCES[task]
    folder = ROOT / "outputs" / version / task
    evaluation = folder / "evaluations" / label
    result = json.loads((evaluation / "results.json").read_text())
    record = next(r for r in result["records"] if r["start"] == 0)
    assert record["termination"] == "reference_end" and record["steps"] == record["expected_steps"]
    rollout = np.load(evaluation / "start_0_rollout.npz")
    poses, references = rollout["qpos"], rollout["qref"]
    assert len(poses) == len(references) == record["steps"]
    config = OmegaConf.load(folder / "base_config.yaml")
    config.experiment.env_params.payload_mass_kg = result["mass_kg"]
    # Saved qpos/qref already contain both streams; no trajectory cache is needed.
    config.experiment.task_factory.params = {}
    print(f"Loading {task} model", flush=True)
    env = make_cpu(config)
    try:
        model = env._model
        data, reference = mujoco.MjData(model), mujoco.MjData(model)
        assert poses.shape[1] == model.nq
        assert model.ntendon > 0
        model.vis.global_.offwidth, model.vis.global_.offheight = 960, 480
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation, camera.distance = 135, -12, 3.1
        camera.lookat[:] = (*np.mean(poses[:, :2], axis=0), 0.94)
        option = mujoco.MjvOption()
        option.sitegroup[:] = 0
        option.tendongroup[:] = 1
        option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
        ghost_option = mujoco.MjvOption()
        ghost_option.sitegroup[:] = 0
        ghost_option.tendongroup[:] = 0
        ghost_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
        if task == "bend":
            box_qpos = int(model.joint("payload_free").qposadr[0])
            assert np.max(poses[:, box_qpos + 2]) - poses[0, box_qpos + 2] > 0.5
        video = output / f"{task}.mp4"
        pending = output / f"{task}.working.mp4"
        # Both saved streams have the same 100 Hz phase; sample without changing poses.
        dt = float(rollout["dt"])
        frames = np.minimum(np.arange(int(np.ceil(len(poses) * dt * 30))) / (30 * dt), len(poses) - 1)
        with (
            mujoco.Renderer(model, width=960, height=480, max_geom=20000) as renderer,
            imageio.get_writer(pending, fps=30, codec="libx264", quality=8, macro_block_size=16) as writer,
        ):
            for frame, index in enumerate(frames.astype(int)):
                data.qpos[:] = poses[index]
                reference.qpos[:] = references[index]
                mujoco.mj_forward(model, data)
                mujoco.mj_forward(model, reference)
                renderer.update_scene(data, camera=camera, scene_option=option)
                tendons = sum(
                    renderer.scene.geoms[i].objtype == mujoco.mjtObj.mjOBJ_TENDON for i in range(renderer.scene.ngeom)
                )
                assert tendons > 0, "Muscle paths must be rendered"
                add_reference(model, reference, ghost_option, renderer.scene, env.payload_body_id)
                image = renderer.render()
                writer.append_data(image)
                if frame in (0, len(frames) // 2, len(frames) - 1):
                    Image.fromarray(image).save(output / f"{task}_frame_{frame}.png")
                if frame % 30 == 0:
                    print(f"{task}: frame {frame + 1}/{len(frames)}", flush=True)
        pending.replace(video)
        print(f"Rendered {video.relative_to(ROOT)}", flush=True)
    finally:
        env.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/readme_object_demos")
    parser.add_argument("--tasks", nargs="+", choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        render(task, args.output)


if __name__ == "__main__":
    main()
