"""Render the README's model, device and terrain previews from project assets.

Run from the repository root with the installed model packages. This renders
static poses only: no policy, training data or simulation stepping is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import mujoco
from PIL import Image

from musclemimic.scenes.composition import build_msk_device_terrain_spec

ROOT = Path(__file__).resolve().parents[1]
MSK_KEYS = ("myofullbody", "myolegs", "myolegs26", "myoarms")
DEVICE_KEYS = ("Hippo_L1", "HipExoskeleton_L1", "ElbowExoskeleton_L1", "franka_emika_panda")
TERRAINS = {
    "flat": {"terrain": "flat"},
    "slope": {"terrain": "slope", "deg": 15},
    "warehouse": {"terrain": "warehouse"},
    "stairs": {
        "terrain_name": "readme_stairs",
        "grid": {"rows": 1, "cols": 1, "tile_size": [4.0, 4.0]},
        "border": {"width": 0.3},
        "tiles": [{"row": 0, "col": 0, "type": "stairs"}],
    },
}


def render(kind, key, output):
    msk = key if kind == "msk" else "myofullbody"
    device = key if kind == "device" else None
    terrain = TERRAINS[key] if kind == "terrain" else TERRAINS["flat"]
    composition = build_msk_device_terrain_spec(msk, device, terrain)
    model = composition.spec.compile()
    data = mujoco.MjData(model)
    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand >= 0:
        mujoco.mj_resetDataKeyframe(model, data, stand)

    if key == "franka_emika_panda":
        # Set a visible static ready posture, without running a controller.
        pose = (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8)
        for index, value in enumerate(pose, 1):
            joint = model.joint(f"franka_emika_panda_robot0_joint{index}")
            data.qpos[joint.qposadr[0]] = value
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.0, 0.0, 0.96)
    camera.distance = 2.85
    camera.azimuth = 135
    camera.elevation = -12
    if kind == "msk" and key == "myoarms":
        camera.lookat[:] = (0.0, 0.0, 1.3)
        camera.distance = 1.95
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0  # Hide auxiliary robot collision geometry.
    option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = kind == "msk"
    human_bodies = set(composition.components["msk"].bodies)
    human_geoms = [
        i
        for i in range(model.ngeom)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) in human_bodies
    ]
    if kind == "device":
        model.geom_rgba[human_geoms, 3] = 0.16
        if key == "ElbowExoskeleton_L1":
            camera.lookat[:] = (0.0, 0.0, 1.27)
            camera.distance = 1.85
        elif key == "franka_emika_panda":
            model.geom_rgba[human_geoms, 3] = 0.0
            # This authored scene keeps the collision meshes in group 0.
            # Render its detailed group-1 meshes without overlapping proxies.
            for geom_id in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                if name.startswith("franka_emika_panda_") and model.geom_group[geom_id] == 0:
                    model.geom_rgba[geom_id, 3] = 0.0
            camera.lookat[:] = data.body("franka_emika_panda_robot0_link0").xpos
            camera.lookat[2] += 0.2
            camera.distance = 2.5
            camera.azimuth = 120
            camera.elevation = -20
        else:
            camera.lookat[:] = (0.0, 0.0, 1.02)
            camera.distance = 1.75
    if kind == "terrain":
        if key == "warehouse":
            model.geom_rgba[human_geoms, 3] = 0.0
            camera.lookat[:] = (0.0, -1.3, 0.75)
            camera.distance = 17.0
            camera.azimuth = 110
            camera.elevation = -32
        elif key == "stairs":
            model.geom_rgba[human_geoms, 3] = 0.0
            camera.lookat[:] = (0.0, 0.0, 0.4)
            camera.distance = 7.0
            camera.azimuth = 135
            camera.elevation = -30
        else:
            camera.distance = 5.5
            camera.elevation = -18
            camera.azimuth = 110

    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    model.vis.headlight.ambient[:] = (0.35, 0.35, 0.35)
    model.vis.headlight.diffuse[:] = (0.75, 0.75, 0.75)
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        image = Image.fromarray(renderer.render())
        path = output / f"{kind}_{key}.png"
        image.save(path, optimize=True)
    return {
        "image": path.name,
        "kind": kind,
        "key": key,
        "msk_key": msk,
        "device_key": device,
        "terrain": terrain,
        "pose": "stand keyframe" if stand >= 0 else "model qpos0",
        "render_only_robot_pose": list(pose) if key == "franka_emika_panda" else None,
        "camera": {
            "lookat": camera.lookat.tolist(),
            "distance": camera.distance,
            "azimuth": camera.azimuth,
            "elevation": camera.elevation,
        },
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/readme")
    parser.add_argument("--only", nargs="*", help="Render only matching model/device/terrain keys")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {
            "mujoco_version": mujoco.__version__,
            "resolution": [960, 720],
            "description": "Static MuJoCo renders from project models, with no policy or dynamics replay; device previews use a translucent human for context.",
            "images": [],
        }
    )
    records = {r["image"]: r for r in manifest["images"]}
    for kind, keys in (("msk", MSK_KEYS), ("device", DEVICE_KEYS), ("terrain", TERRAINS)):
        for key in keys:
            if args.only and key not in args.only:
                continue
            record = render(kind, key, args.output)
            records[record["image"]] = record
            print(f"Rendered {record['image']}", flush=True)
    manifest["images"] = list(records.values())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
