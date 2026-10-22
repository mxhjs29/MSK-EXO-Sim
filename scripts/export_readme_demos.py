"""Export four trained-policy demonstrations for the public README.

Requires the selected local evaluation videos and FFmpeg. For bend/lift, run
render_readme_object_demos.py first to replay the saved poses with muscles and
their phase-aligned reference. This script only crops, resamples and encodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMOS = (
    {
        "task": "walk",
        "title": "行走",
        "source": "outputs/fullbody_walk_kit11/run/validation_42_t2621440000_20261002_153426/MyoFullBodyAssistEnv.mp4",
        "checkpoint": "outputs/fullbody_walk_kit11/checkpoints/fullbody_walk_kit11/checkpoint_10000",
        "crop": "crop=1280:640:0:80",
        "mass_kg": None,
    },
    {
        "task": "run",
        "title": "跑步",
        "source": "outputs/fullbody_run_h/run/validation_64_t2621440000_20261003_020431/MyoFullBodyAssistEnv.mp4",
        "checkpoint": "outputs/fullbody_run_h/checkpoints/fullbody_run_h/checkpoint_10000",
        "crop": "crop=1280:640:0:80",
        "mass_kg": None,
    },
    {
        "task": "bend",
        "title": "弯腰搬箱",
        "source": "outputs/readme_object_demos/bend.mp4",
        "rollout": "outputs/payload_refinement_v4/bend/evaluations/pose_1kg_1/start_0_rollout.npz",
        "checkpoint": "outputs/payload_refinement_v4/bend/training/pose_1kg_1/checkpoints/pose_1kg_1/checkpoint_200",
        "evaluation": "outputs/payload_refinement_v4/bend/evaluations/pose_1kg_1/results.json",
        "crop": "crop=960:480:0:0",
        "mass_kg": 1.0,
    },
    {
        "task": "lift",
        "title": "哑铃举臂",
        "source": "outputs/readme_object_demos/lift.mp4",
        "rollout": "outputs/payload_refinement_v3/lift/evaluations/refine_2kg_1/start_0_rollout.npz",
        "checkpoint": "outputs/payload_refinement_v3/lift/training/refine_2kg_1/checkpoints/refine_2kg_1/checkpoint_200",
        "evaluation": "outputs/payload_refinement_v3/lift/evaluations/refine_2kg_1/results.json",
        "crop": "crop=960:480:0:0",
        "mass_kg": 2.0,
    },
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path):
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration:stream=width,height,r_frame_rate,codec_name",
                "-of",
                "json",
                str(path),
            ]
        )
    )


def export(demo, output):
    source = ROOT / demo["source"]
    if not source.is_file() or not (ROOT / demo["checkpoint"]).exists():
        raise FileNotFoundError(f"Local recording/checkpoint unavailable: {demo['task']}")
    evidence = None
    if demo.get("evaluation"):
        evaluation = json.loads((ROOT / demo["evaluation"]).read_text())
        assert Path(evaluation["checkpoint"]).resolve() == (ROOT / demo["checkpoint"]).resolve()
        evidence = next(r for r in evaluation["records"] if r["start"] == 0)
        assert evidence["steps"] == evidence["expected_steps"] and evidence["termination"] == "reference_end"
        if demo["task"] == "bend":
            heights = [r["height_m"] for r in evidence["trajectory"]]
            assert max(heights) - heights[0] > 0.5, "Selected demo must lift the box"
            evidence = {
                "start": 0,
                "steps": evidence["steps"],
                "termination": evidence["termination"],
                "initial_box_height_m": heights[0],
                "maximum_box_height_m": max(heights),
            }
        else:
            evidence = {k: evidence[k] for k in ("start", "steps", "termination")}

    video = output / f"{demo['task']}.mp4"
    preview = output / f"{demo['task']}.gif"
    filters = (
        demo["crop"] + ",fps=30,scale=960:480:force_original_aspect_ratio=decrease,pad=960:480:(ow-iw)/2:(oh-ih)/2"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-filter_threads",
            "1",
            "-i",
            str(source),
            "-an",
            "-vf",
            filters,
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "medium",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-filter_complex_threads",
            "1",
            "-i",
            str(video),
            "-filter_complex",
            "[0:v]fps=12,scale=480:240:flags=lanczos,split[a][b];"
            "[a]palettegen=max_colors=128:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
            "-loop",
            "0",
            str(preview),
        ],
        check=True,
    )
    original, published = probe(source), probe(video)
    assert abs(float(original["format"]["duration"]) - float(published["format"]["duration"])) < 0.05
    assert published["streams"][0]["width"] == 960 and published["streams"][0]["height"] == 480
    record = dict(
        demo,
        source_sha256=digest(source),
        **({"rollout_sha256": digest(ROOT / demo["rollout"])} if demo.get("rollout") else {}),
        evidence=evidence,
        duration_s=float(published["format"]["duration"]),
        mp4={"file": video.name, "sha256": digest(video), "bytes": video.stat().st_size},
        gif={"file": preview.name, "sha256": digest(preview), "bytes": preview.stat().st_size},
    )
    print(f"Exported {demo['task']}: {record['duration_s']:.2f}s", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/demos")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = [export(demo, args.output) for demo in DEMOS]
    manifest = {
        "model": "myofullbody from models/myo_sim",
        "kind": "trained PPO evaluation recordings",
        "editing": "Full-duration trained-policy motions with moving purple references. Bend/lift replay the saved qpos/qref with visible muscle paths in one full-body view; no new poses generated.",
        "object_rendering": {
            "script": "scripts/render_readme_object_demos.py",
            "layout": "Single full-body view with phase-aligned human reference overlay",
            "muscle_paths_visible": True,
            "reference_rgba": [0.471, 0.38, 0.812, 0.35],
        },
        "mp4": "H.264 / yuv420p / 960x480 / 30fps / faststart",
        "preview": "Looping GIF / 480x240 / 12fps",
        "demos": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
