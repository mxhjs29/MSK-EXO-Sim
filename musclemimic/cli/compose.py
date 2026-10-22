"""Compile and inspect a public scene configuration."""
import argparse
import json
from pathlib import Path
from dataclasses import asdict
import yaml
from musclemimic.scenes.schemas import SceneSpec
from musclemimic.scenes.builder import build_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    scene = build_scene(SceneSpec.from_dict(yaml.safe_load(args.config.read_text())))
    print(json.dumps({"nq": scene.model.nq, "nv": scene.model.nv, "nu": scene.model.nu,
                      "ownership": {k: asdict(v) for k, v in scene.ownership.items()}}, indent=2))


if __name__ == "__main__":
    main()
