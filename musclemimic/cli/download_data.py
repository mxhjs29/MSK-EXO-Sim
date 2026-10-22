"""Download original source data from the pinned existing Hugging Face dataset."""
import argparse
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download
from musclemimic.utils.paths import config_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["all", "amass", "gait120"], default="all")
    parser.add_argument("--destination", type=Path, default=Path("datasets"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=config_root() / "datasets/manifest.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.manifest.read_text())["datasets"]["raw_motion_sources"]
    revision = config.get("revision")
    if not revision or revision in {"main", "master"}:
        raise ValueError("Dataset manifest needs a fixed uploaded Hugging Face revision")
    families = config["families"] if args.family == "all" else [args.family]
    for family in families:
        group = config["groups"][family]
        target = args.destination / group["local_subdir"]
        snapshot_download(repo_id=config["repo_id"], repo_type="dataset", revision=revision,
                          allow_patterns=group["allow_patterns"], local_dir=str(target), cache_dir=args.cache_dir)
        print(f"Downloaded {family} from {config['repo_id']}@{revision} into {target.resolve()}", flush=True)
    if "gait120" in families:
        import hashlib
        import json
        folder=args.destination/'Gait120_101_to_110'
        inventory=json.loads((folder/'file_manifest.json').read_text())
        for row in inventory['files']:
            file=args.destination/row['path']
            if not file.is_file() or file.stat().st_size!=row['bytes']:
                raise ValueError(f"Downloaded Gait120 size mismatch: {file}")
            h=hashlib.sha256()
            with file.open('rb') as handle:
                for block in iter(lambda:handle.read(4*1024*1024),b''):h.update(block)
            if h.hexdigest()!=row['sha256']:raise ValueError(f"Downloaded Gait120 checksum mismatch: {file}")
        print(f"Gait120 original files verified: {len(inventory['files'])}")


if __name__ == "__main__":
    main()
