"""Create a reviewable public checkout from current tracked AND untracked source."""
import argparse
import json
import shutil
from pathlib import Path
from check_repository_scope import ROOT, check_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination == ROOT or (destination.exists() and any(destination.iterdir())):
        raise ValueError("Destination must be empty and different from the working repository")
    paths, errors = check_source()
    if errors:
        raise ValueError("\n".join(errors))
    count = 0
    for name in paths:
        source = ROOT / name
        if not source.exists():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(source.readlink(), target_is_directory=source.is_dir())
        elif source.is_file():
            shutil.copy2(source, target)
        count += 1
    print(json.dumps({"destination": str(destination), "files": count}, indent=2))


if __name__ == "__main__":
    main()
