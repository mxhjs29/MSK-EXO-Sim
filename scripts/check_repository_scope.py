"""Check proposed public files and built distributions for local-only content."""
import argparse
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT / "repository_scope.json").read_text())


def forbidden(name, *, archive=False):
    path = PurePosixPath(name)
    if any(part.startswith(".venv") or part == "__pycache__" for part in path.parts):
        return True
    if not archive and any(
        str(path) == p or str(path).startswith(p + "/") for p in SCOPE.get("included_exceptions", [])
    ):
        return False
    excluded = SCOPE.get("archive_excluded_directories", SCOPE["excluded_directories"]) if archive else SCOPE["excluded_directories"]
    return any(str(path) == p or str(path).startswith(p + "/") for p in excluded)


def check_source():
    result = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    paths = sorted(set(result.decode().split("\0")) - {""})
    errors = []
    for name in paths:
        file = ROOT / name
        if not file.exists():  # tracked paths removed by this uncommitted migration
            continue
        if forbidden(name):
            errors.append("Local-only file in public candidate: " + name)
        if file.is_symlink() and not file.resolve().is_relative_to(ROOT):
            errors.append("External symlink: " + name)
        if file.is_symlink() and file.resolve().is_relative_to(ROOT) and forbidden(str(file.resolve().relative_to(ROOT))):
            errors.append("Symlink to local content: " + name)
    return paths, errors


def check_archive(path):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(path) as archive:
            names = ["/".join(PurePosixPath(n).parts[1:]) for n in archive.getnames()]
    errors = ["Local-only archive member: " + n for n in names if forbidden(n, archive=True)]
    return names, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="*", type=Path)
    args = parser.parse_args()
    paths, errors = check_source()
    counts = {"public_candidate_files": len(paths)}
    for archive in args.archives:
        names, failures = check_archive(archive)
        counts[str(archive)] = len(names)
        errors.extend(failures)
    print(json.dumps({**counts, "errors": errors}, ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
