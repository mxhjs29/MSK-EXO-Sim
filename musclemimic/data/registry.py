"""Versioned external dataset locations; actual datasets remain outside Git."""
from dataclasses import dataclass
from pathlib import Path
from .fingerprints import digest


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo_id: str
    revision: str
    allow_patterns: tuple[str, ...]
    local_dir: Path

    def __post_init__(self):
        if not self.repo_id or not self.revision or self.revision in {"main", "master"}:
            raise ValueError("Dataset needs an explicit repository and pinned revision")


def verify_file(path, sha256):
    actual = digest(path)
    if actual != sha256:
        raise ValueError(f"Checksum mismatch for {path}")
    return Path(path)
