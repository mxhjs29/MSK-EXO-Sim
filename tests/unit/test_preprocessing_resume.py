"""Cleanup must preserve resuming the unchanged myo_sim conversion workflow."""

import hashlib
import json
from pathlib import Path

from musclemimic.data.preprocessing.retarget_approved_continuous import resume_fingerprint_matches


def fingerprints():
    folder = Path(__file__).resolve().parents[2] / "musclemimic/data/preprocessing"
    revision = json.loads((folder / "relocation_fingerprints.json").read_text())["compatible_revisions"][-1]
    current_hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in revision["after"]}
    return (
        {"model": "new", "annotations_sha256": "source", "implementation_sha256": revision["before"]},
        {"model": "new", "annotations_sha256": "source", "implementation_sha256": current_hashes},
    )


def test_resume_accepts_verified_cleanup_with_identical_inputs():
    saved, current = fingerprints()
    assert resume_fingerprint_matches(saved, current)


def test_resume_rejects_changed_inputs():
    saved, current = fingerprints()
    current["annotations_sha256"] = "different-source"
    assert not resume_fingerprint_matches(saved, current)


def test_resume_rejects_unverified_code_changes():
    saved, current = fingerprints()
    current["implementation_sha256"]["retarget_selected_myofullbody.py"] = "different-code"
    assert not resume_fingerprint_matches(saved, current)
