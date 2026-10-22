"""The optional GMR backend must use the same MSK provider as training."""

import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("general_motion_retargeting") is None,
    reason="Requires the optional gmr extra",
)
def test_gmr_import_and_target_without_another_model_provider():
    # A fresh process exercises GMR's first import, not cached module globals.
    code = """
import importlib.abc
import contextlib
import io
import json
import sys
from pathlib import Path
import mujoco
import myo_sim
import numpy as np

class ModelProviderGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("musclemimic_models"):
            raise AssertionError("GMR requested a second MSK provider")

sys.meta_path.insert(0, ModelProviderGuard())
from loco_mujoco.smpl import retargeting
retargeting._ensure_gmr_imports()
assert retargeting._GMR_INSTALLED
assert retargeting._GMR_SHAPE_FITTING_INSTALLED
actual = mujoco.MjModel.from_xml_path(str(retargeting.ROBOT_XML_DICT["myofullbody"]))
expected = myo_sim.load_spec("myofullbody").compile()
assert (actual.nq, actual.nv, actual.nu) == (expected.nq, expected.nv, expected.nu)
assert [actual.joint(i).name for i in range(actual.njnt)] == [expected.joint(i).name for i in range(expected.njnt)]
np.testing.assert_array_equal(actual.qpos0, expected.qpos0)
config_path = Path(retargeting.__file__).parent / "gmr_configs/smplh_to_myofullbody.json"
config = json.loads(config_path.read_text())
data = mujoco.MjData(actual)
mujoco.mj_forward(actual, data)
frame = {}
for table in ("ik_match_table1", "ik_match_table2"):
    for body_name, (human_name, *_) in config[table].items():
        body_id = actual.body(body_name).id
        frame[human_name] = (data.xpos[body_id].copy(), data.xquat[body_id].copy())
with contextlib.redirect_stdout(io.StringIO()):
    solver = retargeting.GeneralMotionRetargeting(
        src_human="smplh", tgt_robot="myofullbody", verbose=False,
        ik_config_path=str(config_path),
    )
    for _ in range(3):
        qpos, diagnostics = solver.retarget(frame, offset_to_ground=False)
        assert qpos.shape == (expected.nq,)
        assert np.isfinite(qpos).all()
assert not any(name.startswith("musclemimic_models") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTORCH_JIT": "0"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
