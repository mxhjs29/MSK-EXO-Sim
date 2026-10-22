from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from loco_mujoco.smpl.retargeting import _initialize_gmr_root_from_target


class MuJoCoConfiguration:
    """Exercise root alignment with native FK, without the optional IK solver."""

    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        mujoco.mj_forward(model, self.data)

    def update(self, q):
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

    def get_transform_frame_to_world(self, name, frame_type):
        assert frame_type == "body"
        body_id = self.model.body(name).id
        return SimpleNamespace(
            translation=lambda: self.data.xpos[body_id].copy(),
            rotation=lambda: Rotation.from_matrix(self.data.xmat[body_id].reshape(3, 3)),
        )


@pytest.fixture(params=["mujoco", "mink"])
def configuration_factory(request):
    if request.param == "mujoco":
        return MuJoCoConfiguration
    mink = pytest.importorskip("mink", reason="Mink integration requires the optional gmr extra")
    return mink.Configuration


@pytest.mark.parametrize("yaw", [0, 90, 180, -135])
def test_root_alignment_handles_heading_and_offset_pelvis(yaw, configuration_factory):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="base" pos="1 2 1"><freejoint/><geom size=".1"/>
        <body name="pelvis" pos=".1 .2 .3" euler="10 20 30"><geom size=".1"/>
          <body name="leg"><joint name="hip"/><geom size=".1"/></body>
        </body>
      </body></worldbody></mujoco>""")
    configuration = configuration_factory(model)
    q = configuration.data.qpos.copy()
    q[-1] = 0.25
    configuration.update(q=q)
    position = np.array([4.0, -3.0, 1.2])
    rotation = Rotation.from_euler("xyz", [5, -10, yaw], degrees=True)
    retarget = SimpleNamespace(
        configuration=configuration,
        robot_root_name="pelvis",
        human_root_name="pelvis",
        scaled_human_data={"pelvis": (position, rotation.as_quat(scalar_first=True))},
        update_targets=lambda frame, offset: None,
    )
    _initialize_gmr_root_from_target(retarget, {})
    pelvis_id = model.body("pelvis").id
    np.testing.assert_allclose(configuration.data.xpos[pelvis_id], position, atol=1e-10)
    np.testing.assert_allclose(configuration.data.xmat[pelvis_id].reshape(3, 3), rotation.as_matrix(), atol=1e-10)
    assert configuration.data.qpos[-1] == 0.25
