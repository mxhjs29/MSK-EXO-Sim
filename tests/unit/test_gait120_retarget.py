from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from musclemimic.retargeting.gait120_retarget import constrain_coupled_roots, project_joint_couplings
from musclemimic.web_viewer import c3d_to_smpl


def test_joint_projection_respects_reference_offsets_and_dependency_order():
    model=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
      <worldbody><body><joint name="a" ref="0.3"/><geom size=".1"/>
        <body pos="0 0 .3"><joint name="b" ref="-0.2"/><geom size=".1"/>
          <body pos="0 0 .3"><joint name="c" ref="0.1"/><geom size=".1"/></body>
        </body></body></worldbody>
      <equality><joint joint1="c" joint2="b" polycoef=".2 .5 .1 0 0"/>
      <joint joint1="b" joint2="a" polycoef="-.1 .7 0 0 0"/></equality></mujoco>''')
    source=np.array([[.6,1.,2.],[.8,-1.,-2.]])
    projected=project_joint_couplings(model,source)
    np.testing.assert_array_equal(projected[:,0],source[:,0])
    np.testing.assert_array_equal(source[:,1],[1.,-1.])
    data=mujoco.MjData(model)
    for q in projected:
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        np.testing.assert_allclose(data.efc_pos[data.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY],0,atol=1e-12)


def test_pose_warm_start_resets_at_independent_clip_boundary(monkeypatch):
    torch = pytest.importorskip("torch", reason="Marker pose fitting requires the optional smpl extra")
    model=SimpleNamespace(get_joints_verts=lambda *a,**kw:(torch.zeros(1,3,3),None))
    markers=SimpleNamespace(vids=[0,1,2])
    monkeypatch.setattr(c3d_to_smpl.SurfaceMarkerModel,"compute_dynamic_frame_vids",lambda *a:np.array([[0,1,2]]*3))
    monkeypatch.setattr(c3d_to_smpl,"_initialize_pose_and_trans",lambda **kw:(torch.zeros(1,66),torch.zeros(1,3)))
    previous=[]
    def optimize(**kw):
        if kw["body_pose_weight_scale"]==1:
            previous.append(None if kw["prev_pose"] is None else float(kw["prev_pose"][0,0]))
        value=kw["obs_frame"][0,0]
        return torch.full((1,66),float(value)),torch.zeros(1,3),0.
    monkeypatch.setattr(c3d_to_smpl,"_optimize_frame",optimize)
    observed=torch.arange(1,5,dtype=torch.float32)[:,None,None].expand(4,3,3)
    poses,_,_=c3d_to_smpl._fit_stageii(smpl=model,marker_model=markers,observed_t=observed,
        availability=np.ones((4,3),bool),betas_opt=torch.zeros(1,16),markers_latent_opt=torch.zeros(3,3),
        pose_dim=66,body_end=66,stage2_iters=10,device="cpu",sequence_starts=(0,2))
    assert previous==[None,1.,None,3.]
    np.testing.assert_array_equal(poses[:,0],[1,2,3,4])


def test_coupled_limits_keep_disconnected_feasible_intervals():
    model=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
      <worldbody><body><joint name="a" range="-2 2"/><geom size=".1"/>
        <body pos="0 0 .3"><joint name="b" range=".25 1"/><geom size=".1"/></body>
      </body></worldbody><equality><joint joint1="b" joint2="a" polycoef="0 0 1 0 0"/></equality></mujoco>''')
    limited,_=constrain_coupled_roots(model,np.array([[-1.5,0.],[-.2,0.],[.2,0.],[1.5,0.]]))
    q=project_joint_couplings(model,limited)
    np.testing.assert_allclose(q[:,0],[-1,-.5,.5,1])
    np.testing.assert_allclose(q[:,1],[1,.25,.25,1])


def test_c3d_adapter_preserves_marker_frame_and_existing_options(monkeypatch):
    points=np.arange(18,dtype=float).reshape(2,3,3)
    labels=["LASI","RASI","C7"]
    monkeypatch.setattr(c3d_to_smpl,"load_c3d_markers",lambda _: (points,labels,100.))
    def fit(raw_positions,raw_labels,fps,**kwargs):
        np.testing.assert_array_equal(raw_positions,points)
        assert raw_labels==labels and fps==100.
        assert kwargs["stage2_iters"]==25 and kwargs["optimize_toes"]
        return {"ok":True}
    monkeypatch.setattr(c3d_to_smpl,"fit_smpl_to_markers",fit)
    assert c3d_to_smpl.fit_smpl_to_c3d("sample.c3d","models",stage2_iters=25,optimize_toes=True)=={"ok":True}


def test_reference_model_repairs_dependent_knee_ranges_without_changing_base():
    from musclemimic.environments.humanoids.myofullbody_assist import MyoFullBodyAssistEnv
    from musclemimic.retargeting.gait120_retarget import make_reference_env
    base=MyoFullBodyAssistEnv(headless=True)
    before=base._model.jnt_range.copy()
    env,changes=make_reference_env()
    assert "knee_angle_translation2_l" in {c["joint"] for c in changes}
    np.testing.assert_array_equal(base._model.jnt_range,before)
    assert env._model.nq==base._model.nq==93
    assert env._model.nu==base._model.nu==354
    q=np.broadcast_to(env._model.qpos0,(101,env._model.nq)).copy()
    for side in ["r","l"]:
        joint=env._model.joint("knee_angle_"+side)
        q[:,joint.qposadr[0]]=np.linspace(*joint.range,101)
        np.testing.assert_array_equal(joint.range,base._model.joint(joint.name).range)
    q=project_joint_couplings(env._model,q)
    for change in changes:
        joint=env._model.joint(change["joint"]);values=q[:,joint.qposadr[0]]
        assert values.min()>=joint.range[0]-1e-10
        assert values.max()<=joint.range[1]+1e-10
