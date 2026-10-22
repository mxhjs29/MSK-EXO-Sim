"""Regression coverage for publishing, compatibility imports and shared worlds."""
import importlib
import numpy as np
import pytest
import mujoco
from musclemimic.agents.specs import AgentSpec
from musclemimic.agents.routing import ActionRouter
from musclemimic.environments.worlds.runtime import WorldRuntime
from musclemimic.scenes.schemas import EntitySpec, SceneSpec
from musclemimic.scenes.builder import build_scene
from musclemimic.core.sensors.foot_contact import FootContactSensor


@pytest.mark.parametrize('old,new',[
 ('musclemimic.models.composed_spec','musclemimic.scenes.composition'),
 ('musclemimic.core.reward.trajectory_based','musclemimic.rewards.common.imitation'),
 ('musclemimic.experiments.payload_refinement','musclemimic.runner.workflows.payload_refinement'),
 ('musclemimic.web_viewer.c3d.markers','musclemimic.visualization.web_viewer.c3d.markers'),
 ('musclemimic.web_viewer.c3d_to_smpl','musclemimic.visualization.web_viewer.c3d_to_smpl'),
])
def test_compatibility_module_identity(old,new):
    assert importlib.import_module(old) is importlib.import_module(new)


def two_agent_model():
    return mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002"/><worldbody>
      <body><joint name="a"/><geom size=".1"/></body>
      <body pos="1 0 0"><joint name="b"/><geom size=".1"/></body>
      </worldbody><actuator><motor joint="a"/><motor joint="b"/></actuator></mujoco>''')


def test_shared_world_advances_once_and_matches_joint_command():
    m=two_agent_model()
    agents=[AgentSpec('human','human','human_ppo',(0,)),AgentSpec('device','device','device_fixed',(1,),trainable=False)]
    world=WorldRuntime(m,agents,n_substeps=3);world.reset()
    expected=mujoco.MjData(m);expected.ctrl[:]=[.2,-.1];mujoco.mj_step(m,expected,nstep=3)
    state=world.step({'human':np.array([.2]),'device':np.array([-.1])})
    np.testing.assert_array_equal(state.qpos,expected.qpos)
    assert state.time==pytest.approx(.006)
    prior=state.qpos.copy()
    with pytest.raises(ValueError):world.step({'human':[.2]})
    np.testing.assert_array_equal(state.qpos,prior)
    world.reset();assert world.data.time==0 and np.all(world.data.ctrl==0)


def test_routing_rejects_overlapping_ownership():
    with pytest.raises(ValueError):ActionRouter([AgentSpec('a','a','p',(0,)),AgentSpec('b','b','q',(0,))],2)


def test_scene_compiles_leg_device_slope_with_named_ownership():
    scene=build_scene(SceneSpec('test',(EntitySpec('human','msk','myolegs26'),EntitySpec('device','device','Hippo_L1')),{'terrain':'slope','deg':5}))
    owned=scene.ownership
    assert len(owned['human'].actuator_indices)==26
    assert len(owned['device'].actuator_indices)>0
    assert not set(owned['human'].actuator_indices)&set(owned['device'].actuator_indices)
    sensor=FootContactSensor(scene.model)
    data=mujoco.MjData(scene.model);mujoco.mj_forward(scene.model,data)
    size=mujoco.mj_stateSize(scene.model,mujoco.mjtState.mjSTATE_INTEGRATION)
    before=np.zeros(size);after=np.zeros(size)
    mujoco.mj_getState(scene.model,data,before,mujoco.mjtState.mjSTATE_INTEGRATION)
    assert sensor.sample(data).shape==(4,)
    mujoco.mj_getState(scene.model,data,after,mujoco.mjtState.mjSTATE_INTEGRATION)
    np.testing.assert_array_equal(before,after)


def test_unknown_or_unsupported_scene_is_rejected():
    with pytest.raises(ValueError):build_scene(SceneSpec('two',(EntitySpec('a','msk','myoarms'),EntitySpec('b','msk','myoarms'))))


def test_base_reward_is_evaluated_once_before_task_bonus():
    from musclemimic.rewards.composer import RewardComposer
    count=[]
    def common(*args):count.append(1);return 0.,'updated',{'reward_total':0.}
    def bonus(*args):assert args[-2]=='updated';return .7,{'task_bonus':.7}
    value,carry,terms=RewardComposer(common,[bonus])(*([None]*10))
    assert value==.7 and carry=='updated' and count==[1] and terms['reward_total']==.7
