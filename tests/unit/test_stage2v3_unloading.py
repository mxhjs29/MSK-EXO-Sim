"""Physical consistency checks for isolated unloading experiments."""
import numpy as np
import mujoco
from musclemimic.controllers.muscle_recruitment.dynamics import activation_step,activation_future,excitation_for_target
from musclemimic.controllers.muscle_recruitment.kinematics import independent_directions


def test_activation_matches_mujoco():
    rng=np.random.default_rng(41)
    a=rng.uniform(0,1,100);u=rng.uniform(-1,1,100)
    times=np.tile([.015,.05],(100,1));dt=1/1500
    expected=a+dt*np.array([mujoco.mju_muscleDynamics(float(ctrl),float(act),np.array([.015,.05,0.])) for act,ctrl in zip(a,u)])
    np.testing.assert_allclose(activation_step(a,u,times,dt),expected,atol=1e-14,rtol=0)


def test_reachable_activation_inverse():
    rng=np.random.default_rng(19)
    a=rng.uniform(0,1,60);u=rng.uniform(0,1,60)
    times=np.tile([.015,.05],(60,1));dt=1/1500
    target=activation_future(a,u,times,dt)
    recovered=excitation_for_target(a,target,times,dt)
    assert np.all((recovered>=0)&(recovered<=1))
    np.testing.assert_allclose(activation_future(a,recovered,times,dt),target,atol=3e-8,rtol=0)


def test_equality_projection_preserves_virtual_work():
    xml='''<mujoco><worldbody><body><joint name="a"/><geom size=".1"/>
    <body pos="0 0 .3"><joint name="b"/><geom size=".1"/>
    <body pos="0 0 .3"><joint name="c"/><geom size=".1"/></body></body></body></worldbody>
    <equality><joint joint1="b" joint2="a" polycoef="0 2 0 0 0"/>
    <joint joint1="c" joint2="b" polycoef="0 .5 0 0 0"/></equality></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    directions=independent_directions(m,d)
    np.testing.assert_array_equal(directions,np.array([[1.],[2.],[1.]]))
    torque=np.array([3.,4.,5.]);dq=np.array([.03])
    np.testing.assert_allclose(torque@(directions@dq),(directions.T@torque)@dq)
