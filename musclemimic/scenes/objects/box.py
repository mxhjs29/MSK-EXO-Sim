"""Physical object construction, independent of policy and reward."""
import numpy as np
import mujoco

def add_friction_box(spec, config):
    for side in ['r','l']:
        g=spec.geom('radius_coll_'+side)
        # Composed specs may have already compiled fromto into pos/quat/size.
        ends=np.asarray(g.fromto).reshape(2,3).copy()
        if not np.isfinite(ends).all() or np.linalg.norm(ends[1]-ends[0])<1e-8:
            rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,np.asarray(g.quat))
            half_axis=rotation.reshape(3,3)[:,2]*g.size[1]
            ends=np.array([np.asarray(g.pos)-half_axis,np.asarray(g.pos)+half_axis])
        # Capsule orientation can reverse at compilation. Radius body origin is proximal.
        if np.linalg.norm(ends[0])>np.linalg.norm(ends[1]):ends=ends[::-1]
        axis=ends[1]-ends[0]
        quat=np.zeros(4);mujoco.mju_quatZ2Vec(quat,axis)
        g.fromto=[np.nan,0,0,0,0,0]
        g.pos=ends[0]+axis*.31;g.quat=quat;g.size[1]=np.linalg.norm(axis)*.31
        for label,start,end,radius in [('mid',.45,.82,.024),('wrist',.72,1.,.017)]:
            spec.body('radius_'+side).add_geom(name='box_forearm_'+label+'_'+side,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                pos=ends[0]+axis*(start+end)/2,quat=quat,
                size=[radius,np.linalg.norm(axis)*(end-start)/2,0],
                mass=0,contype=g.contype,conaffinity=g.conaffinity,condim=g.condim,
                friction=g.friction,rgba=[.65,.65,.65,0],group=g.group)
    for key in spec.keys:
        key.qpos = []; key.qvel = []; key.ctrl = []
    common = dict(contype=4, conaffinity=1, condim=3, friction=[config.hand_friction, 0, 0],
                  priority=2, solref=[.01, 1], solimp=[.95, .99, .001, .5, 2])
    body = spec.worldbody.add_body(name='payload', pos=[0, 0, 1])
    body.add_freejoint(name='payload_free')
    body.add_geom(name='payload_box', type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=config.box_half_size, mass=config.payload_mass_kg, rgba=[.70, .43, .18, 1], **common)
    for sign, side in [(-1, 'r'), (1, 'l')]:
        body.add_site(name='payload_grip_'+side,
                      pos=[sign*(config.box_half_size[0]-.001), 0, .02], size=[.004, 0, 0])
        palm = spec.body('lunate_'+side)
        # A thin soft-tissue surface on the existing palm, massless to retain hand inertia.
        palm.add_geom(name='box_palm_pad_'+side, type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[-.001, -.045, sign*.014], size=[.026, .031, .008],
                      mass=0, rgba=[.70, .62, .56, .7], contype=8, conaffinity=4,
                      condim=3, friction=[config.hand_friction, 0, 0], priority=2,
                      solref=[.01, 1], solimp=[.95, .99, .001, .5, 2])
        palm.add_site(name='box_palm_'+side, pos=[-.001, -.045, sign*.022], size=[.004, 0, 0])
        # The engine reduces all subtree/object contacts per world, including Warp batches.
        spec.add_sensor(name='box_contact_depth_'+side,type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,objname='lunate_'+side,
            reftype=mujoco.mjtObj.mjOBJ_BODY,refname='payload',intprm=[8,1,1])
    body.add_site(name='payload_center', size=[.004, 0, 0])
    spec.worldbody.add_geom(name='box_support', type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=config.box_support_pos, size=config.box_support_half_size, contype=1, conaffinity=5,
        rgba=[.35, .38, .40, 1], friction=[.8, 0, 0])
    # No hand-object equality, excludes, adhesion actuators or applied support forces.
    return spec
