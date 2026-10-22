"""Physical object construction, independent of policy and reward."""
import numpy as np
import mujoco

def add_payload(spec, config):
    common=dict(contype=4,conaffinity=1,friction=[.8,.01,.001],condim=3,group=0)
    if config.payload_kind=='dumbbell':
        site=spec.site('S_grasp_r')
        body=spec.body('trapezium_r').add_body(name='payload',pos=site.pos+config.payload_mount_offset)
        body.add_geom(name='payload_handle',type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[-.085,0,0,.085,0,0],size=[.013,0,0],mass=config.payload_mass_kg*.10,
            rgba=[.35,.38,.43,1],**common)
        for side in [-1,1]:
            body.add_geom(name='payload_plate_'+str(side),type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[side*.105,0,0],quat=[np.sqrt(.5),0,np.sqrt(.5),0],size=[.055,.022,0],
                mass=config.payload_mass_kg*.45,rgba=[.1,.36,.65,1],**common)
    else:
        for key in spec.keys:key.qpos=[];key.qvel=[];key.ctrl=[]
        body=spec.worldbody.add_body(name='payload',pos=[0,0,1])
        body.add_freejoint(name='payload_free')
        span=config.payload_grip_span_m
        body.add_geom(name='payload_box',type=mujoco.mjtGeom.mjGEOM_BOX,size=[span/2-.014,.10,.08],
            mass=config.payload_mass_kg*.9,rgba=[.70,.43,.18,1],**common)
        for side,s in [(-1,'r'),(1,'l')]:
            body.add_geom(name='payload_handle_'+s,type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[side*span/2,-.045,.055,side*span/2,.045,.055],size=[.009,0,0],
                mass=config.payload_mass_kg*.05,rgba=[.23,.21,.18,1],**common)
            body.add_site(name='payload_grip_'+s,pos=[side*span/2,0,.055],size=[.008,0,0],rgba=[0,1,0,1])
            spec.add_equality(name='payload_hold_'+s,type=mujoco.mjtEq.mjEQ_CONNECT,
                objtype=mujoco.mjtObj.mjOBJ_SITE,name1='S_grasp_'+s,name2='payload_grip_'+s,
                solref=[.02,1],solimp=[.95,.99,.001,.5,2])
    body.add_site(name='payload_center',size=[.008,0,0],rgba=[1,0,0,1])
    # Ideal attachment replaces finger contact mechanics at the holding hands.
    # Keep collisions with the ground and the rest of the human body.
    compiled=spec.compile()
    hands=[compiled.body('lunate_'+s).id for s in (['r'] if config.payload_kind=='dumbbell' else ['r','l'])]
    for b in range(1,compiled.nbody):
        if compiled.body(b).name=='payload':continue
        ancestor=b
        while ancestor>0 and ancestor not in hands:ancestor=int(compiled.body_parentid[ancestor])
        if ancestor in hands:
            spec.add_exclude(bodyname1='payload',bodyname2=compiled.body(b).name)
    return spec
