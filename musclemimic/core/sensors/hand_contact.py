import numpy as np
import mujoco
CONTACT_SLOTS = 24

def add_contact_feedback(spec, hand_names=('lunate_r','lunate_l'), human_name='Full Body', box_name='payload'):
    for side,body in zip(('r','l'),hand_names):
        # found + force + normal + tangent = 10 values per slot.
        spec.add_sensor(name='refine_contact_'+side,type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,objname=body,
            reftype=mujoco.mjtObj.mjOBJ_BODY,refname=box_name,intprm=[99,0,CONTACT_SLOTS])
    spec.add_sensor(name='refine_contact_human',type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_XBODY,objname=human_name,
        reftype=mujoco.mjtObj.mjOBJ_BODY,refname=box_name,intprm=[2,3,1])

class ContactReadout:
    def __init__(self,model):
        addresses=[model.sensor_adr[model.sensor('refine_contact_'+s).id] for s in ('r','l')]
        self.indices=np.array([a+np.arange(CONTACT_SLOTS*10) for a in addresses]).reshape(2,CONTACT_SLOTS,10)
        a=int(model.sensor_adr[model.sensor('refine_contact_human').id]);self.human=np.arange(a,a+3)

    def __call__(self,data,xp):
        raw=data.sensordata[self.indices]
        local=raw[:,:,1:4];normal=raw[:,:,4:7];tangent=raw[:,:,7:10]
        tangential=local[:,:,1:2]*tangent+local[:,:,2:3]*xp.cross(normal,tangent)
        hand=xp.sum(local[:,:,:1]*normal+tangential,axis=1)
        return dict(hand=hand,normal=xp.sum(local[:,:,0],axis=1),
            friction_z=xp.sum(tangential[:,:,2],axis=1),
            other=data.sensordata[self.human]-xp.sum(hand,axis=0),count=raw[:,0,0])
