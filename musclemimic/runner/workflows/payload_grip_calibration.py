"""Fit a comfortable ideal-grip reference to the actual collision geometry."""
from musclemimic.runner.workflows.payload_finetune import *
from scipy.optimize import least_squares

def main():
    e=make_cpu(OmegaConf.load(OUT/'lift/base_config.yaml'));m,d=e._model,e._data
    d.qpos[:]=e.th.traj.data.qpos[108]
    first=m.joint('cmc_flexion').id;last=m.joint('md5_flexion_r').id
    js=list(range(first,last+1));names=[m.joint(j).name for j in js]
    adr=m.jnt_qposadr[js];bounds=m.jnt_range[js]
    handbody=m.body('lunate_r').id
    def hand(b):
        while b>0:
            if b==handbody:return True
            b=int(m.body_parentid[b])
        return False
    geoms=[i for i in range(m.ngeom) if hand(int(m.geom_bodyid[i])) and m.geom_contype[i] and not m.geom(i).name.startswith('payload')]
    handle=m.geom('payload_handle').id
    tips=[i for i,g in enumerate(geoms) if any(s in m.geom(g).name for s in ['distph','distal_thumb','5distph'])]
    preferred=[]
    for n in names:
        if 'abduction' in n:v=0.
        elif n.startswith('mcp'):v=.8
        elif n.startswith('pm'):v=1.1
        elif n.startswith('md'):v=.7
        else:v=-.3
        preferred.append(v)
    preferred=np.array(preferred);base=m.body_pos[e.payload_body_id].copy()
    def residual(v):
        d.qpos[adr]=v[:-3];m.body_pos[e.payload_body_id]=base+v[-3:]
        mujoco.mj_forward(m,d)
        dist=np.array([mujoco.mj_geomDistance(m,d,g,handle,.1,None) for g in geoms])
        return np.r_[np.minimum(dist-.001,0)*1000,(dist[tips]-.002)*80,(v[:-3]-preferred)*.15,v[-3:]*10]
    fit=least_squares(residual,np.r_[np.clip(preferred,bounds[:,0]+1e-4,bounds[:,1]-1e-4),[0,0,0]],
        bounds=(np.r_[bounds[:,0],[-.025,-.025,-.025]],np.r_[bounds[:,1],[.025,.025,.025]]),max_nfev=100)
    residual(fit.x)
    save_json(OUT/'audit/grip_fit.json',dict(joint_angles_rad=dict(zip(names,fit.x[:-3].tolist())),
        mount_offset_m=fit.x[-3:].tolist(),cost=float(fit.cost),
        distances_m={m.geom(g).name:float(mujoco.mj_geomDistance(m,d,g,handle,.1,None)) for g in geoms}))
    render(e,OUT/'audit/grip_fitted',[108],policy=True);e.stop()

if __name__=='__main__':main()
