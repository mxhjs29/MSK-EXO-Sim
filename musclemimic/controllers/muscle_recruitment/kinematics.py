import numpy as np
import mujoco

def independent_directions(model,data):
    rules=[]
    for k in range(model.neq):
        if model.eq_type[k]!=mujoco.mjtEq.mjEQ_JOINT or not data.eq_active[k]:continue
        child,parent=int(model.eq_obj1id[k]),int(model.eq_obj2id[k])
        if parent<0:continue
        q=data.qpos[model.jnt_qposadr[parent]]-model.qpos0[model.jnt_qposadr[parent]]
        derivative=sum(i*model.eq_data[k,i]*q**(i-1) for i in range(1,5))
        rules.append((model.jnt_dofadr[child],model.jnt_dofadr[parent],derivative))
    children={a for a,_,_ in rules}
    independent=[i for i in range(model.nv) if i not in children]
    directions=np.eye(model.nv)[:,independent]
    for _ in range(len(rules)+1):
        old=directions.copy()
        for child,parent,ratio in rules:directions[child]=ratio*directions[parent]
        if np.array_equal(old,directions):break
    return directions
