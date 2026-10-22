"""Retarget the v2 recommendation, once per full source, to native Trajectory NPZ.
Run with the project .venv.
"""
import os
for k,v in {'PYTORCH_JIT':'0','JAX_PLATFORMS':'cpu','MUJOCO_GL':'egl','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}.items():
    os.environ[k]=v
import sys, csv, json, time, hashlib, shutil, argparse, contextlib, io, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from musclemimic.data.preprocessing.dataset_paths import OUTPUT_ROOT, ROOT, DATASET, SCRIPTS, CONFIG
sys.path.insert(0,str(ROOT))
AUDIT=DATASET/'dataset_audit/six_task_selection_v2'
OUT=OUTPUT_ROOT/'selected_myofullbody_local/gmr'

def init_worker(fitted_shape_path=None):
    global np, torch, mujoco, Rotation, Slerp, body, ik, model, initial, names, site_ids, shape_fit
    import numpy as np
    import torch
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp
    from loco_mujoco.smpl.retargeting import _ensure_gmr_imports, get_smpl_model_path
    _ensure_gmr_imports()
    from general_motion_retargeting import GeneralMotionRetargeting, ROBOT_XML_DICT, IK_CONFIG_DICT
    from general_motion_retargeting.utils.smpl import SMPLH_Parser
    from smplx.joint_names import JOINT_NAMES
    import general_motion_retargeting.motion_retarget as module
    from unittest.mock import patch
    torch.set_num_threads(1)
    model=mujoco.MjModel.from_binary_path(str(OUT/'model.mjb'))
    assert (model.nq,model.nv)==(89,88)
    class ModelFactory:
        @staticmethod
        def from_xml_path(path): return model
    class Adapter:
        MjModel=ModelFactory
        def __getattr__(self,key): return getattr(mujoco,key)
    ROBOT_XML_DICT['myofullbody']=OUT/'model.mjb'
    IK_CONFIG_DICT.setdefault('smplh',{})['myofullbody']=CONFIG
    with patch.object(module,'mj',Adapter()),contextlib.redirect_stdout(io.StringIO()):
        ik=GeneralMotionRetargeting(src_human='smplh',tgt_robot='myofullbody',actual_human_height=1.6,verbose=False,use_fitted_shape=fitted_shape_path is not None,fitted_shape_path=str(fitted_shape_path) if fitted_shape_path else None)
    shape_fit=None
    if fitted_shape_path:
        from general_motion_retargeting.utils.shape_fitting import load_fitted_shape
        shape_fit=load_fitted_shape(str(fitted_shape_path))
        assert ik.learned_offsets is not None, 'Fitted alignment offsets were not loaded'
    initial=model.qpos0.copy()
    for j in range(model.njnt):
        if model.jnt_limited[j] and model.jnt_type[j] in (2,3):
            a=model.jnt_qposadr[j]; initial[a]=np.clip(initial[a],*model.jnt_range[j])
    body=SMPLH_Parser(model_path=get_smpl_model_path(),gender='neutral',use_pca=False,num_betas=16).eval()
    names=JOINT_NAMES[:22]
    site_ids=np.array([i for i in range(model.nsite) if model.site(i).name.endswith('_mimic')],dtype=int)
    assert len(site_ids)==17

def convert(path, source_file=None, frequency=30., source_start_frame=0, offset_to_ground=True, correct_floor_penetration=True):
    started=time.time()
    try:
        from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryData, TrajectoryModel
        from loco_mujoco.smpl.retargeting import max_penetration_with_floor
        with np.load(source_file or DATASET/path,allow_pickle=False) as z:
            poses=z['poses'][:,:66].copy(); trans=z['trans'].copy(); beta=z['betas'].copy(); fps=float(z['mocap_framerate'])
        if shape_fit is not None:
            fitted_beta,fitted_scale,offset_z,height_scale,_=shape_fit
            beta=np.asarray(fitted_beta).reshape(-1)
            trans[:,2]+=offset_z
            trans[:,:2]*=height_scale
            trans[:,2]=(trans[:,2]-trans[0,2])*height_scale+trans[0,2]
        n=len(poses); source_times=np.arange(n)/fps
        times=np.arange(int(np.floor(source_times[-1]*frequency+1e-8))+1)/frequency
        sampled=np.zeros((len(times),156),dtype=np.float32)
        for j in range(22):
            sampled[:,3*j:3*j+3]=Slerp(source_times,Rotation.from_rotvec(poses[:,3*j:3*j+3]))(times).as_rotvec()
        translations=np.stack([np.interp(times,source_times,trans[:,k]) for k in range(3)],axis=-1)
        translations[:,:2]-=translations[0,:2]
        positions=[]
        with torch.no_grad():
            for start in range(0,len(times),64):
                p=sampled[start:start+64]; b=np.repeat(beta[None],len(p),axis=0)
                verts,joints=body.get_joints_verts(torch.tensor(p),torch.tensor(b,dtype=torch.float32),torch.tensor(translations[start:start+64],dtype=torch.float32))
                positions.append(joints[:,:22].cpu().numpy().copy())
                del verts,joints
        positions=np.concatenate(positions)
        if shape_fit is not None:
            root=positions[:,0:1].copy()
            positions=(positions-root)*float(np.asarray(fitted_scale).reshape(-1)[0])+root
        rots=[]
        for j in range(22):
            r=Rotation.from_rotvec(sampled[:,3*j:3*j+3]); parent=int(body.parents[j])
            rots.append(r if parent<0 else rots[parent]*r)
        quats=np.stack([r.as_quat(scalar_first=True) for r in rots],axis=1)
        ik.configuration.update(q=initial.copy())
        qrows=[]; errors=[]
        for i in range(len(times)):
            frame={names[j]:(positions[i,j],quats[i,j]) for j in range(22)}
            if i == 0:
                # Align once per complete source, using the fitted IK target frame.
                # Later frames keep the previous solution; cropping does not reset IK.
                ik.update_targets(frame,offset_to_ground=offset_to_ground)
                root_position,root_quaternion=ik.scaled_human_data['pelvis']
                aligned_initial=initial.copy()
                aligned_initial[:3]=root_position
                aligned_initial[3:7]=root_quaternion
                assert np.isfinite(aligned_initial).all()
                assert np.isclose(np.linalg.norm(aligned_initial[3:7]),1.0)
                ik.configuration.update(q=aligned_initial)
            q,d=ik.retarget(frame,offset_to_ground=offset_to_ground)
            qrows.append(q.copy());errors.append(d.copy())
        qpos=np.asarray(qrows); errors=np.asarray(errors)
        data=mujoco.MjData(model); corrections=[]
        for q in qpos:
            data.qpos[:]=q; mujoco.mj_forward(model,data)
            correction=0.
            if correct_floor_penetration:
                pen,_,_=max_penetration_with_floor(model,data)
                correction=max(0.,-float(pen))
                q[2]+=correction
            corrections.append(correction)
        qvel=np.empty((len(times),model.nv)); differences=np.empty((len(times)-1,model.nv))
        for i in range(len(times)-1): mujoco.mj_differentiatePos(model,differences[i],1/frequency,qpos[i],qpos[i+1])
        qvel[0]=differences[0];qvel[-1]=differences[-1];qvel[1:-1]=.5*(differences[:-1]+differences[1:])
        xpos=np.empty((len(times),len(site_ids),3),dtype=np.float32); xmat=np.empty((len(times),len(site_ids),9),dtype=np.float32)
        for i,q in enumerate(qpos):
            data.qpos[:]=q; mujoco.mj_kinematics(model,data)
            xpos[i]=data.site_xpos[site_ids];xmat[i]=data.site_xmat[site_ids]
        assert np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(errors).all()
        tm=TrajectoryModel(model.njnt,np.array(model.jnt_type),nsite=len(site_ids),site_bodyid=np.array(model.site_bodyid[site_ids]),site_pos=np.array(model.site_pos[site_ids]),site_quat=np.array(model.site_quat[site_ids]))
        info=TrajectoryInfo([model.joint(i).name for i in range(model.njnt)],tm,frequency,site_names=[model.site(i).name for i in site_ids],metadata={'offset_to_ground':offset_to_ground,'correct_floor_penetration':correct_floor_penetration,'target':'myofullbody_no_fingers','retarget_method':'gmr','root_initialization':'first_frame_fitted_pelvis_target','model_file':'model.mjb','hand_pose':'zeroed; wrists retained','source_shape':('fitted target betas, scale and learned alignment offsets; fitted_shape.pkl' if shape_fit is not None else 'source betas, neutral SMPLH, no fitted shape')})
        trajectory=Trajectory(info,TrajectoryData(qpos=qpos,qvel=qvel,site_xpos=xpos,site_xmat=xmat,split_points=np.array([0,len(times)])))
        destination=OUT/path;destination.parent.mkdir(parents=True,exist_ok=True)
        trajectory.save(str(destination))
        loaded=Trajectory.load(str(destination),backend=np)
        assert loaded.data.qpos.shape==(len(times),89) and loaded.data.qvel.shape==(len(times),88)
        assert np.array_equal(loaded.data.qpos,qpos)
        hinge=[]; slide=[]
        for j in range(model.njnt):
            if model.jnt_limited[j] and model.jnt_type[j] in (2,3):
                values=qpos[:,model.jnt_qposadr[j]]; lo,hi=model.jnt_range[j]
                violation=float(np.maximum(0,np.maximum(lo-values,values-hi)).max())
                (hinge if model.jnt_type[j]==3 else slide).append(violation)
        result={'path':path,'status':'converted','frames':len(times),'source_frames':n,'source_fps':fps,'frequency':frequency,'position_mean_m':float(errors.mean()),'position_p95_m':float(np.percentile(errors,95)),'max_hinge_limit_violation_rad':max(hinge,default=0),'max_slide_limit_violation_m':max(slide,default=0),'max_floor_correction_m':max(corrections),'max_abs_qvel':float(np.abs(qvel).max()),'seconds':round(time.time()-started,2)}
        np.savez_compressed(destination.with_name(destination.stem+'_analysis.npz'),source_times_s=times+source_start_frame/fps,source_frame_coordinates=times*fps+source_start_frame,position_errors_m=errors,floor_correction_m=corrections)
        destination.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    except Exception:
        return {'path':path,'status':'error','error':traceback.format_exc(),'seconds':round(time.time()-started,2)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int);p.add_argument('--resume',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(AUDIT/'myofullbody_no_fingers_audit.mjb',OUT/'model.mjb')
    shutil.copy2(AUDIT/'retarget_model.json',OUT/'model_metadata.json')
    rows=list(csv.DictReader(open(AUDIT/'recommended.csv')))
    paths=sorted({r['path'] for r in rows})
    if args.limit: paths=paths[:args.limit]
    results=[]
    if args.resume:
        pending=[]
        for path in paths:
            sidecar=(OUT/path).with_suffix('.json')
            if sidecar.exists() and (OUT/path).exists(): results.append(json.loads(sidecar.read_text()))
            else:pending.append(path)
        paths=pending
    manifest={'python':sys.executable,'selected_records':len(rows),'unique_sources':len({r['path'] for r in rows}),'fps':30,'selection_sha256':hashlib.sha256((AUDIT/'recommended.csv').read_bytes()).hexdigest(),'scope':'full sources; task intervals retained in selection_index.csv','dynamics_validated':False,'model_sha256':hashlib.sha256((OUT/'model.mjb').read_bytes()).hexdigest()}
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(AUDIT/'recommended.csv',OUT/'selection_index.csv')
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        futures={pool.submit(convert,path):path for path in paths}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            (OUT/'conversion_results.json').write_text(json.dumps(results,indent=2)+'\n')
            print(len(results),manifest['unique_sources'],json.dumps(result,ensure_ascii=False),flush=True)
    failures=[r for r in results if r['status']!='converted']
    print('FINISHED',len(results),'failures',len(failures),flush=True)
    if failures:sys.exit(1)
if __name__=='__main__':main()
