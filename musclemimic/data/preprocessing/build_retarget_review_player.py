"""Build a self-contained synchronized source/retarget skeleton review player."""
import os
for k,v in {'PYTORCH_JIT':'0','JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}.items(): os.environ[k]=v
import base64,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from musclemimic.data.preprocessing.dataset_paths import OUTPUT_ROOT, ROOT, DATASET, SCRIPTS, CONFIG
sys.path.insert(0,str(ROOT))
DATA=DATASET/'converted_dataset/approved_continuous_100hz/clips/new'
OUT=OUTPUT_ROOT/'retarget_review_100hz'

def pack(a):
    return base64.b64encode(np.asarray(a,dtype='<f4').tobytes()).decode()

def main():
    import mujoco,torch
    from loco_mujoco.smpl.retargeting import _ensure_gmr_imports,get_smpl_model_path
    _ensure_gmr_imports()
    from general_motion_retargeting.utils.smpl import SMPLH_Parser
    torch.set_num_threads(1)
    body=SMPLH_Parser(model_path=get_smpl_model_path(),gender='neutral',use_pca=False,num_betas=16).eval()
    model=mujoco.MjModel.from_binary_path(str(DATA/'model.mjb')); d=mujoco.MjData(model)
    # GMR body origins, in anatomical order, derived directly from saved qpos.
    target_names=['pelvis','femur_l','femur_r','lumbar1','tibia_l','tibia_r','calcn_l','calcn_r','toes_l','toes_r','head','humerus_l','humerus_r','ulna_l','ulna_r','lunate_l','lunate_r']
    bids=[model.body(name).id for name in target_names]
    edges=[[0,1],[0,2],[0,3],[1,4],[2,5],[4,6],[5,7],[6,8],[7,9],[3,10],[3,11],[3,12],[11,13],[12,14],[13,15],[14,16]]
    parents=body.parents[:22].cpu().numpy().astype(int)
    review_ids={r['id'] for r in csv.DictReader((DATA/'needs_review.csv').open())}
    tasks={r['id']:r for r in map(json.loads,(DATA/'tasks.jsonl').read_text().splitlines())}
    schema=json.loads((DATA/'joint_schema.json').read_text())
    rows=[];checks=[];rests={};bodies={'neutral':body}
    manifest=json.loads((DATA/'run_manifest.json').read_text()) if (DATA/'run_manifest.json').exists() else {}
    for r in tasks.values():
        gender=r.get('source_gender','neutral')
        if gender not in bodies:
            model_dir=Path(get_smpl_model_path())
            release=model_dir/'smplh'/gender/'model.npz'
            if release.exists():
                from smplx.utils import Struct
                with np.load(release,allow_pickle=True) as model_data:fields={k:model_data[k] for k in model_data.files}
                # AMASS release lacks hand PCA data; unused for the 22 body joints.
                for side in ('l','r'):
                    fields.setdefault('hands_components'+side,np.zeros((45,45),dtype=np.float32))
                    fields.setdefault('hands_mean'+side,np.zeros(45,dtype=np.float32))
                bodies[gender]=SMPLH_Parser(model_path=str(release),data_struct=Struct(**fields),gender=gender,use_pca=False,flat_hand_mean=True,num_betas=16).eval()
            else:
                # Local MANO release stores gender-specific SMPL-H files here.
                local_model=model_dir/'mano_v1_2'/'models'/f'SMPLH_{gender}.pkl'
                standard_model=model_dir/f'SMPLH_{gender.upper()}.pkl'
                model_path=standard_model if standard_model.exists() else (local_model if local_model.exists() else model_dir)
                bodies[gender]=SMPLH_Parser(model_path=str(model_path),gender=gender,use_pca=False,num_betas=16).eval()
        body=bodies[gender]
        path=DATA/r['trajectory_path']
        quality=r['quality']['quality_flag']
        assert quality in ('within_position_thresholds','review_position_error')
        assert (r['id'] in review_ids)==(quality=='review_position_error')
        with np.load(path,allow_pickle=True) as z:
            q=z['qpos'];freq=float(z['frequency']);site=z['site_xpos'];siteids=[model.site(str(n)).id for n in z['site_names']]
        with np.load(path.with_name(path.stem+'_analysis.npz')) as z:
            times=z['source_times_s'].copy();err=z['position_errors_m'].copy();floor=z['floor_correction_m'].copy();coords=z['source_frame_coordinates'].copy()
        n=len(q); assert freq==100 and len(times)==n and np.allclose(np.diff(times),.01)
        with np.load(DATASET/r['source_path'],allow_pickle=False) as z:
            lo,hi=r['source_start_frame'],r['source_end_frame_exclusive'];poses=z['poses'][lo:hi,:66];trans=z['trans'][lo:hi].copy();beta=z['betas'];fps=float(z['mocap_framerate'])
        local=times-lo/fps;st=np.arange(len(poses))/fps
        assert local[-1]<=st[-1]+1e-8 and np.allclose(coords,times*fps)
        local=np.clip(local,0,st[-1]);sampled=np.zeros((n,156),dtype=np.float32)
        for j in range(22):sampled[:,3*j:3*j+3]=Slerp(st,Rotation.from_rotvec(poses[:,3*j:3*j+3]))(local).as_rotvec()
        t=np.stack([np.interp(local,st,trans[:,i]) for i in range(3)],-1);t[:,:2]-=t[0,:2]
        key=(gender,beta.tobytes())
        if key not in rests:
            with torch.no_grad():_,j=body.get_joints_verts(torch.zeros((1,156)),torch.tensor(beta[None],dtype=torch.float32),torch.zeros((1,3)))
            rests[key]=j[0,:22].numpy().copy()
        rest=rests[key];rot=[];pos=[]
        for j in range(22):
            rr=Rotation.from_rotvec(sampled[:,3*j:3*j+3]);parent=parents[j]
            rot.append(rr if parent<0 else rot[parent]*rr)
            pos.append(t+rest[j] if parent<0 else pos[parent]+rot[parent].apply(np.broadcast_to(rest[j]-rest[parent],(n,3))))
        source=np.stack(pos,1)
        probe=np.unique([0,n//2,n-1])
        with torch.no_grad():_,actual=body.get_joints_verts(torch.tensor(sampled[probe]),torch.tensor(np.repeat(beta[None],len(probe),axis=0),dtype=torch.float32),torch.tensor(t[probe],dtype=torch.float32))
        fk_error=float(np.max(np.abs(actual[:,:22].numpy()-source[probe])));assert fk_error<2e-5
        target=[];site_error=0.
        for i,qp in enumerate(q):
            d.qpos[:]=qp;mujoco.mj_kinematics(model,d);target.append(d.xpos[bids].copy())
            site_error=max(site_error,float(np.max(np.abs(d.site_xpos[siteids]-site[i]))))
        assert site_error<2e-6
        target=np.asarray(target)
        # Fixed source ground estimate, never frame-dependent; target ground stays z=0.
        source_floor=float(np.min(source[:,[7,8,10,11],2]))
        shoulders=source[0,16]-source[0,17];yaw=float(np.arctan2(shoulders[1],shoulders[0]))
        row={'id':r['id'],'shape_fitted':bool(r.get('shape_fit_file')),'ground_correction':bool(r.get('offset_to_ground',manifest.get('offset_to_ground',True)) or r.get('correct_floor_penetration',manifest.get('correct_floor_penetration',True))),'revision':r.get('retarget_revision','original_100hz'),'quality':quality,'task':r['task'],'subtype':r['subtype'],'source_path':r['source_path'],'trajectory_path':r['trajectory_path'],
             'start':lo,'end':hi,'source_fps':fps,'n':n,'frequency':freq,'source_floor':source_floor,'yaw':yaw,
             'mean':float(err.mean()),'p95':float(np.percentile(err,95)),'max_floor':float(floor.max()),
             'source':pack(source),'target':pack(target),'errors':pack(err),'floor':pack(floor),'start_time':float(times[0]),
             'peak_frame':int(np.argmax(err.max(axis=1))),'labels':list(dict.fromkeys(a.get('text','') for a in r.get('babel_provenance',[])))}
        rows.append(row);checks.append({'id':r['id'],'frames':n,'source_fk_max_error_m':fk_error,'target_saved_site_max_error_m':site_error})
        print(len(rows),len(tasks),r['id'],n,flush=True)
    rows.sort(key=lambda r:r['p95'],reverse=True)
    payload={'records':rows,'source_edges':[[int(parents[j]),j] for j in range(1,22)],'target_edges':edges,'target_names':target_names,'error_names':schema['ik_error_target_names'],
             'selection_hash':hashlib.sha256((DATA/'tasks.jsonl').read_bytes()).hexdigest(),
             'legacy_selection_hash':hashlib.sha256((DATA/'needs_review.csv').read_bytes()).hexdigest()}
    assert len(rows)==len(tasks) and len({r['id'] for r in rows})==len(tasks)
    OUT.mkdir(parents=True,exist_ok=True)
    html=(SCRIPTS/'retarget_review_player.html').read_text().replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
    (OUT/'重映射复核播放器.html').write_text(html)
    (OUT/'validation.json').write_text(json.dumps({'clips':len(rows),'quality_counts':{q:sum(r['quality']==q for r in rows) for q in ('within_position_thresholds','review_position_error')},'total_frames':sum(r['n'] for r in rows),'checks':checks,'synchronization':'100 Hz source SLERP/FK at saved analysis timestamps; target FK from delivered qpos','input_sha256':{name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['needs_review.csv','tasks.jsonl','model.mjb']}},indent=2))
    (OUT/'README.md').write_text(f'''# 重映射复核播放器

直接用浏览器打开 `重映射复核播放器.html`，无需服务或联网。此单个 HTML 已内嵌全部 {len(rows)} 条片段（{sum(r["quality"]=="within_position_thresholds" for r in rows)} 条误差达标、{sum(r["quality"]=="review_position_error" for r in rows)} 条待复核）。可筛选重映射质量；质量标签与人工复核状态独立。复核记录按数据版本保存；更新重映射数据后应重新复核。
左侧为原始 SMPL-H 22 关节，右侧为实际重映射 qpos 的 MyoFullBody 17 个身体节点；均为骨架示意，不是表面网格或动力学仿真。
使用重映射结果中的原始时间坐标进行源旋转 SLERP/平移插值，按真实秒数同步，不按相对进度匹配。
右侧按实际转换结果显示，是否启用地面修正见页面说明；左侧原始高度默认不修改，可选固定地面估计对齐。
支持同尺度双视窗、旋转、缩放、正/侧视角、骨盆跟随、根轨迹、0.1–2 倍速、100 Hz 逐帧、误差曲线定位、最差帧跳转、复核标记与 JSON 导入导出。
误差来自原转换 GMR 指标（额外地面穿透修正前），不能把两侧不同解剖定义的节点距离直接当作拟合误差。
复核标记只保存在当前浏览器或导出的 JSON 中，不改变原人工筛选记录、训练数据或 needs_review.csv。

生成：`.venv/bin/python dataset/scripts/build_retarget_review_player.py`
''')
    print(OUT/'重映射复核播放器.html',flush=True)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA,help='Retargeted trajectory directory containing tasks.jsonl and model.mjb')
    parser.add_argument('--output',type=Path,help='Player output directory; defaults to <data>/viewer')
    args=parser.parse_args()
    DATA=args.data.resolve()
    OUT=args.output.resolve() if args.output else DATA/'viewer'
    main()
