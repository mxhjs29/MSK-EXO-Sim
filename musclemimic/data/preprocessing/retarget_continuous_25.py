"""Retarget complete sources from frame zero, then extract the same 25 tasks."""
from musclemimic.data.preprocessing import retarget_selected_myofullbody as engine
from musclemimic.data.preprocessing.dataset_paths import OUTPUT_ROOT, DATASET, SCRIPTS, CONFIG
from pathlib import Path
import numpy as np,json,shutil,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
ROOT=engine.ROOT;CACHE=ROOT/'.cache/continuous_25_inputs';OUT=OUTPUT_ROOT/'continuous_25_100hz'
ASSETS=['model.mjb','joint_schema.json','fitted_shape.pkl','fitted_shape_metadata.json','shape_fit_report.json']
def dump(folder,name,x):
    (folder/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def init(folder):
    engine.OUT=Path(folder);engine.init_worker(engine.OUT/'fitted_shape.pkl')
def job(r):
    return engine.convert(r['trajectory_path'],DATASET/r['source_path'],100.,0,False,False)
def crop(r,full,clips):
    import mujoco
    from loco_mujoco.trajectory import Trajectory
    m=mujoco.MjModel.from_binary_path(str(clips/'model.mjb'));d=mujoco.MjData(m)
    src=full/r.get('full_trajectory_path',r['trajectory_path']);dst=clips/r['trajectory_path'];dst.parent.mkdir(parents=True,exist_ok=True)
    a=int(np.ceil(r['source_start_frame']/r['fps']*100-1e-8));b=int(np.floor((r['source_end_frame_exclusive']-1)/r['fps']*100+1e-8))+1
    with np.load(src,allow_pickle=True) as z:payload={k:z[k] for k in z.files}
    original=payload['qpos'][a:b].copy();q=original.copy();shift=q[0,:2].copy();q[:,:2]-=shift
    n=len(q);assert n>1
    dv=np.empty((n-1,88))
    for i in range(n-1):mujoco.mj_differentiatePos(m,dv[i],.01,q[i],q[i+1])
    v=np.vstack([dv[0],(dv[:-1]+dv[1:])/2,dv[-1]])
    ids=[m.site(str(name)).id for name in payload['site_names']]
    xpos=[];xmat=[]
    for qp in q:
        d.qpos[:]=qp;mujoco.mj_kinematics(m,d);xpos.append(d.site_xpos[ids].copy());xmat.append(d.site_xmat[ids].copy())
    payload.update(qpos=q,qvel=v,site_xpos=np.asarray(xpos,dtype=np.float32),site_xmat=np.asarray(xmat,dtype=np.float32),split_points=np.array([0,n]))
    metadata=payload['metadata'].item();metadata.update(retarget_scope='full source from frame zero; cropped afterwards; no IK reset at crop',full_output_frame_interval=[a,b],crop_xy_translation_removed=shift.tolist(),velocity='Recomputed from cropped qpos at 100Hz; one-sided endpoints')
    payload['metadata']=metadata;np.savez_compressed(dst,**payload)
    loaded=Trajectory.load(str(dst),backend=np);assert np.array_equal(loaded.data.qpos,q)
    assert np.array_equal(q[:,2:],original[:,2:]) and np.allclose(q[:,:2]+shift,original[:,:2],atol=1e-12)
    with np.load(src.with_name(src.stem+'_analysis.npz')) as z:analysis={k:z[k][a:b] for k in z.files}
    assert np.allclose(analysis['source_times_s'],np.arange(a,b)/100)
    np.savez_compressed(dst.with_name(dst.stem+'_analysis.npz'),**analysis)
    errors=analysis['position_errors_m'];hinge=[];slide=[]
    for j in range(m.njnt):
        if m.jnt_limited[j] and m.jnt_type[j] in (2,3):
            values=q[:,m.jnt_qposadr[j]];lo,hi=m.jnt_range[j]
            (hinge if m.jnt_type[j]==3 else slide).append(float(np.maximum(0,np.maximum(lo-values,values-hi)).max()))
    result=dict(path=r['trajectory_path'],status='converted',frames=n,source_frames=r['clip_frames'],source_fps=r['fps'],frequency=100.,position_mean_m=float(errors.mean()),position_p95_m=float(np.percentile(errors,95)),max_hinge_limit_violation_rad=max(hinge,default=0),max_slide_limit_violation_m=max(slide,default=0),max_floor_correction_m=float(analysis['floor_correction_m'].max()),max_abs_qvel=float(np.abs(v).max()),seconds=0.,full_output_frame_start=a,full_output_frame_end_exclusive=b)
    dst.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    return dict(r,sample_start_time_s=a/100,sample_count=n,full_output_frame_start=a,full_output_frame_end_exclusive=b,xy_translation_removed=shift.tolist())
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name,digest in json.loads((CACHE/'input_manifest.json').read_text()).items():assert hashlib.sha256((CACHE/name).read_bytes()).hexdigest()==digest
    assert (CACHE/'ik_config.json').read_bytes()==(CONFIG).read_bytes()
    rows=[json.loads(s) for s in (CACHE/'selection.jsonl').read_text().splitlines()];assert len(rows)==25
    fullrows=[]
    for r in rows:
        p=DATASET/r['source_path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['source_file_sha256']
        with np.load(p,allow_pickle=False) as z:n=len(z['poses']);gender=str(z['gender'].item())
        r.update(source_gender=gender,offset_to_ground=False,correct_floor_penetration=False,retarget_revision='continuous_full_source_then_crop_100hz',full_source_frames=n)
        f=dict(r,source_start_frame=0,source_end_frame_exclusive=n,clip_frames=n,source_start_t=0,source_end_t=n/r['fps'],seconds=(n-1)/r['fps'],review_status='not_human_reviewed');fullrows.append(f)
    (OUT/'selection.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    for label in ['new']:
        full=OUT/'full'/label;clips=OUT/'clips'/label
        run=dict(target_model=label,frequency=100,offset_to_ground=False,correct_floor_penetration=False,shape_fitting=True,left_elbow_position_weight=40,left_wrist_position_weight=40,scope='Full source from first frame, IK initialized once per source. Crop tasks afterwards on global 100Hz grid.',cropping='First sample >= task start, last sample <= last annotated source frame; no interpolation of retargeted qpos.',source_gender_used_in_visualization=True)
        for folder in [full,clips]:
            folder.mkdir(parents=True,exist_ok=True)
            for name in ASSETS:shutil.copy2(CACHE/label/name,folder/name)
            dump(folder,'run_manifest.json',run)
        pending=[];results=[]
        for r in fullrows:
            p=full/r['trajectory_path']
            if p.exists() and p.with_suffix('.json').exists():
                result=json.loads(p.with_suffix('.json').read_text());assert result['source_frames']==r['clip_frames'] and result['frequency']==100;results.append(result)
            else:pending.append(r)
        # Long sources first to keep all workers occupied near the end.
        pending.sort(key=lambda r:r['seconds'],reverse=True)
        with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=init,initargs=(str(full),)) as pool:
            for f in as_completed([pool.submit(job,r) for r in pending]):
                result=f.result();results.append(result);dump(full,'conversion_progress.json',results)
                print(label,len(results),25,json.dumps(result),flush=True)
        assert all(r['status']=='converted' for r in results),results
        from musclemimic.data.preprocessing import retarget_approved_clips as verify
        verify.STAGE=full;verify.verify(fullrows,100)
        cropped=[crop(r,full,clips) for r in rows]
        verify.STAGE=clips;verify.verify(cropped,100)
        import build_retarget_review_player as viewer
        viewer.DATA=clips;viewer.OUT=clips/'viewer';viewer.main()
        (full/'README.md').write_text('25个完整源文件从首帧连续重映射，100Hz。中途不重置IK，不进行任务区间筛选。tasks.jsonl中的原人工标签只描述对应任务片段，不代表整条源文件。\n完整源文件并未经过全段人工动作验收。needs_review.csv统计完整文件误差，任务片段质量见../../clips/'+label+'/。\n')
        (clips/'README.md').write_text('完整源文件连续重映射后裁剪原25个任务区间；100Hz。\n保留全局100Hz网格：首采样时间不早于任务起点，末采样不超过标注区间最后一个原始帧，因此相较此前片段起点网格，最多偏移不足10ms。\n裁剪时不重新求解IK、不插值qpos，仅将根xy起点平移至零并重新计算片段速度和跟踪点。shape_fit、左臂约束40、无地面修正。\n位置误差阈值不等于动力学或连续性验收。\n')
    (OUT/'README.md').write_text('原25条任务的完整源文件使用 myo_sim 连续重映射，再裁剪原标注区间。\nfull/new：完整源轨迹；clips/new：按任务类别保存的25个片段。\n100Hz，左肘/腕位置权重40，体型拟合，关闭贴地与地面穿透修正。原始视窗采用源文件性别和betas。\n')
    rep=OUT/'reproduction';rep.mkdir(exist_ok=True)
    for name in ['retarget_continuous_25.py','retarget_selected_myofullbody.py','retarget_approved_clips.py','build_retarget_review_player.py','retarget_review_player.html','dataset_paths.py']:shutil.copy2(SCRIPTS/name,rep/name)
    shutil.copy2(CACHE/'ik_config.json',rep/'ik_config.json')
    for folder in [OUT/'full/new',OUT/'clips/new',OUT]:
        dump(folder,'delivery_manifest.json',{str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'})
    print('DONE',OUT,flush=True)
if __name__=='__main__':main()
