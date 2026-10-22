"""Retarget unique approved sources continuously, then crop all approved tasks.

100 Hz; project-default IK constraints; fitted shape; no ground correction.
No dependency on previous outputs or .cache. Supports restart after interruption.
"""
from musclemimic.data.preprocessing import retarget_selected_myofullbody as engine
from musclemimic.data.preprocessing.dataset_paths import OUTPUT_ROOT, DATASET, SCRIPTS, CONFIG
import argparse,json,hashlib,shutil,collections,zipfile,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=engine.ROOT
ASSETS=['model.mjb','joint_schema.json','fitted_shape.pkl','fitted_shape_metadata.json','shape_fit_report.json']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(folder,name,value):(folder/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def resume_fingerprint_matches(saved, current):
    if saved == current:
        return True
    migration_path = Path(__file__).resolve().with_name('relocation_fingerprints.json')
    if not migration_path.exists():
        return False
    migration = json.loads(migration_path.read_text())
    revisions = [migration, *migration.get('compatible_revisions', [])]
    if not any(saved.get('implementation_sha256') == revision['before'] and
               current['implementation_sha256'] == revision['after'] for revision in revisions):
        return False
    return {**saved, 'implementation_sha256': current['implementation_sha256']} == current

def load_selection(path,limit=None):
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if limit:rows=rows[:limit]
    assert rows and len({r['id'] for r in rows})==len(rows),'Empty selection or duplicate IDs'
    sources={};clean=[]
    for raw in rows:
        r=dict(raw);source=Path(r['source_path'])
        assert not source.is_absolute() and '..' not in source.parts
        assert r['review_status'] in ('pass','human_pass'),r['id']
        assert all('/' not in r[k] and '\\' not in r[k] and r[k] not in ('.','..') for k in ['id','task'])
        p=DATASET/source
        if str(source) not in sources:
            digest=sha(p)
            with np.load(p,allow_pickle=False) as z:
                n=len(z['poses']);fps=float(z['mocap_framerate']);gender=str(z['gender'].item())
                assert z['poses'].shape[1]>=66 and z['trans'].shape==(n,3) and n>1 and fps>0
            sources[str(source)]=dict(id='__'.join(source.with_suffix('').parts),task='full_source',subtype='完整源文件（包含多个动作）',source_path=str(source),trajectory_path=str(source),source_start_frame=0,source_end_frame_exclusive=n,clip_frames=n,fps=fps,seconds=(n-1)/fps,source_file_sha256=digest,source_gender=gender,review_status='not_human_reviewed',annotation_ids=[])
        f=sources[str(source)]
        assert r['source_file_sha256']==f['source_file_sha256'],f'Source changed: {source}'
        assert r['fps']==f['fps'] and 0<=r['source_start_frame']<r['source_end_frame_exclusive']<=f['clip_frames']
        assert r['clip_frames']==r['source_end_frame_exclusive']-r['source_start_frame']
        f['annotation_ids'].append(r['id'])
        r.pop('clip_path',None);r.pop('clip_file_sha256',None)
        r.update(trajectory_path=f"{r['task']}/{r['id']}.npz",full_trajectory_path=str(source),source_gender=f['source_gender'],shape_fit_file='fitted_shape.pkl',offset_to_ground=False,correct_floor_penetration=False,retarget_revision='full_source_continuous_then_crop_100hz')
        a=int(np.ceil(r['source_start_frame']/r['fps']*100-1e-8));b=int(np.floor((r['source_end_frame_exclusive']-1)/r['fps']*100+1e-8))+1
        assert b-a>=2,r['id']
        clean.append(r)
    return clean,list(sources.values())
def prepare_assets(folder,label):
    import mujoco
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody
    marker=folder/'assets_manifest.json'
    if marker.exists():
        for name,digest in json.loads(marker.read_text()).items():assert sha(folder/name)==digest,f'Asset changed: {name}'
        return
    import myo_sim
    spec=myo_sim.load_spec('myofullbody')
    env=MyoFullBody.__new__(MyoFullBody);env._disable_fingers=True;spec=env._apply_spec_changes(spec)
    for joint in list(spec.joints):
        if joint.name in ('cmc_flexion','cmc_abduction','md3_flexion','pm4_flexion'):spec.delete(joint)
    for actuator in list(spec.actuators):
        if actuator.name in {x+'_left' for x in env._FINGER_MUSCLE_STEMS}:spec.delete(actuator)
    for tendon in list(spec.tendons):
        if tendon.name in {x+'_tendon_left' for x in env._FINGER_MUSCLE_STEMS}:spec.delete(tendon)
    model=spec.compile();assert (model.nq,model.nv,model.nu)==(89,88,354)
    mujoco.mj_saveModel(model,str(folder/'model.mjb'))
    config=json.loads(CONFIG.read_text());joints=[]
    for j in range(model.njnt):
        kind=int(model.jnt_type[j]);qp=int(model.jnt_qposadr[j]);qv=int(model.jnt_dofadr[j])
        joints.append(dict(name=model.joint(j).name,type={0:'free',2:'slide',3:'hinge'}[kind],qpos_indices=list(range(qp,qp+(7 if kind==0 else 1))),qvel_indices=list(range(qv,qv+(6 if kind==0 else 1))),limited=bool(model.jnt_limited[j]),range=model.jnt_range[j].tolist()))
    dump(folder,'joint_schema.json',dict(nq=89,nv=88,nu=354,joints=joints,site_names=[model.site(i).name for i in range(model.nsite) if model.site(i).name.endswith('_mimic')],ik_error_target_names=[k for k,v in config['ik_match_table2'].items() if v[1]!=0 or v[2]!=0],root_quaternion_order='wxyz'))
    import fit_review_target_shape as fit
    fit.BASE=folder;fit.OUT=folder;fit.main()
    dump(folder,'assets_manifest.json',{name:sha(folder/name) for name in ASSETS})
def init(folder):
    engine.OUT=Path(folder);engine.init_worker(engine.OUT/'fitted_shape.pkl')
def job(row):return engine.convert(row['trajectory_path'],DATASET/row['source_path'],100.,0,False,False)
def complete(full,row):
    p=full/row['trajectory_path'];side=p.with_suffix('.json');analysis=p.with_name(p.stem+'_analysis.npz')
    if not all(x.exists() for x in [p,side,analysis]):return False
    try:
        result=json.loads(side.read_text());expected=int(np.floor((row['clip_frames']-1)/row['fps']*100+1e-8))+1
        if result['status']!='converted' or result['source_frames']!=row['clip_frames']:return False
        with np.load(p,allow_pickle=True) as z:
            meta=z['metadata'].item()
            if z['qpos'].shape!=(expected,89) or z['qvel'].shape!=(expected,88) or float(z['frequency'])!=100 or not np.isfinite(z['qpos']).all() or not np.isfinite(z['qvel']).all():return False
            if z['site_xpos'].shape!=(expected,17,3) or z['site_xmat'].shape!=(expected,17,9):return False
            if meta['offset_to_ground'] or meta['correct_floor_penetration']:return False
        with np.load(analysis) as z:
            if len(z['source_times_s'])!=expected or not np.isfinite(z['position_errors_m']).all():return False
        return True
    except (ValueError,KeyError,OSError,EOFError,zipfile.BadZipFile):return False

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--annotations',type=Path,default=DATASET/'dataset_audit/approved_annotations.jsonl')
    p.add_argument('--output',type=Path,default=OUTPUT_ROOT/'approved_continuous_100hz_local')
    p.add_argument('--model',choices=['new'],default='new',help='myo_sim target; the new label preserves existing output paths')
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--resume',action='store_true',help='Reuse verified complete source conversions with identical run settings')
    p.add_argument('--dry-run',action='store_true',help='Validate annotations/sources and print counts; write no outputs')
    p.add_argument('--visualize',action='store_true',help='Build standalone review HTML')
    p.add_argument('--limit',type=int,help='Use first N annotations, for a small trial')
    args=p.parse_args()
    if args.workers<1 or (args.limit is not None and args.limit<1):p.error('workers and limit must be positive')
    rows,sources=load_selection(args.annotations.resolve(),args.limit)
    labels=[args.model]
    cfg=json.loads(CONFIG.read_text())
    left_costs={t:{b:cfg[t][b][1] for b in ['ulna_l','lunate_l']} for t in ['ik_match_table1','ik_match_table2']}
    summary=dict(annotations=len(rows),unique_source_files=len(sources),task_counts=dict(collections.Counter(r['task'] for r in rows)),full_source_seconds=sum(r['seconds'] for r in sources),models=labels,frequency=100,root_initialization="first_frame_fitted_pelvis_target",shape_fitting=True,left_arm_position_costs=left_costs,offset_to_ground=False,correct_floor_penetration=False)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if args.dry_run:return
    out=args.output.resolve()
    fingerprint=dict(model=args.model,annotations_sha256=sha(args.annotations.resolve()),limit=args.limit,frequency=100,root_initialization="first_frame_fitted_pelvis_target",config_sha256=sha(CONFIG),implementation_sha256={name:sha(SCRIPTS/name) for name in ['retarget_approved_continuous.py','retarget_selected_myofullbody.py','retarget_continuous_25.py','fit_review_target_shape.py','retarget_approved_clips.py','dataset_paths.py']})
    if out.exists() and any(out.iterdir()):
        if not args.resume:p.error('Output directory is not empty; use --resume or a new --output')
        if not (out/'run_fingerprint.json').exists() or not resume_fingerprint_matches(json.loads((out/'run_fingerprint.json').read_text()),fingerprint):p.error('Resume settings/input/code differ from the saved run; use a new --output')
    out.mkdir(parents=True,exist_ok=True);dump(out,'run_fingerprint.json',fingerprint);dump(out,'selection_summary.json',summary)
    (out/'selection.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    shutil.copy2(CONFIG,out/'ik_config.json')
    for label in labels:
        full=out/'full'/label;clips=out/'clips'/label
        for folder in [full,clips]:folder.mkdir(parents=True,exist_ok=True)
        prepare_assets(full,label)
        for name in ASSETS:shutil.copy2(full/name,clips/name)
        for folder in [full,clips]:dump(folder,'run_manifest.json',dict(summary,target_model=label,scope='Full source from first frame; reset IK only between source files; crop annotated intervals afterwards',velocity='Cropped velocities recomputed at 100Hz',shape_fit_file='fitted_shape.pkl'))
        pending=[r for r in sources if not complete(full,r)];done=len(sources)-len(pending)
        print(label,'reuse',done,'pending',len(pending),flush=True)
        pending.sort(key=lambda r:r['seconds'],reverse=True);failures=[]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(str(full),)) as pool:
            for future in as_completed([pool.submit(job,r) for r in pending]):
                result=future.result();done+=1
                print(label,done,len(sources),json.dumps(result),flush=True)
                if result['status']!='converted':failures.append(result)
        if failures:
            dump(full,'failures.json',failures);raise RuntimeError('Some sources failed; fix the cause and rerun with --resume')
        from musclemimic.data.preprocessing import retarget_approved_clips as verify
        verify.STAGE=full;verify.verify(sources,100)
        from musclemimic.data.preprocessing.retarget_continuous_25 import crop
        cropped=[crop(r,full,clips) for r in rows]
        verify.STAGE=clips;verify.verify(cropped,100)
        (full/'README.md').write_text(f'{len(sources)} unique source files retargeted continuously from frame zero at 100 Hz. No task labels or human approval apply to the full source. Annotation IDs link each source to its selected task clips.\n')
        (clips/'README.md').write_text(f'{len(rows)} approved task intervals cropped AFTER full-source retargeting. No IK reset at crop boundaries.\nGlobal 100Hz samples: first sample at/after annotation start, last sample at/before last annotated source frame. Crop start may shift by <10ms.\nRoot xy recentered once; all other qpos unchanged. qvel recomputed using MuJoCo differences, with one-sided endpoints.\nIK constraints recorded in ik_config.json and run_manifest.json; fitted shape; no ground offset or floor penetration correction.\nPosition quality is not a dynamics validation. See tasks.jsonl, task_index_100hz.csv and needs_review.csv.\n')
        if args.visualize:
            import build_retarget_review_player as viewer
            viewer.DATA=clips;viewer.OUT=clips/'viewer';viewer.main()
    rep=out/'reproduction';rep.mkdir(exist_ok=True)
    for name in fingerprint['implementation_sha256']:shutil.copy2(SCRIPTS/name,rep/name)
    for name in ['build_retarget_review_player.py','retarget_review_player.html','ik_config.json']:shutil.copy2(SCRIPTS/name,rep/name)
    (out/'README.md').write_text(f'{len(rows)} selected task intervals from {len(sources)} unique full sources.\nfull/<model>/<dataset>/<subject>: continuously retargeted source files.\nclips/<model>/<task>: extracted task trajectories and error diagnostics.\n100Hz, IK configuration saved in ik_config.json, target shape fitting, no ground correction. Source annotations are retained in selection.jsonl.\nRe-run the same command with --resume after interruption.\n')
    for folder in [*(out/'full'/label for label in labels),*(out/'clips'/label for label in labels),out]:dump(folder,'delivery_manifest.json',{str(f.relative_to(folder)):sha(f) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='delivery_manifest.json'})
    print('DONE',out,flush=True)
if __name__=='__main__':main()
