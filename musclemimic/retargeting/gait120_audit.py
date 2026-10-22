"""Audit local Gait120 sources without treating OpenSim angles as target qpos.

TRC coordinates remain in their source axes in the audit. The optional marker
example applies a proper rotation from Y-up to Z-up and converts mm to meters;
it is a marker observation file, not a training-ready muscle trajectory.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_trc(path: Path) -> dict:
    lines=path.read_text(encoding='utf-8-sig').splitlines()
    if len(lines)<6 or not lines[0].startswith('PathFileType'):
        raise ValueError(f'Invalid TRC header: {path}')
    meta=dict(zip(lines[1].split(),lines[2].split()))
    labels=[x.strip() for x in lines[3].split('\t')[2:] if x.strip()]
    count=int(meta['NumMarkers'])
    if len(labels)!=count or len(set(labels))!=count:raise ValueError('TRC marker labels do not match header')
    width=2+3*count
    rows=[]
    for line in lines[5:]:
        if not line.strip():continue
        fields=line.split('\t')
        if len(fields)<width:raise ValueError(f'Truncated TRC row: {path}')
        rows.append([float(v) if v.strip() else np.nan for v in fields[:width]])
    data=np.asarray(rows,dtype=float)
    if len(data)!=int(meta['NumFrames']):raise ValueError(f'TRC frame count mismatch: {path}')
    if len(data)<2 or not np.isfinite(data[:,:2]).all() or np.any(np.diff(data[:,1])<=0):
        raise ValueError(f'Invalid TRC timestamps: {path}')
    units=meta['Units'].lower();scale={'mm':.001,'cm':.01,'m':1.}.get(units)
    if scale is None:raise ValueError(f'Unknown TRC length unit: {units}')
    points=data[:,2:].reshape(len(data),count,3)*scale
    missing=~np.isfinite(points).all(axis=-1)|np.all(points==0,axis=-1)
    points[missing]=np.nan
    return dict(metadata=meta,labels=labels,frames=data[:,0].astype(int),time=data[:,1],points_m=points,missing=missing)


def read_mot(path: Path) -> dict:
    lines=path.read_text(encoding='utf-8-sig').splitlines()
    end=next((i for i,s in enumerate(lines) if s.strip().lower()=='endheader'),None)
    if end is None:raise ValueError(f'Missing MOT endheader: {path}')
    meta={}
    for line in lines[:end]:
        if '=' in line:
            k,v=line.split('=',1);meta[k.strip()]=v.strip()
    body=[s for s in lines[end+1:] if s.strip()]
    columns=body[0].split();data=np.loadtxt(io.StringIO('\n'.join(body[1:])),ndmin=2)
    if data.shape[1]!=len(columns) or columns[0]!='time':raise ValueError(f'MOT columns mismatch: {path}')
    if 'nColumns' in meta and int(meta['nColumns'])!=len(columns):raise ValueError(f'MOT declared column count mismatch: {path}')
    if 'nRows' in meta and int(meta['nRows'])!=len(data):raise ValueError(f'MOT row count mismatch: {path}')
    if len(data)<2 or not np.isfinite(data).all() or np.any(np.diff(data[:,0])<=0):
        raise ValueError(f'Invalid MOT values or timestamps: {path}')
    return dict(metadata=meta,columns=columns,data=data)


def y_up_to_z_up(points):
    # +90 degrees about X: (x,y,z) -> (x,-z,y), determinant +1.
    return np.asarray(points)[...,[0,2,1]]*np.array([1.,-1.,1.])


def audit(root: Path, output: Path) -> dict:
    root=root.resolve();output.mkdir(parents=True,exist_ok=True)
    files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name.startswith('.'))
    records=[];errors=[];headers=Counter();marker_sets=Counter()
    trcs={};angles={};example=None
    for path in files:
        rel=path.relative_to(root);parts=rel.parts
        if path.suffix.lower() not in ['.trc','.mot']:continue
        try:
            if path.suffix.lower()=='.trc':
                d=read_trc(path);kind='markers';time=d['time'];marker_sets[tuple(d['labels'])]+=1
                key=(parts[0],parts[2],parts[-2],path.stem.lower());trcs[key]=time
                row=dict(path=str(rel),subject=parts[0],activity=parts[2],kind=kind,frames=len(time),
                         duration_s=float(time[-1]-time[0]),fps=float(1/np.median(np.diff(time))),
                         missing_fraction=float(d['missing'].mean()),declared_fps=float(d['metadata']['DataRate']))
                if parts[2]=='LevelWalking' and example is None:example=(path,d)
            else:
                d=read_mot(path);time=d['data'][:,0]
                kind='joint_angles' if parts[1]=='JointAngle' else 'ground_reaction'
                headers[(kind,tuple(d['columns']))]+=1
                if kind=='joint_angles':angles[(parts[0],parts[2],parts[-2],path.stem.lower())]=time
                row=dict(path=str(rel),subject=parts[0],activity=parts[2],kind=kind,frames=len(time),
                         duration_s=float(time[-1]-time[0]),fps=float(1/np.median(np.diff(time))),missing_fraction=0.,
                         in_degrees=d['metadata'].get('inDegrees'))
            row['time_step_max_deviation_s']=float(np.max(abs(np.diff(time)-np.median(np.diff(time)))))
            records.append(row)
        except (ValueError,KeyError,IndexError) as e:errors.append({'path':str(rel),'error':str(e)})
    pairs=set(angles)&set(trcs)
    time_mismatches=[list(k) for k in sorted(pairs) if len(angles[k])!=len(trcs[k]) or not np.allclose(angles[k],trcs[k],atol=1e-7,rtol=0)]
    activities=[]
    for activity in sorted({r['activity'] for r in records}):
        counts=Counter(r['kind'] for r in records if r['activity']==activity)
        durations=[r['duration_s'] for r in records if r['activity']==activity and r['kind']=='markers']
        activities.append(dict(activity=activity,counts=dict(counts),duration_s_min_median_max=np.quantile(durations,[0,.5,1]).tolist() if durations else None))
    result=dict(source=str(root),subjects=sorted(p.name for p in root.iterdir() if p.is_dir()),
                files_by_extension=dict(Counter(p.suffix for p in files)),activities=activities,
                empty_files=[str(p.relative_to(root)) for p in files if p.stat().st_size==0],
                files_at_256kib=[str(p.relative_to(root)) for p in files if p.stat().st_size==262144],
                valid_source_files=len(records),errors=errors,paired_trc_and_angle_clips=len(pairs),time_mismatches=time_mismatches,
                unpaired_angle_clips=[list(k) for k in sorted(set(angles)-set(trcs))],
                marker_sets=[dict(labels=list(k),files=v) for k,v in marker_sets.items()],
                mot_columns=[dict(kind=k[0],columns=list(k[1]),files=v) for k,v in headers.items()],
                max_marker_missing_fraction=max((r['missing_fraction'] for r in records if r['kind']=='markers'),default=None),
                training_reference_created=False)
    if example:
        path,d=example
        np.savez_compressed(output/'example_markers_zup.npz',time_s=d['time']-d['time'][0],source_time_s=d['time'],
                            source_frame=d['frames'],markers_m=y_up_to_z_up(d['points_m']),marker_labels=np.asarray(d['labels']),
                            frequency=float(d['metadata']['DataRate']),source_path=str(path),is_training_reference=False)
        result['example_marker_observations']=str(output/'example_markers_zup.npz')
    (output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    keys=sorted({k for r in records for k in r})
    with (output/'clips.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(records)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--output',type=Path,default=Path('outputs/gait120_audit'))
    args=p.parse_args();result=audit(args.root,args.output)
    summary={k:result[k] for k in ['files_by_extension','activities','valid_source_files','paired_trc_and_angle_clips','max_marker_missing_fraction']}
    summary.update(errors=len(result['errors']),empty_files=len(result['empty_files']),files_at_256kib=len(result['files_at_256kib']),time_mismatches=len(result['time_mismatches']))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
