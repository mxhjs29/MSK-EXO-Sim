"""Retarget intact Gait120 level walking TRCs to the current MyoFullBody.

Each participant shares a marker/shape calibration. Pose warm starts and
velocities never cross clip boundaries. All time alignment remains at 100 Hz.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .gait120_audit import read_trc, y_up_to_z_up

LOG = logging.getLogger(__name__)
SETTINGS = dict(stage1_iters=8, stage2_iters=60, n_ref_frames=6,
                surface_model_type="smplh", gender="neutral", optimize_toes=True)


def write_json(path, value):
    def plain(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, Path): return str(v)
        raise TypeError(type(v).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=plain) + "\n")


def discover(root: Path):
    clips, excluded = [], []
    for path in sorted(root.glob("*/MotionCapture/LevelWalking/TRC/*/*.trc")):
        try:
            d = read_trc(path)
            if d["missing"].any(): raise ValueError("Missing marker observations")
            if not np.allclose(np.diff(d["time"]), .01, atol=1e-7, rtol=0):
                raise ValueError("Expected regular 100 Hz timestamps")
            rel = path.relative_to(root)
            clips.append(dict(id="_".join([rel.parts[0], path.parent.name, path.stem]),
                              subject=rel.parts[0], path=str(rel), frames=len(d["time"]),
                              source_start_s=float(d["time"][0]), source_end_s=float(d["time"][-1]),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        except (ValueError, IndexError, KeyError) as exc:
            excluded.append(dict(path=str(path.relative_to(root)), reason=str(exc)))
    return clips, excluded


def fit_subject(root: Path, output: Path, clips: list[dict], device: str):
    from loco_mujoco.smpl.retargeting import get_smpl_model_path
    from musclemimic.web_viewer.c3d_to_smpl import fit_smpl_to_markers

    subject = clips[0]["subject"]
    folder = output / "subject_fits"; folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{subject}.npz"
    protocol = dict(settings=SETTINGS, clips=clips, coordinate_rotation="(x,y,z)->(x,-z,y)",
                    fit_api_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath("web_viewer/c3d_to_smpl.py").read_bytes()).hexdigest())
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if path.exists():
        with np.load(path) as cache:
            if str(cache["signature"]) == signature:
                LOG.info("Reusing subject fit %s", subject)
                return path
        raise ValueError(f"Stale subject fit {path}; select a new output directory")
    sources = [read_trc(root / c["path"]) for c in clips]
    labels = sources[0]["labels"]
    assert all(d["labels"] == labels for d in sources)
    splits = np.cumsum([0] + [len(d["time"]) for d in sources])
    markers = np.concatenate([y_up_to_z_up(d["points_m"]) for d in sources])
    result = fit_smpl_to_markers(markers, labels, 100., get_smpl_model_path(),
                                device=device, source_name=subject,
                                sequence_starts=tuple(splits[:-1]), **SETTINGS)
    error = result["debug"]["marker_error"]
    np.savez_compressed(path, signature=signature, splits=splits,
                        markers_m=markers, marker_labels=np.asarray(labels),
                        fitted_markers_m=error.pop("predicted_markers_m"),
                        marker_error_m=error.pop("per_marker_error_m"),
                        fitted_marker_labels=np.asarray(result["debug"]["canonical_labels"]),
                        **{k:v for k,v in result.items() if k != "debug"})
    write_json(path.with_suffix(".json"), dict(protocol=protocol, debug=result["debug"]))
    LOG.info("SUBJECT_FIT_COMPLETE %s mean %.2f mm p95 %.2f mm", subject, error["mean_mm"], error["p95_mm"])
    return path


def inspect_trajectory(env, trajectory):
    import mujoco
    from loco_mujoco.smpl.retargeting import max_penetration_with_floor

    m, d = env._model, env._data
    q = np.asarray(trajectory.data.qpos)
    v = np.asarray(trajectory.data.qvel)
    assert q.shape[1] == m.nq and v.shape == (len(q), m.nv)
    assert np.isfinite(q).all() and np.isfinite(v).all()
    limits = {}
    for jid in range(m.njnt):
        if m.jnt_limited[jid]:
            col = int(m.jnt_qposadr[jid]); lo, hi = m.jnt_range[jid]
            violation = float(max(0., np.max(np.maximum(lo-q[:, col], q[:, col]-hi))))
            if violation > 1e-5: limits[m.joint(jid).name] = violation
    equality, penetration, feet, skeleton = [], [], [], []
    sites = [m.site(name).id for name in env.sites_for_mimic]
    feet_ids = [m.site(name).id for name in ["left_ankle_mimic", "left_toes_mimic", "right_ankle_mimic", "right_toes_mimic"]]
    for row in q:
        d.qpos[:] = row; mujoco.mj_forward(m, d)
        eq = d.efc_pos[d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
        equality.append(float(np.max(abs(eq))) if len(eq) else 0.)
        pen, _, _ = max_penetration_with_floor(m, d)
        penetration.append(float(max(0., -pen)))
        skeleton.append(d.site_xpos[sites].copy()); feet.append(d.site_xpos[feet_ids].copy())
    feet = np.asarray(feet)
    velocity = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=-1) * trajectory.info.frequency
    # A diagnostic based on low foot sites, not a measured ground-contact label.
    low = feet[1:, :, 2] < np.quantile(feet[:, :, 2], .1, axis=0) + .025
    metrics = dict(frames=len(q), frequency_hz=trajectory.info.frequency, nq=m.nq, nv=m.nv, nu=m.nu,
                   finite=True, quaternion_norm_max_error=float(np.max(abs(np.linalg.norm(q[:,3:7],axis=1)-1))),
                   joint_limit_violations=limits, equality_residual_max=float(max(equality)),
                   floor_penetration_max_m=float(max(penetration)),
                   low_foot_horizontal_speed_mean_m_s=float(velocity[low].mean()) if low.any() else None,
                   root_speed_max_m_s=float(np.linalg.norm(v[:,:3],axis=1).max()))
    return metrics, np.asarray(skeleton)


def retarget_subject(root: Path, output: Path, clips: list[dict], fit_path: Path):
    from loco_mujoco.smpl.retargeting import fit_gmr_motion
    from musclemimic.web_viewer.c3d_to_smpl import save_motion_data_as_amass_smplh_npz
    from musclemimic.environments.humanoids.myofullbody_assist import MyoFullBodyAssistEnv
    from .msk_profiles import load_fullbody_solver_config

    conf = load_fullbody_solver_config()
    conf.env_params.disable_fingers = True
    conf.env_params.headless = True
    env = MyoFullBodyAssistEnv(msk_key="myofullbody", disable_fingers=True, headless=True)
    data = dict(np.load(fit_path))
    for i, clip in enumerate(clips):
        folder = output / "clips" / clip["id"]; folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / "result.json"
        if result_path.exists():
            cached=json.loads(result_path.read_text())
            if cached["source"]["sha256"] != clip["sha256"]: raise ValueError("Source changed")
            LOG.info("Reusing retarget %s", clip["id"]); continue
        a, b = data["splits"][i:i+2]
        motion = {k:data[k][a:b] for k in ["pose_aa", "fullpose", "trans"]}
        motion.update(betas=data["betas"], fps=100., gender="neutral")
        source = read_trc(root / clip["path"])
        amass = save_motion_data_as_amass_smplh_npz(motion, folder / "smplh.npz")
        trajectory, analysis = fit_gmr_motion(
            "MyoFullBodyAssistEnv", conf, str(amass), LOG,
            dict(target_fps=100, use_fitted_shape=True, initialize_root_from_target=True,
                 joint_equality_cost=1000.))
        # GMR's centered differentiation deliberately discards first/last frames.
        assert len(trajectory.data.qpos) == clip["frames"]-2
        trajectory.info.metadata = dict(source="Gait120", source_clip=clip["path"],
                                        source_sha256=clip["sha256"], source_time_start_s=float(source["time"][1]),
                                        source_time_end_s=float(source["time"][-2]),
                                        method="TRC -> SMPL-H marker fit -> GMR", msk_key="myofullbody",
                                        disable_fingers=True, training_dynamics_validated=False)
        metrics, skeleton = inspect_trajectory(env, trajectory)
        trajectory.save(str(folder / "trajectory.npz"))
        # Exercise the actual imitation trajectory reader, not only NPZ loading.
        env.load_trajectory(traj=trajectory, warn=False, cache_type="sparse", site_names=env.sites_for_mimic)
        metrics["environment_load_passed"] = True
        marker_error = data["marker_error_m"][a:b]
        metrics.update(marker_fit_mean_mm=float(np.nanmean(marker_error)*1000),
                       marker_fit_p95_mm=float(np.nanpercentile(marker_error,95)*1000),
                       marker_fit_max_mm=float(np.nanmax(marker_error)*1000),
                       gmr_body_position_error_mean_m=float(np.mean(analysis["pos_error"])),
                       gmr_body_position_error_p95_m=float(np.percentile(analysis["pos_error"],95)))
        np.savez_compressed(folder / "comparison.npz", source_time_s=source["time"][1:-1],
                            source_frame=source["frames"][1:-1], markers_m=data["markers_m"][a+1:b-1],
                            marker_labels=data["marker_labels"], fitted_markers_m=data["fitted_markers_m"][a+1:b-1],
                            fitted_marker_labels=data["fitted_marker_labels"],
                            target_sites_m=skeleton, target_site_names=np.asarray(env.sites_for_mimic),
                            gmr_body_errors_m=analysis["pos_error"][1:-1])
        write_json(result_path, dict(source=clip, metrics=metrics, trajectory=str(folder / "trajectory.npz")))
        LOG.info("CLIP_COMPLETE %s %s", clip["id"], metrics)


def summarize(root: Path, output: Path, clips, excluded):
    from loco_mujoco.trajectory import Trajectory
    records=[]; trajectories=[]; original_trajectories=[]
    for c in clips:
        folder=output / "clips" / c["id"]
        if not (folder / "result.json").exists(): continue
        records.append(json.loads((folder / "result.json").read_text()))
        trajectory=Trajectory.load(folder / "trajectory.npz", backend=np)
        trajectory.info.metadata=None  # clip-specific provenance lives in the collection manifest.
        trajectories.append(trajectory)
        if (folder / "trajectory_gmr.npz").exists():
            original=Trajectory.load(folder / "trajectory_gmr.npz",backend=np)
            original.info.metadata=None;original_trajectories.append(original)
    if trajectories:
        combined=Trajectory.concatenate(trajectories, backend=np)
        combined.info.metadata=dict(source="Gait120", activity="LevelWalking", msk_key="myofullbody",
                                    disable_fingers=True, clip_ids=[r["source"]["id"] for r in records],
                                    target_model_variant="gait120_knee_coupling_consistent_v1",
                                    training_dynamics_validated=False)
        combined.save(str(output / "levelwalking_trajectory.npz"))
    if len(original_trajectories)==len(trajectories) and original_trajectories:
        original=Trajectory.concatenate(original_trajectories,backend=np)
        original.info.metadata=dict(source="Gait120",target_model_variant="original_myofullbody",
                                    clip_ids=[r["source"]["id"] for r in records],
                                    note="Soft GMR joint-coupling residuals remain; see before_coupling_projection_metrics")
        original.save(str(output / "levelwalking_original_model.npz"))
    write_json(output / "manifest.json", dict(source_root=root.resolve(), settings=SETTINGS,
               complete=len(records)==len(clips), available_clips=len(clips), completed_clips=len(records),
               target_model_variant="gait120_knee_coupling_consistent_v1",
               model_corrections="model/range_corrections.json",
               retargeting_settings=dict(target_fps=100,use_fitted_shape=True,
                                         initialize_root_from_target=True,joint_equality_cost=1000.),
               excluded=excluded, records=records,
               notes=["Independent clips; no looping, stitching or time stretching",
                      "Source 100 Hz; first/last frames removed for centered velocity",
                      "Joint reference animation, not a muscle-policy rollout; no training performed"]))
    with (output / "clips.csv").open("w", newline="") as stream:
        columns=["id","subject","source_path","trajectory_path","frames","frequency_hz",
                 "marker_fit_mean_mm","marker_fit_p95_mm","gmr_body_position_error_mean_m",
                 "equality_residual_max","floor_penetration_max_m"]
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for r in records:
            row={k:r["metrics"][k] for k in columns[4:]}
            row.update(id=r["source"]["id"],subject=r["source"]["subject"],source_path=r["source"]["path"],trajectory_path=r["trajectory"])
            writer.writerow(row)


def project_joint_couplings(model, qpos):
    """Enforce active scalar joint polynomials in dependency order."""
    import mujoco
    q = np.asarray(qpos, dtype=np.float64).copy()
    pending = [i for i in range(model.neq)
               if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and model.eq_active0[i]]
    while pending:
        dependent = {int(model.eq_obj1id[i]) for i in pending}
        ready = [i for i in pending if int(model.eq_obj2id[i]) not in dependent]
        if not ready: raise ValueError("Cyclic joint coupling dependencies")
        for i in ready:
            j1, j2 = int(model.eq_obj1id[i]), int(model.eq_obj2id[i])
            a = int(model.jnt_qposadr[j1])
            x = 0. if j2 < 0 else q[..., int(model.jnt_qposadr[j2])] - model.qpos0[int(model.jnt_qposadr[j2])]
            q[..., a] = model.qpos0[a] + np.polynomial.polynomial.polyval(x, model.eq_data[i, :5])
            pending.remove(i)
    return q


def constrain_coupled_roots(model, qpos):
    """Project independent joints to intervals allowed by all coupled limits.

    Clamping dependent coordinates independently breaks anatomical polynomials.
    Instead derive the exact polynomial preimages of their allowed ranges.
    """
    import mujoco
    from numpy.polynomial import Polynomial
    equations={int(model.eq_obj1id[i]):i for i in range(model.neq)
               if model.eq_type[i]==mujoco.mjtEq.mjEQ_JOINT and model.eq_active0[i]}
    expressions={};visiting=set()
    def expression(jid):
        if jid in expressions:return expressions[jid]
        if jid in visiting:raise ValueError("Cyclic joint coupling dependencies")
        if jid not in equations:return jid,Polynomial([0.,1.])
        visiting.add(jid);eid=equations[jid];parent=int(model.eq_obj2id[eid])
        if parent<0:root,poly=None,Polynomial([0.])
        else:
            root,poly=expression(parent)
            poly=poly-model.qpos0[int(model.jnt_qposadr[parent])]
        result=Polynomial(model.eq_data[eid,:5])(poly)+model.qpos0[int(model.jnt_qposadr[jid])]
        visiting.remove(jid);expressions[jid]=(root,result)
        return root,result
    groups={}
    for jid in equations:
        root,poly=expression(jid)
        if root is not None:groups.setdefault(root,{root:Polynomial([0.,1.])})[jid]=poly
    q=np.asarray(qpos,dtype=float).copy();intervals={}
    for root,polys in groups.items():
        constraints=[(poly,*model.jnt_range[jid]) for jid,poly in polys.items() if model.jnt_limited[jid]]
        if not constraints:continue
        def valid(x):return all(lo-1e-9<=poly(x)<=hi+1e-9 for poly,lo,hi in constraints)
        boundaries=[]
        for poly,lo,hi in constraints:
            for limit in (lo,hi):
                boundaries.extend(float(r.real) for r in (poly-limit).roots() if abs(r.imag)<1e-8)
        boundaries=sorted(set(boundaries))
        edges=[-np.inf,*boundaries,np.inf];allowed=[]
        for lo,hi in zip(edges,edges[1:]):
            mid=(lo+hi)/2 if np.isfinite(lo) and np.isfinite(hi) else (hi-max(1.,abs(hi)) if np.isfinite(hi) else (lo+max(1.,abs(lo)) if np.isfinite(lo) else 0.))
            if valid(mid):allowed.append((lo,hi))
        allowed.extend((v,v) for v in boundaries if valid(v))
        if not allowed:raise ValueError(f"No feasible coupled interval for {model.joint(root).name}")
        col=int(model.jnt_qposadr[root]);values=q[...,col]
        candidates=np.stack([np.clip(values,lo,hi) for lo,hi in allowed],axis=-1)
        closest=np.argmin(abs(candidates-values[...,None]),axis=-1)
        q[...,col]=np.take_along_axis(candidates,closest[...,None],axis=-1)[...,0]
        intervals[model.joint(root).name]=allowed
    return q,intervals


def make_reference_env(**kwargs):
    """Build an isolated target with knee ranges consistent with its equations.

    The installed myo_sim asset and existing training environments are unchanged.
    Only dependent knee coordinates are adjusted; primary knee flexion limits,
    couplings, body geometry and muscle parameters remain unchanged.
    """
    import mujoco
    from numpy.polynomial import Polynomial
    from musclemimic.environments.humanoids.myofullbody_assist import MyoFullBodyAssistEnv
    params=dict(msk_key="myofullbody",disable_fingers=True,headless=True);params.update(kwargs)
    env=MyoFullBodyAssistEnv(**params)
    model=env._model;spec=env.mjspec.copy();changes=[]
    for eid in range(model.neq):
        if model.eq_type[eid]!=mujoco.mjtEq.mjEQ_JOINT or not model.eq_active0[eid]:continue
        j1,j2=int(model.eq_obj1id[eid]),int(model.eq_obj2id[eid])
        if j2<0 or model.joint(j2).name not in {"knee_angle_r","knee_angle_l"}:continue
        if not model.jnt_limited[j1]:continue
        ref1=model.qpos0[int(model.jnt_qposadr[j1])];ref2=model.qpos0[int(model.jnt_qposadr[j2])]
        lo,hi=model.jnt_range[j2]-ref2
        polynomial=Polynomial(model.eq_data[eid,:5])
        critical=[lo,hi]+[r.real for r in polynomial.deriv().roots() if abs(r.imag)<1e-8 and lo<r.real<hi]
        values=polynomial(critical)+ref1
        expected=np.array([min(values),max(values)])
        old=model.jnt_range[j1].copy()
        if expected[0]>=old[0]-1e-10 and expected[1]<=old[1]+1e-10:continue
        # Reversed left-knee translation bounds are replaced, not widened
        # into an artificial symmetric translation range.
        new=expected+np.array([-1e-9,1e-9])
        name=model.joint(j1).name;spec.joint(name).range=new
        changes.append(dict(joint=name,parent=model.joint(j2).name,old_range=old,new_range=new,
                            reason="Dependent range did not contain the equality polynomial over the primary knee range"))
    env.reload_mujoco(spec)
    return env,changes


def refine_collection(root, output, clips, excluded, *, finalize=True):
    """Remove residual soft-IK coupling error and rebuild dependent kinematics."""
    import mujoco
    import shutil
    from loco_mujoco.trajectory import Trajectory
    from loco_mujoco.smpl.retargeting import max_penetration_with_floor
    env,changes=make_reference_env()
    m,d=env._model,env._data
    (output / "model").mkdir(exist_ok=True)
    (output / "model/myofullbody_reference.xml").write_text(env.mjspec.to_xml())
    mujoco.mj_saveModel(m,str(output / "model/myofullbody_reference.mjb"),None)
    write_json(output / "model/range_corrections.json",dict(variant="gait120_knee_coupling_consistent_v1",
              changes=changes,original_model_assets_modified=False,
              note="Use this target variant for the projected reference; the original GMR trajectory is also preserved"))
    for clip in clips:
        folder=output / "clips" / clip["id"]
        record=json.loads((folder / "result.json").read_text())
        original=folder / "trajectory_gmr.npz"
        if not original.exists(): shutil.copy2(folder / "trajectory.npz",original)
        trajectory=Trajectory.load(original,backend=np)
        before=np.asarray(trajectory.data.qpos)
        limited,intervals=constrain_coupled_roots(m,before)
        q=project_joint_couplings(m,limited)
        sites=[m.site(n).id for n in trajectory.info.site_names]
        positions=[];matrices=[]
        for row in q:
            d.qpos[:]=row;mujoco.mj_forward(m,d)
            pen,_,_=max_penetration_with_floor(m,d)
            if pen < 0:
                row[2]-=pen;d.qpos[:]=row;mujoco.mj_forward(m,d)
            positions.append(d.site_xpos[sites].copy());matrices.append(d.site_xmat[sites].copy())
        differences=np.empty((len(q)-1,m.nv))
        for i in range(len(q)-1): mujoco.mj_differentiatePos(m,differences[i],.01,q[i],q[i+1])
        v=np.concatenate([differences[:1],.5*(differences[:-1]+differences[1:]),differences[-1:]])
        trajectory.data=trajectory.data.replace(qpos=q,qvel=v,site_xpos=np.asarray(positions),site_xmat=np.asarray(matrices))
        trajectory.info.metadata["exact_joint_coupling_projection"]=True
        trajectory.info.metadata["target_model_variant"]="gait120_knee_coupling_consistent_v1"
        metrics,skeleton=inspect_trajectory(env,trajectory)
        assert not metrics["joint_limit_violations"], metrics["joint_limit_violations"]
        assert metrics["equality_residual_max"] < 1e-7
        assert metrics["floor_penetration_max_m"] < 1e-5
        env.load_trajectory(traj=trajectory,warn=False,cache_type="sparse",site_names=env.sites_for_mimic)
        metrics["environment_load_passed"]=True
        metrics["coupling_projection_max_qpos_change"]=float(abs(q-before).max())
        metrics["independent_joint_limit_adjustment_max_rad"]=float(abs(limited-before).max())
        original_comparison=folder / "comparison_gmr.npz"
        if not original_comparison.exists():shutil.copy2(folder / "comparison.npz",original_comparison)
        comparison=dict(np.load(original_comparison))
        metrics["coupling_projection_site_shift_max_m"]=float(np.linalg.norm(skeleton-comparison["target_sites_m"],axis=-1).max())
        record["before_coupling_projection_metrics"]=record.get("before_coupling_projection_metrics",record["metrics"].copy())
        record["metrics"].update(metrics)
        record["coupled_joint_feasible_intervals"]=intervals
        comparison["target_sites_m"]=skeleton
        np.savez_compressed(folder / "comparison.npz",**comparison)
        trajectory.save(str(folder / "trajectory.npz"))
        write_json(folder / "result.json",record)
        LOG.info("REFINED %s equality %.3g",clip["id"],metrics["equality_residual_max"])
    if finalize:summarize(root,output,clips,excluded)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("datasets/Gait120_101_to_110"))
    p.add_argument("--output", type=Path, default=Path("outputs/gait120_levelwalking"))
    p.add_argument("--subject"); p.add_argument("--device", default="cuda:0")
    p.add_argument("--stage", choices=["all","fit","retarget","summarize","refine"], default="all")
    args=p.parse_args(); logging.basicConfig(level=logging.INFO)
    clips, excluded=discover(args.source)
    if args.stage=="refine": refine_collection(args.source,args.output,clips,excluded); return
    if args.stage=="summarize": summarize(args.source,args.output,clips,excluded); return
    subjects=sorted({c["subject"] for c in clips})
    for subject in subjects:
        if args.subject and args.subject!=subject: continue
        selected=[c for c in clips if c["subject"]==subject]
        fit_path=args.output / "subject_fits" / f"{subject}.npz"
        if args.stage in ["all","fit"]: fit_path=fit_subject(args.source,args.output,selected,args.device)
        if args.stage in ["all","retarget"]: retarget_subject(args.source,args.output,selected,fit_path)
    if not args.subject and args.stage=="all": refine_collection(args.source,args.output,clips,excluded)


if __name__ == "__main__": main()
