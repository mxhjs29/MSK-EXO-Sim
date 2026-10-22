"""Offline S101 EMG synergy extraction and constrained IK for MyoLegs26.

This module never starts PPO or changes installed model assets. Run on CPU with
CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS=cpu and MUJOCO_GL=osmesa.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from musclemimic.runner.workflows.gait120_synergy import (
    SPLITS, choose_rank, fit_nmf, load_clips, project_nonnegative,
    reconstruction_metrics,
)
from musclemimic.retargeting.gait120_retarget import project_joint_couplings, write_json

STEMS = ['rectfem', 'tibant', 'soleus', 'vasti', 'hamstrings', 'gastroc']
LABELS = ['RF', 'TA', 'SOL', 'VAS', 'HAM', 'GAS']
BODIES = ['pelvis', 'femur_r', 'tibia_r', 'talus_r', 'calcn_r', 'toes_r',
          'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l']
JOINTS = [f'{stem}_{side}' for side in ('r', 'l') for stem in
          ('hip_flexion', 'hip_adduction', 'hip_rotation', 'knee_angle', 'ankle_angle', 'mtp_angle')]


def group_six(emg_eleven):
    """Input follows gait120_synergy.GROUP_NAMES; merge normalized envelopes."""
    x = np.asarray(emg_eleven)
    return np.stack([x[..., 1, :], x[..., 3, :], x[..., 8, :],
                     (x[..., 0, :] + x[..., 2, :])/2,
                     (x[..., 4, :] + x[..., 5, :])/2,
                     (x[..., 6, :] + x[..., 7, :])/2], axis=-2)


def symmetric_coefficients(right, phase):
    """Periodic half-cycle shift; endpoint averaging applies to the proxy only.

    Right measured coefficients remain untouched. The left proxy closes the
    source clip seam by averaging its two endpoint values before interpolation.
    This is a symmetry assumption, not measured left-leg EMG.
    """
    periodic = np.asarray(right).copy()
    periodic[..., 0] = periodic[..., -1] = (right[..., 0] + right[..., -1])/2
    query = (np.asarray(phase) + .5) % 1.
    return np.stack([np.interp(query, phase, row) for row in periodic])


def bilateral_basis(right):
    """Same anatomical weights, separate right/left coefficient blocks."""
    zeros = np.zeros_like(right)
    return np.block([[right, zeros], [zeros, right]])


def continuous_root_quaternions(qpos):
    """Choose a continuous quaternion hemisphere without changing any pose."""
    q = np.asarray(qpos).copy()
    for i in range(1, len(q)):
        if np.dot(q[i-1, 3:7], q[i, 3:7]) < 0:
            q[i, 3:7] *= -1
    return q


def extract_synergies(output, source_dir):
    folder = output / 'synergy'
    folder.mkdir(parents=True, exist_ok=True)
    protocol = dict(subject='S101', splits=SPLITS, rank_candidates=list(range(1, 7)),
                    restarts=8, seed=42, phase_samples=101,
                    normalization='Published MVC envelopes, six-group equal means, train-only RMS weighting',
                    selection='Smallest K meeting train and validation global VAF >= .90 and each-group VAF >= .75',
                    VAF='1 - sum((E-WH)^2)/sum(E^2); not centered R2',
                    left_assumption='W_left = W_right; H_left(phi) = periodic H_right((phi+.5)%1)',
                    endpoint_rule='Average H endpoints only in periodic proxy before shifting; measured right H is unchanged',
                    scope='Six groups per side; remaining seven actuators per side have no inferred target',
                    split_limitation='Trial05 has previously been inspected; use as a descriptive holdout, not a fresh blind test',
                    physical_phase='Normalized published EMG clip interval, not newly detected heel-strike events',
                    policy_training=False,
                    source_sha256={name: hashlib.sha256((source_dir/name).read_bytes()).hexdigest()
                                   for name in ['RawData.mat', 'ProcessedData.mat']})
    write_json(folder/'protocol.json', protocol)
    raw, clips, names, fs = load_clips(source_dir, 101)
    arrays = group_six(raw)
    matrices = {split: np.concatenate([arrays[i] for i, c in enumerate(clips) if c['split'] == split], axis=1)
                for split in SPLITS}
    rms = np.sqrt(np.mean(matrices['train']**2, axis=1))
    rows, bases = [], {}
    for rank in protocol['rank_candidates']:
        best, records = None, []
        for restart in range(8):
            w, h, record = fit_nmf(matrices['train']/rms[:, None], rank, seed=42+100*rank+restart)
            records.append(record)
            if best is None or record['sse'] < best[1]['sse']:
                best = w, record
        w = best[0]*rms[:, None]
        w /= np.maximum(w.max(axis=0), 1e-20)
        w = w[:, np.argsort(np.argmax(w, axis=0), kind='stable')]
        row = dict(rank=rank, best_fit=best[1], restarts=records)
        for split in ('train', 'validation'):
            h = project_nonnegative(w/rms[:, None], matrices[split]/rms[:, None])
            row[split] = reconstruction_metrics(matrices[split], w@h)
        rows.append(row)
        bases[rank] = w
        print('NMF', rank, row['train']['global_uncentered_vaf'], row['validation']['global_uncentered_vaf'], flush=True)
    rank = choose_rank(rows)
    if rank is None:
        write_json(folder/'rank_sweep_failed.json', rows)
        raise ValueError('No candidate synergy rank passed the declared criteria')
    w, phase = bases[rank], np.linspace(0, 1, 101)
    h = np.stack([project_nonnegative(w/rms[:, None], a/rms[:, None]) for a in arrays])
    hl = np.stack([symmetric_coefficients(c, phase) for c in h])
    metrics = {}
    for split in SPLITS:
        idx = [i for i, c in enumerate(clips) if c['split'] == split]
        metrics[split] = reconstruction_metrics(matrices[split], np.concatenate([w@h[i] for i in idx], axis=1))
    common = dict(W_right=w, W_left=w.copy(), W_bilateral=bilateral_basis(w), phase=phase,
                  train_rms=rms, group_stems=STEMS, group_labels=LABELS,
                  bilateral_group_names=[f'{stem}_{s}' for s in ('r', 'l') for stem in STEMS],
                  bilateral_measured_mask=np.array([True]*6+[False]*6))
    np.savez_compressed(folder/'analysis_all_clips.npz', **common, H_right=h, H_left_assumed=hl,
                        emg_right=arrays, clip_ids=[c['id'] for c in clips], splits=[c['split'] for c in clips])
    idx = [i for i, c in enumerate(clips) if c['split'] == 'train']
    np.savez_compressed(folder/'training_prior.npz', **common, H_right=h[idx], H_left_assumed=hl[idx],
                        emg_right=arrays[idx], clip_ids=[clips[i]['id'] for i in idx])
    result = dict(selected_rank_per_leg=rank, bilateral_coefficient_dimension=2*rank,
                  groups=STEMS, clips=clips, rank_sweep=rows, metrics=metrics,
                  source_variable_names=names, source_fs_hz=fs,
                  right_H_endpoint_difference=np.abs(h[:, :, 0]-h[:, :, -1]).tolist(),
                  left_measured=False, policy_training=False)
    write_json(folder/'results.json', result)
    plot_synergies(folder, result, w, h, hl, arrays, phase)
    return result


def plot_synergies(folder, result, w, h, hl, emg, phase):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    im = axes[0].imshow(w, aspect='auto', vmin=0, vmax=1)
    axes[0].set(yticks=range(6), yticklabels=LABELS, xticks=range(w.shape[1]),
                xticklabels=[f'S{i+1}' for i in range(w.shape[1])], title='Shared right / left weights')
    fig.colorbar(im, ax=axes[0], label='Column-normalized weight')
    for split in ['train', 'validation']:
        axes[1].plot([r['rank'] for r in result['rank_sweep']],
                     [r[split]['global_uncentered_vaf']*100 for r in result['rank_sweep']], 'o-', label=split)
    axes[1].axhline(90, color='gray', ls='--')
    axes[1].axvline(w.shape[1], color='gray', ls=':')
    axes[1].set(xlabel='K per leg', ylabel='Global uncentered VAF (%)')
    axes[1].legend()
    fig.savefig(folder/'weights_and_rank.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), layout='constrained')
    for i, ax in enumerate(axes.flat):
        ax.plot(phase*100, emg[0, i], color='black', label='Right measured envelope')
        ax.plot(phase*100, (w@h[0])[i], color='tab:blue', label='Right NMF reconstruction')
        ax.plot(phase*100, (w@hl[0])[i], color='tab:orange', ls='--', label='Left symmetry assumption')
        ax.set(title=STEMS[i], xlabel='Clip phase (%)', ylabel='MVC fraction')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Trial01 Step01 / left target is synthesized, not measured')
    fig.savefig(folder/'bilateral_phase_targets.png', dpi=150)
    plt.close(fig)


def correct_dependent_ranges(spec):
    """Correct only dependent ranges in a private spec; keep muscle physics intact."""
    import mujoco
    from numpy.polynomial import Polynomial
    m = spec.copy().compile()
    children = {int(m.eq_obj1id[e]) for e in range(m.neq)
                if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and m.eq_active0[e]}
    changes = []
    for e in range(m.neq):
        if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT or not m.eq_active0[e]:
            continue
        child, parent = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        if parent < 0 or parent in children or not m.jnt_limited[parent]:
            raise ValueError('This MyoLegs26 audit expects independent, limited scalar coupling parents')
        if not m.jnt_limited[child]:
            continue
        poly = Polynomial(m.eq_data[e, :5])
        lo, hi = m.jnt_range[parent]-m.qpos0[m.jnt_qposadr[parent]]
        points = [lo, hi]+[r.real for r in poly.deriv().roots() if abs(r.imag)<1e-10 and lo<r.real<hi]
        values = poly(points)+m.qpos0[m.jnt_qposadr[child]]
        bounds = np.array([np.min(values)-1e-9, np.max(values)+1e-9])
        old = m.jnt_range[child]
        if bounds[0] < old[0]-2e-9 or bounds[1] > old[1]+2e-9:
            spec.joint(m.joint(child).name).range = bounds
            changes.append(dict(joint=m.joint(child).name, parent=m.joint(parent).name,
                                old_range=old.tolist(), new_range=bounds.tolist()))
    return changes


def build_target():
    from musclemimic.environments.humanoids.myofullbody_assist import MyoFullBodyAssistEnv
    env = MyoFullBodyAssistEnv(msk_key='myolegs26', disable_fingers=True, headless=True)
    original = env._model
    spec = env.mjspec.copy()
    changes = correct_dependent_ranges(spec)
    env.reload_mujoco(spec)
    for field in ['actuator_gainprm', 'actuator_biasprm', 'actuator_dynprm', 'body_mass',
                  'body_inertia', 'body_pos', 'body_quat', 'eq_data']:
        np.testing.assert_array_equal(getattr(original, field), getattr(env._model, field))
    return env, changes


class RetargetIK:
    """18 independent coordinates; scalar polynomial couplings are exact in FK."""
    def __init__(self, model):
        import mujoco
        self.model, self.data = model, mujoco.MjData(model)
        self.jids = [model.joint(n).id for n in JOINTS]
        self.cols = [int(model.jnt_qposadr[j]) for j in self.jids]
        self.bids = [model.body(n).id for n in BODIES]
        self.lo = np.r_[[-np.inf]*6, model.jnt_range[self.jids, 0]]
        self.hi = np.r_[[np.inf]*6, model.jnt_range[self.jids, 1]]
        self.nfev = []

    def decode(self, x):
        q = self.model.qpos0.copy()
        q[:3] = x[:3]
        xyzw = Rotation.from_rotvec(x[3:6]).as_quat()
        q[3:7] = xyzw[[3, 0, 1, 2]]
        q[self.cols] = x[6:]
        return project_joint_couplings(self.model, q)

    def solve(self, positions, orientations, initial, previous):
        import mujoco
        weights = np.array([2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2])[:, None]
        def residual(x):
            self.data.qpos[:] = self.decode(x)
            mujoco.mj_kinematics(self.model, self.data)
            pos = (self.data.xpos[self.bids]-positions)*weights
            current = self.data.xmat[self.bids].reshape(-1, 3, 3)
            # Matrix residual is smooth at the nearly-zero rotations seen in walking.
            rot = .035*(current-orientations)
            pieces = [pos.ravel(), rot.ravel(), .01*(x[6:]-initial[6:])]
            if previous is not None:
                pieces.append(.008*(x[3:]-previous[3:]))
            return np.concatenate(pieces)
        x0 = initial if previous is None else np.r_[initial[:6], previous[6:]]
        result = least_squares(residual, np.clip(x0, self.lo+1e-9, self.hi-1e-9),
                               bounds=(self.lo, self.hi), max_nfev=40,
                               ftol=1e-7, xtol=1e-7, gtol=1e-7, diff_step=1e-5)
        self.nfev.append(dict(nfev=result.nfev, success=bool(result.success), cost=float(result.cost)))
        return result.x, self.decode(result.x)


def retarget_clips(output, source_root, only_clip=None):
    import mujoco
    from loco_mujoco.trajectory import Trajectory
    from loco_mujoco.smpl.retargeting import max_penetration_with_floor
    from musclemimic.environments.humanoids.myofullbody_assist import MyoFullBodyAssistEnv
    from musclemimic.retargeting.msk_retarget import _make_target_trajectory
    from musclemimic.retargeting.gait120_retarget import inspect_trajectory

    result = json.loads((output/'synergy/results.json').read_text())
    factors = np.load(output/'synergy/analysis_all_clips.npz')
    source_manifest = json.loads((source_root/'manifest.json').read_text())
    source_records = {r['source']['id']: r for r in source_manifest['records']}
    env, changes = build_target()
    m, d = env._model, env._data
    model_dir = output/'model'
    model_dir.mkdir(exist_ok=True)
    (model_dir/'myolegs26_reference.xml').write_text(env.mjspec.to_xml())
    mujoco.mj_saveModel(m, str(model_dir/'myolegs26_reference.mjb'), None)
    write_json(model_dir/'range_corrections.json', dict(changes=changes, installed_assets_modified=False,
                reason='Dependent ranges cover original equality polynomials over original primary joint ranges',
                muscle_physics_unchanged=True))
    source_env = MyoFullBodyAssistEnv(msk_key='myofullbody', disable_fingers=True, headless=True)
    sm, sd = source_env._model, source_env._data
    source_bids = [sm.body(n).id for n in BODIES]
    body_ids = [m.body(n).id for n in BODIES]
    # Root-to-pelvis transform is rigid in the lower-body model.
    d.qpos[:] = m.qpos0
    mujoco.mj_kinematics(m, d)
    pelvis_rel = d.xmat[m.body('pelvis').id].reshape(3, 3).copy()
    pelvis_offset = d.xpos[m.body('pelvis').id]-m.qpos0[:3]
    priors = [f'{stem}_{s}' for s in ('r', 'l') for stem in STEMS]
    names = [m.actuator(i).name for i in range(m.nu)]
    write_json(model_dir/'activation_mapping.json', dict(actuator_names=names, prior_groups=priors,
                prior_actuator_indices=[names.index(n) for n in priors],
                prior_activation_addresses=[int(m.actuator_actadr[names.index(n)]) for n in priors],
                observed_mask=[n in priors[:6] for n in names],
                prior_mask=[n in priors for n in names],
                unmeasured_uninferred=[n for n in names if n not in priors]))
    for clip_index, clip in enumerate(result['clips']):
        if only_clip and clip['id'] != only_clip:
            continue
        folder = output/'clips'/clip['id']
        folder.mkdir(parents=True, exist_ok=True)
        source_folder = source_root/'clips'/clip['id']
        src = Trajectory.load(source_folder/'trajectory.npz', backend=np)
        comparison = dict(np.load(source_folder/'comparison.npz'))
        source_q = np.asarray(src.data.qpos)
        positions, orientations, initial = [], [], []
        for q in source_q:
            sd.qpos[:] = q
            mujoco.mj_kinematics(sm, sd)
            positions.append(sd.xpos[source_bids].copy())
            orientations.append(sd.xmat[source_bids].reshape(-1, 3, 3).copy())
            rot = orientations[-1][0]@pelvis_rel.T
            root = positions[-1][0]-rot@pelvis_offset
            joints = [float(q[sm.joint(n).qposadr[0]])*(-1 if n.startswith('knee_angle') else 1) for n in JOINTS]
            initial.append(np.r_[root, Rotation.from_matrix(rot).as_rotvec(), joints])
        positions, orientations = np.asarray(positions), np.asarray(orientations)
        solver = RetargetIK(m)
        qposes, previous, floor_lifts = [], None, []
        actual_positions = []
        for i, x in enumerate(initial):
            previous, q = solver.solve(positions[i], orientations[i], x, previous)
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            penetration, _, _ = max_penetration_with_floor(m, d)
            lift = max(0., -penetration)
            q[2] += lift
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            qposes.append(q)
            floor_lifts.append(lift)
            actual_positions.append(d.xpos[body_ids].copy())
        qposes = continuous_root_quaternions(np.asarray(qposes))
        source_times = comparison['source_time_s']
        np.testing.assert_allclose(np.diff(source_times), .01, atol=1e-7)
        metadata = dict(source='Gait120', source_clip=clip['id'], source_time_start_s=float(source_times[0]),
                        source_time_end_s=float(source_times[-1]), msk_key='myolegs26',
                        target_model_variant='myolegs26_coupling_consistent_reference_v1',
                        method='TRC -> existing SMPL-H/GMR fullbody fit -> independent-coordinate MyoLegs26 IK',
                        training_dynamics_validated=False, left_emg_is_symmetric_assumption=True)
        traj = _make_target_trajectory(m, qposes, None, 100., [0, len(qposes)], metadata)
        metrics, sites = inspect_trajectory(env, traj)
        if metrics['joint_limit_violations'] or metrics['equality_residual_max'] > 1e-7 or metrics['floor_penetration_max_m'] > 1e-5:
            raise ValueError(f'Invalid retarget {clip["id"]}: {metrics}')
        env.load_trajectory(traj=traj, warn=False, cache_type='sparse', site_names=env.sites_for_mimic)
        error = np.linalg.norm(np.asarray(actual_positions)-positions, axis=-1)
        metrics.update(environment_load_passed=True, transfer_body_error_mean_m=float(error.mean()),
                       transfer_body_error_p95_m=float(np.percentile(error, 95)),
                       transfer_body_error_max_m=float(error.max()),
                       floor_lift_max_m=float(max(floor_lifts)),
                       ik_nonconverged_frames=sum(not r['success'] for r in solver.nfev),
                       source_marker_fit_mean_mm=source_records[clip['id']]['metrics']['marker_fit_mean_mm'])
        traj.save(str(folder/'trajectory.npz'))
        comparison.update(target_sites_m=sites, target_site_names=np.asarray(env.sites_for_mimic),
                          source_fullbody_body_positions=positions, target_body_positions=np.asarray(actual_positions),
                          body_names=np.asarray(BODIES), transfer_body_errors_m=error)
        np.savez_compressed(folder/'comparison.npz', **comparison)
        physical_phase = (source_times-clip['start_time_s'])/(clip['last_emg_time_s']-clip['start_time_s'])
        if physical_phase.min() < 0 or physical_phase.max() > 1:
            raise ValueError('Reference extends beyond the available EMG interval')
        def sample(values):
            return np.stack([np.interp(physical_phase, factors['phase'], row) for row in values], axis=1)
        hr = sample(factors['H_right'][clip_index])
        hl = sample(factors['H_left_assumed'][clip_index])
        np.savez_compressed(folder/'emg_reference.npz', source_time_s=source_times, source_phase=physical_phase,
                            group_names=priors, H_right=hr, H_left_assumed=hl,
                            right_measured=sample(factors['emg_right'][clip_index]),
                            right_reconstructed=hr@factors['W_right'].T,
                            left_assumed=hl@factors['W_left'].T,
                            measured_group_mask=[True]*6+[False]*6,
                            split=clip['split'], activation_dynamics_applied=False)
        write_json(folder/'result.json', dict(source=source_records[clip['id']]['source'], split=clip['split'],
                    metrics=metrics, trajectory=str(folder/'trajectory.npz'), ik_solver=solver.nfev,
                    source_trajectory_sha256=hashlib.sha256((source_folder/'trajectory.npz').read_bytes()).hexdigest()))
        print('RETARGET', clip['id'], json.dumps(metrics), flush=True)
    summarize_collection(output)


def concatenate_isolated(trajectories, metadata):
    """The library returns the first input info by reference; isolate each export."""
    from loco_mujoco.trajectory import Trajectory
    copies = [Trajectory(info=copy.deepcopy(t.info), data=t.data) for t in trajectories]
    for t in copies:
        t.info.metadata = None
    combined = Trajectory.concatenate(copies, backend=np)
    combined.info.metadata = metadata
    return combined


def summarize_collection(output):
    from loco_mujoco.trajectory import Trajectory
    result = json.loads((output/'synergy/results.json').read_text())
    records, trajectories = [], []
    for clip in result['clips']:
        folder = output/'clips'/clip['id']
        if not (folder/'result.json').exists():
            continue
        records.append(json.loads((folder/'result.json').read_text()))
        trajectory = Trajectory.load(folder/'trajectory.npz', backend=np)
        trajectory.info.metadata = None
        trajectories.append(trajectory)
    for split in ['all', 'train', 'validation', 'test']:
        chosen = [t for r, t in zip(records, trajectories) if split == 'all' or r['split'] == split]
        if chosen:
            metadata = dict(msk_key='myolegs26', split=split,
                target_model_variant='myolegs26_coupling_consistent_reference_v1',
                clip_ids=[r['source']['id'] for r in records if split == 'all' or r['split'] == split],
                independent_clip_boundaries=True, training_dynamics_validated=False)
            combined = concatenate_isolated(chosen, metadata)
            combined.save(str(output/f'levelwalking_{split}.npz'))
    write_json(output/'manifest.json', dict(complete=len(records)==len(result['clips']), records=records,
                 msk_key='myolegs26', model_xml='model/myolegs26_reference.xml', model_mjb='model/myolegs26_reference.mjb',
                 synergy_rank_per_leg=result['selected_rank_per_leg'], left_emg_measured=False,
                 left_prior_assumption='Bilateral weight symmetry and half-cycle phase shift',
                 trajectory_is_kinematic_reference=True, policy_training=False,
                 installed_assets_modified=False, ppo_code_modified=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('outputs/gait120_myolegs26_synergy_v1'))
    p.add_argument('--source-dir', type=Path, default=Path('outputs/gait120_emg_comparison_v1/data'))
    p.add_argument('--source-motion', type=Path, default=Path('outputs/gait120_levelwalking'))
    p.add_argument('--stage', choices=['synergy', 'retarget', 'summarize', 'all'], default='all')
    p.add_argument('--clip')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ['synergy', 'all']:
        extract_synergies(args.output, args.source_dir)
    if args.stage in ['retarget', 'all']:
        retarget_clips(args.output, args.source_motion, args.clip)
    if args.stage == 'summarize':
        summarize_collection(args.output)


if __name__ == '__main__':
    main()
