"""CPU checkpoint evaluation: actual muscle states, six observed groups, videos."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from omegaconf import OmegaConf

from musclemimic.algorithms import PPOJax
from musclemimic.runner.eval_utils import load_checkpoint
from musclemimic.runner.workflows.gait120_synergy_train import cpu_env, save_json, CLIP, STEMS


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 10 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def render_video(destination, env, arrays):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    model = env._model
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.2, 125, -12
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    data = mujoco.MjData(model)
    try:
        with imageio.get_writer(destination/'motion.mp4', fps=25, codec='libx264', quality=7) as writer:
            for i in range(len(arrays['qpos'])):
                views = []
                for title, q in [('reference', arrays['reference_qpos'][i]), ('actual policy', arrays['qpos'][i])]:
                    data.qpos[:] = q
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = [*q[:2], .9]
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    frame = Image.fromarray(renderer.render().copy())
                    ImageDraw.Draw(frame).text((12, 12), f'{title} | t={arrays["time_s"][i]:.2f}s | 0.25x', fill='white')
                    views.append(np.asarray(frame))
                writer.append_data(np.concatenate(views, axis=1))
    finally:
        renderer.close()


def cross_correlation(a, b, max_lag=10):
    """Positive lag means activation follows EMG; lag is a descriptive diagnostic."""
    rows = []
    for lag in range(-max_lag, max_lag+1):
        x, y = (a[lag:], b[:len(b)-lag]) if lag >= 0 else (a[:len(a)+lag], b[-lag:])
        r = pearson(x, y) if len(x) >= max(10, len(a)//2) else None
        if r is not None:
            rows.append(dict(lag_ms=lag*10, r=r, overlap_samples=len(x)))
    return max(rows, key=lambda row: row['r']) if rows else None


def correlations(actual, expected):
    groups = {}
    for i, name in enumerate(STEMS):
        a, e = actual[:, i], expected[:, i]
        groups[name] = dict(pearson_r=pearson(a, e), rmse=float(np.sqrt(np.mean((a-e)**2))) if len(a) else None,
            activation_std=float(np.std(a)) if len(a) else None,
            peak_cross_correlation=cross_correlation(a, e))
    valid = [g['pearson_r'] for g in groups.values() if g['pearson_r'] is not None]
    core = [groups[name]['pearson_r'] for name in STEMS[:3]]
    return dict(groups=groups, valid_group_count=len(valid),
        all_six_mean_r=float(np.mean(valid)) if len(valid) == 6 else None,
        valid_only_mean_r=float(np.mean(valid)) if valid else None,
        rf_ta_sol_mean_r=float(np.mean(core)) if all(v is not None for v in core) else None,
        invalid_groups=[name for name, row in groups.items() if row['pearson_r'] is None])


def rollout(env, policy):
    obs = env.reset(jax.random.PRNGKey(42))
    m, d = env._model, env._data
    count = int(env.th.len_trajectory(0))
    qref = np.asarray(env.th.traj.data.qpos).copy()
    qref[:, :2] -= qref[0, :2]-d.qpos[:2]
    names = [stem+'_r' for stem in STEMS]
    act_ids = np.array([m.actuator_actadr[m.actuator(name).id] for name in names])
    site_ids = [m.site(name).id for name in env.sites_for_mimic]
    shadow = mujoco.MjData(m)
    ref_sites = []
    for q in qref:
        shadow.qpos[:] = q
        mujoco.mj_forward(m, shadow)
        ref_sites.append(shadow.site_xpos[site_ids].copy())
    arrays = {name: [] for name in ['qpos', 'qvel', 'act', 'ctrl', 'activation_right', 'site_xpos', 'frame_index', 'time_s']}
    warning_before = d.warning.number.copy()
    termination, finite = 'reference_end', True
    for step in range(count-1):
        action = np.asarray(policy(obs)).reshape(-1)
        obs, reward, absorbing, done, info = env.step(action)
        d = env._data
        index = int(env._additional_carry.traj_state.subtraj_step_no)
        assert index == step+1
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.act).all() or not np.isfinite(action).all():
            termination, finite = 'nonfinite', False
            break
        row = dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), act=d.act.copy(), ctrl=d.ctrl.copy(),
                   activation_right=d.act[act_ids].copy(), site_xpos=d.site_xpos[site_ids].copy(),
                   frame_index=index, time_s=index*.01)
        for key, value in row.items():
            arrays[key].append(value)
        if absorbing:
            termination = 'task_failure'
            break
        if done and index < count-1:
            termination = 'early_done'
            break
    arrays = {key: np.asarray(values) for key, values in arrays.items()}
    indices = arrays['frame_index'].astype(int)
    arrays['reference_qpos'] = qref[indices]
    arrays['reference_site_xpos'] = np.asarray(ref_sites)[indices]
    warnings = d.warning.number-warning_before
    bad_ids = [int(mujoco.mjtWarning.mjWARN_BADQPOS), int(mujoco.mjtWarning.mjWARN_BADQVEL), int(mujoco.mjtWarning.mjWARN_BADQACC)]
    quality = dict(complete=len(indices) == count-1 and termination == 'reference_end', termination=termination,
                   numerical_warnings=int(warnings[bad_ids].sum()), passed=False)
    if len(indices):
        sites, refs = arrays['site_xpos'], arrays['reference_site_xpos']
        pelvis = env.sites_for_mimic.index('pelvis_mimic')
        relative = (sites-sites[:, pelvis:pelvis+1])-(refs-refs[:, pelvis:pelvis+1])
        relative_error = float(np.linalg.norm(relative, axis=-1).mean())
        root_error = float(np.linalg.norm(arrays['qpos'][:, :3]-arrays['reference_qpos'][:, :3], axis=1).max())
        joints = [stem+'_'+side for side in ['r', 'l'] for stem in
                  ['hip_flexion', 'hip_adduction', 'hip_rotation', 'knee_angle', 'ankle_angle']]
        ids = [m.joint(name).qposadr[0] for name in joints]
        actual, expected = arrays['qpos'][:, ids], arrays['reference_qpos'][:, ids]
        joint_error = float(np.rad2deg(np.abs(actual-expected)).mean())
        rom = {name: float(a/b) for name, a, b in zip(joints, np.ptp(actual, axis=0), np.ptp(expected, axis=0))
               if b >= np.deg2rad(15)}
        rom_passed = all(.7 <= value <= 1.4 for value in rom.values())
        quality.update(mean_relative_site_error_m=relative_error, peak_root_error_m=root_error,
                       leg_joint_mae_deg=joint_error, significant_joint_rom_ratios=rom,
                       passed=quality['complete'] and finite and quality['numerical_warnings'] == 0 and
                       relative_error <= .075 and root_error <= .25 and joint_error <= 20 and rom_passed)
    return arrays, quality, finite


def evaluate(root, arm, checkpoint, label, only_training_clip=False):
    protocol = json.loads((root/'protocol.json').read_text())
    references = Path(protocol['references'])
    config, agent, metadata = load_checkpoint(str(checkpoint))
    entries = json.loads((references/'manifest.json').read_text())['records']
    if only_training_clip:
        entries = [row for row in entries if row['source']['id'] == CLIP]
    folder = root/arm/'evaluations'/label
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in entries:
        clip = entry['source']['id']
        destination = folder/clip
        destination.mkdir(exist_ok=True)
        env = cpu_env(config, references/'clips'/clip/'trajectory.npz')
        conf = PPOJax.init_agent_conf(env, config)
        variables = dict(params=agent.train_state.params, run_stats=agent.train_state.run_stats)
        policy = jax.jit(lambda obs: conf.network.apply(variables, jnp.atleast_2d(obs))[0].mode())
        arrays, quality, finite = rollout(env, policy)
        source = np.load(references/'clips'/clip/'emg_reference.npz', allow_pickle=False)
        expected = source['right_measured'][arrays['frame_index'].astype(int)]
        mask = arrays['time_s'] >= .10
        metrics = correlations(arrays['activation_right'][mask], expected[mask]) if len(expected) else None
        np.savez_compressed(destination/'rollout.npz', **arrays, measured_right_emg=expected,
                            right_group_names=[name+'_r' for name in STEMS])
        result = dict(clip=clip, split=entry['split'], used_as_ppo_reference=clip == CLIP,
                      steps=len(expected), finite_rollout=finite, quality=quality, primary=metrics,
                      eligible_for_all_six_ranking=bool(quality['passed'] and metrics and metrics['valid_group_count'] == 6))
        save_json(destination/'result.json', result)
        if len(expected):
            render_video(destination, env, arrays)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(3, 2, figsize=(12, 8), layout='constrained')
            for i, ax in enumerate(axes.flat):
                ax.plot(arrays['time_s'], expected[:, i], label='Measured EMG')
                ax.plot(arrays['time_s'], arrays['activation_right'][:, i], label='Actual muscle activation')
                ax.set(title=STEMS[i], xlabel='Time (s)', ylabel='MVC / activation')
                ax.grid(alpha=.2)
            axes[0, 0].legend()
            fig.suptitle(f'{arm} / {clip} / quality_pass={quality["passed"]}')
            fig.savefig(destination/'activation_vs_emg.png', dpi=130)
            plt.close(fig)
        results.append(result)
        env.stop()
    summary = dict(arm=arm, checkpoint=str(checkpoint), label=label, clips=results,
        left_measured=False, primary_startup_exclusion_s=.1,
        ranking_note='Rank only quality-passing runs on the same clip; common valid groups must be reported explicitly.')
    save_json(folder/'summary.json', summary)
    cards = ''.join(f'<h2>{row["clip"]}</h2><p>运动质量通过：{row["quality"]["passed"]}；'
                    f'终止原因：{row["quality"]["termination"]}</p><video controls src="{row["clip"]}/motion.mp4"></video>'
                    f'<img src="{row["clip"]}/activation_vs_emg.png">' for row in results)
    (folder/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<style>body{max-width:1200px;margin:auto;font:16px/1.6 sans-serif}video,img{width:100%}</style>'
        f'<h1>{arm} / {label}</h1><p>参考与实际策略运动对照；视频为 0.25 倍速。左侧 EMG 未实测，不纳入相关性。</p>'
        '<p>提前失败与未通过运动质量的结果仅供诊断。</p><a href="summary.json">全部指标</a>'+cards+'</html>')
    print('EVALUATION_COMPLETE', arm, label, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=['ppo', 'synergy_reward'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--label', default='final')
    parser.add_argument('--only-training-clip', action='store_true')
    args = parser.parse_args()
    evaluate(args.root.resolve(), args.arm, args.checkpoint, args.label, args.only_training_clip)
