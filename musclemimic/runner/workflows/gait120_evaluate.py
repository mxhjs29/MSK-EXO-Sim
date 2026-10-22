"""Frozen-checkpoint, real MuJoCo EMG evaluation and automatic comparison.

Primary statistic: unshifted Pearson r averaged over 11 anatomical groups,
after a common 100 ms startup, provided the complete motion passes quality
checks. EMG is never used to select checkpoints or controller parameters.
"""
import argparse
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from omegaconf import OmegaConf, open_dict

from musclemimic.algorithms import PPOJax
from musclemimic.runner.eval_utils import load_checkpoint
from musclemimic.runner.engine import instantiate_env
from musclemimic.retargeting.gait120_emg import CHANNELS, MUSCLES
from musclemimic.runner.workflows.gait120_recruitment import FrozenRecruitment

ARMS = ("ppo", "recruitment_reward", "frozen_recruitment")
GROUPS = [(c, [i]) for i, c in enumerate(CHANNELS) if c not in ("SM", "SL")]
GROUPS.insert(8, ("SOL", [8, 9]))


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 10 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def correlation_metrics(activation, times, emg):
    expected = np.column_stack([np.interp(times, emg["source_time_s"], emg["normalized"][:, i])
                                for i in range(12)])
    per_sensor = {c: pearson(activation[:, i], expected[:, i]) for i, c in enumerate(CHANNELS)}
    per_group = {name: pearson(activation[:, indices].mean(axis=1), expected[:, indices].mean(axis=1))
                 for name, indices in GROUPS}
    values = list(per_group.values())
    complete = all(v is not None for v in values)
    return dict(per_sensor_r=per_sensor, per_group_r=per_group, valid_groups=sum(v is not None for v in values),
                mean_r=float(np.mean(values)) if complete else None,
                median_r=float(np.median(values)) if complete else None,
                constant_or_invalid_groups=[k for k, v in per_group.items() if v is None]), expected


def make_env_and_policy(checkpoint):
    config, agent, metadata = load_checkpoint(str(checkpoint))
    cpu = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    with open_dict(cpu):
        params = cpu.experiment.env_params
        params.env_name = "MyoFullBodyAssistEnv"
        params.pop("num_envs", None)
        params.pop("mjx_backend", None)
        # Match Warp's model options; no CPU-default solver differences.
        params.model_option_conf = dict(iterations=4, ls_iterations=8,
                                       disableflags=int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP))
        params.th_params = dict(random_start=False, fixed_start_conf=[0, 0], start_from_random_step=False)
        params.headless = True
    env = instantiate_env(cpu)
    network = PPOJax.init_agent_conf(env, config).network
    variables = dict(params=agent.train_state.params, run_stats=agent.train_state.run_stats)
    policy = jax.jit(lambda obs: network.apply(variables, jnp.atleast_2d(obs))[0].mode())
    return env, policy, metadata


def rollout(env, policy, recruitment=False):
    obs = env.reset(jax.random.PRNGKey(42))
    m, d = env._model, env._data
    reference = env.th.traj.data
    n = int(env.th.len_trajectory(0))
    start_qpos = d.qpos.copy()
    ref_qpos = np.asarray(reference.qpos).copy()
    # Trajectory reset removes initial world XY. Apply exactly that translation.
    offset = ref_qpos[0, :2] - start_qpos[:2]
    ref_qpos[:, :2] -= offset
    muscle_ids = np.array([m.actuator(name).id for name in MUSCLES])
    act_ids = m.actuator_actadr[muscle_ids]
    site_ids = np.array([m.site(name).id for name in env.body2sites_for_mimic.values()])
    reference_sites = []
    shadow = mujoco.MjData(m)
    for q in ref_qpos:
        shadow.qpos[:] = q
        mujoco.mj_forward(m, shadow)
        reference_sites.append(shadow.site_xpos[site_ids].copy())
    reference_sites = np.asarray(reference_sites)
    controller = FrozenRecruitment(env) if recruitment else None
    original_compute = env._compute_action
    if controller:
        def compute(action, model, data, carry):
            ctrl, carry = original_compute(action, model, data, carry)
            return controller.transform(ctrl), carry
        env._compute_action = compute
    arrays = {name: [] for name in ("time_s", "qpos", "qvel", "act", "ctrl", "actuator_force",
                                    "emg_activation", "site_xpos", "reward", "source_frame_index")}
    termination = "reference_end"
    error = None
    warnings_before = np.array(d.warning.number)
    try:
        for step in range(n - 1):
            actor_obs = controller.actor_observation(obs) if controller else obs
            action = np.asarray(policy(actor_obs)).reshape(-1)
            if not np.isfinite(action).all():
                raise FloatingPointError("Nonfinite policy command")
            obs, reward, absorbing, done, info = env.step(action)
            d = env._data
            actual_index = int(env._additional_carry.traj_state.subtraj_step_no)
            # Reference state must advance one frame per physical control step.
            if actual_index != step + 1:
                raise ValueError(f"Trajectory/physics time mismatch: {actual_index} != {step + 1}")
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.act).all():
                raise FloatingPointError("Nonfinite physical state")
            values = dict(time_s=(step + 1) * env.dt, qpos=d.qpos.copy(), qvel=d.qvel.copy(),
                          act=d.act.copy(), ctrl=d.ctrl.copy(), actuator_force=d.actuator_force.copy(),
                          emg_activation=d.act[act_ids].copy(), site_xpos=d.site_xpos[site_ids].copy(),
                          reward=float(reward), source_frame_index=actual_index)
            for name, value in values.items():
                arrays[name].append(value)
            if absorbing:
                termination = "task_failure"
                break
            if done and step + 1 < n - 1:
                termination = "early_done"
                break
    except (FloatingPointError, mujoco.FatalError, ValueError, RuntimeError) as exc:
        error = str(exc)
        termination = "evaluation_error"
    finally:
        env._compute_action = original_compute
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    indices = arrays["source_frame_index"].astype(int)
    arrays["reference_qpos"] = ref_qpos[indices]
    arrays["reference_site_xpos"] = reference_sites[indices]
    arrays["muscle_names"] = np.array([m.actuator(i).name for i in range(m.nu)])
    arrays["actuator_actadr"] = m.actuator_actadr.copy()
    warnings = np.array(d.warning.number) - warnings_before
    bad_ids = [int(mujoco.mjtWarning.mjWARN_BADQPOS), int(mujoco.mjtWarning.mjWARN_BADQVEL),
               int(mujoco.mjtWarning.mjWARN_BADQACC)]
    quality = dict(termination=termination, error=error, expected_steps=n - 1, steps=len(indices),
                   complete=len(indices) == n - 1 and termination == "reference_end",
                   numerical_warning_count=int(warnings[bad_ids].sum()), warning_counts=warnings.tolist())
    if len(indices):
        site, ref = arrays["site_xpos"], arrays["reference_site_xpos"]
        root_id = list(env.body2sites_for_mimic.values()).index("pelvis_mimic")
        relative = (site - site[:, root_id:root_id+1]) - (ref - ref[:, root_id:root_id+1])
        mean_relative = float(np.linalg.norm(relative, axis=-1).mean())
        root_max = float(np.linalg.norm(arrays["qpos"][:, :3] - arrays["reference_qpos"][:, :3], axis=-1).max())
        joints = [f"{name}_{side}" for side in ("r", "l")
                  for name in ("hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle")]
        qids = np.array([m.jnt_qposadr[m.joint(j).id] for j in joints])
        joint_mean = float(np.rad2deg(np.abs(arrays["qpos"][:, qids] - arrays["reference_qpos"][:, qids])).mean())
        reference_rom = np.ptp(arrays["reference_qpos"][:, qids], axis=0)
        actual_rom = np.ptp(arrays["qpos"][:, qids], axis=0)
        rom_ratios = {name:float(actual / expected) for name, actual, expected in zip(joints, actual_rom, reference_rom)
                      if expected >= np.deg2rad(15)}
        rom_passed = all(.7 <= ratio <= 1.4 for ratio in rom_ratios.values())
        quality.update(mean_relative_site_error_m=mean_relative, max_root_error_m=root_max,
                       mean_leg_joint_error_deg=joint_mean, joint_rom_ratios=rom_ratios,
                       joint_rom_passed=rom_passed)
        quality["passed"] = bool(quality["complete"] and quality["numerical_warning_count"] == 0
                                 and mean_relative <= .075 and root_max <= .25 and joint_mean <= 20 and rom_passed)
    else:
        quality["passed"] = False
    return arrays, quality, controller.records if controller else []


def plot_result(destination, arrays, emg_expected, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(13, 10), sharex=True)
    for i, (axis, channel) in enumerate(zip(axes.flat, CHANNELS)):
        actual, expected = arrays["emg_activation"][:, i], emg_expected[:, i]
        # Unit-peak display only; numerical Pearson uses original values.
        axis.plot(arrays["time_s"], actual / max(actual.max(), 1e-8), label="simulated activation")
        axis.plot(arrays["time_s"], expected / max(expected.max(), 1e-8), label="MVC-normalized EMG")
        r = metrics["primary"]["per_sensor_r"][channel]
        axis.set_title(f"{channel}: r={r:.3f}" if r is not None else f"{channel}: undefined")
        axis.axvline(.1, color="gray", linestyle=":")
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("simulation time (s)")
    fig.suptitle("Actual activation vs synchronized EMG; peak scaling for display only")
    fig.tight_layout()
    fig.savefig(destination / "activation_emg.png", dpi=160)
    plt.close(fig)


def render_video(destination, env, arrays):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    m = env._model
    renderer = mujoco.Renderer(m, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.2, 125, -12
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
    d = mujoco.MjData(m)
    writer = imageio.get_writer(destination / "motion.mp4", fps=25, codec="libx264", quality=7)
    try:
        # Fourfold slow playback for this single stride; labels retain physical time.
        for i in range(len(arrays["qpos"])):
            views = []
            for title, q in (("reference", arrays["reference_qpos"][i]), ("simulated", arrays["qpos"][i])):
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                camera.lookat[:] = [*q[:2], .9]
                renderer.update_scene(d, camera=camera, scene_option=option)
                frame = Image.fromarray(renderer.render().copy())
                ImageDraw.Draw(frame).text((12, 12), f"{title} | t={arrays['time_s'][i]:.2f}s | 0.25x", fill="black")
                views.append(np.asarray(frame))
            writer.append_data(np.concatenate(views, axis=1))
    finally:
        writer.close()
        renderer.close()


def evaluate(root, arm, checkpoint, smoke=False):
    root = Path(root)
    destination = root / ("evaluation_smoke" if smoke else arm) / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    env, policy, metadata = make_env_and_policy(checkpoint)
    emg = np.load(root / "data/emg_reference.npz")
    # First retained retargeted frame is TRC frame 486, t=4.85 s.
    source_origin = float(emg["source_time_s"][0]) + env.dt
    arrays, quality, controller_records = rollout(env, policy, arm == "frozen_recruitment")
    arrays["source_time_s"] = source_origin + arrays["time_s"]
    if len(arrays["time_s"]) < 10:
        raise RuntimeError(f"Insufficient valid rollout: {quality}")
    all_metrics, expected = correlation_metrics(arrays["emg_activation"], arrays["source_time_s"], emg)
    mask = arrays["time_s"] > .1 + 1e-8
    primary, _ = correlation_metrics(arrays["emg_activation"][mask], arrays["source_time_s"][mask], emg)
    # Prespecified secondary lag; never maximize over lags or use it to rank.
    lag50, _ = correlation_metrics(arrays["emg_activation"][mask], arrays["source_time_s"][mask] - .05, emg)
    result = dict(arm=arm, checkpoint=str(checkpoint), update_number=int(metadata.update_number),
                  evaluated_at=datetime.now(timezone.utc).isoformat(), primary=primary,
                  secondary_full_stride=all_metrics, secondary_emg_leads_50ms=lag50,
                  quality=quality, eligible_for_ranking=bool(quality["passed"] and primary["valid_groups"] == 11),
                  frozen_policy=True, frozen_observation_normalization=True, emg_used_for_training=False,
                  source_origin_s=source_origin, group_count=11,
                  recruitment=dict(enabled=arm == "frozen_recruitment", strength=30, warmup_s=.1,
                                   selected_muscles=env._model.nu, exoskeleton_torque=0))
    if arm == "frozen_recruitment":
        baseline, baseline_quality, _ = rollout(env, policy, False)
        baseline_times = source_origin + baseline["time_s"]
        baseline_mask = baseline["time_s"] > .1 + 1e-8
        baseline_r, _ = correlation_metrics(baseline["emg_activation"][baseline_mask], baseline_times[baseline_mask], emg)
        result["same_checkpoint_without_recruitment"] = dict(primary=baseline_r, quality=baseline_quality)
        np.savez_compressed(destination / "without_recruitment.npz", **baseline)
    np.savez_compressed(destination / "rollout.npz", **arrays, expected_emg=expected)
    write_json(destination / "recruitment_diagnostics.json", controller_records)
    write_json(destination / "metrics.json", result)
    plot_result(destination, arrays, expected, result)
    render_video(destination, env, arrays)
    if not smoke:
        summarize(root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def summarize(root):
    root = Path(root)
    with (root / "comparison.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = {a:json.loads((root / a / "evaluation/metrics.json").read_text()) for a in ARMS
                   if (root / a / "evaluation/metrics.json").exists()}
        finished = len(results) == 3 and all(r["update_number"] == 10000 for r in results.values())
        protocol_path = root / "protocol.json"
        protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
        shared_required = protocol.get("arm3", {}).get("source_arm") == "ppo"
        paired_verified = None
        if shared_required and "ppo" in results and "frozen_recruitment" in results:
            original = results["ppo"].get("checkpoint")
            recruited = results["frozen_recruitment"].get("checkpoint")
            paired_verified = bool(original and recruited and Path(original).resolve() == Path(recruited).resolve())
        if shared_required:
            finished = finished and paired_verified is True
        eligible = {a:r for a, r in results.items() if r["eligible_for_ranking"]}
        winner = max(eligible, key=lambda a: eligible[a]["primary"]["mean_r"]) if finished and eligible else None
        report = dict(status="complete" if finished else "training_or_evaluation_pending", results=results,
                      winner_among_quality_eligible_methods=winner,
                      all_methods_quality_eligible=finished and len(eligible) == 3,
                      shared_checkpoint_required=shared_required, shared_checkpoint_verified=paired_verified,
                      limitation="Single subject, one trained stride, one seed; no population or held-out-motion claim.")
        write_json(root / "comparison.json", report)
        rows = []
        for arm in ARMS:
            if arm not in results:
                rows.append(f"| {arm} | 等待训练/评估 | — | — |")
                continue
            r = results[arm];score = r["primary"]["mean_r"]
            rows.append(f"| {arm} | {r['update_number']} | {score if score is not None else '未定义'} | {r['quality']['passed']} |")
        text = "# Gait120 三方法 EMG 对比\n\n状态：" + report["status"] + "\n\n"
        text += "| 方法 | PPO 更新数 | 11 肌群平均 Pearson r | 运动质量通过 |\n| --- | --- | --- | --- |\n" + "\n".join(rows)
        text += "\n\n主指标使用原时间对齐、零时移，统一排除起始 0.1 秒；比目鱼肌两电极合并为一个解剖肌群。常量信号不计为零相关，也不静默剔除。\n"
        text += f"\n通过运动质量要求的方法中，当前最高相关性方法：{winner or '尚不能判定'}。\n"
        text += "\n本实验仅比较同一训练步态上的肌肉模式，不代表新受试者或新轨迹的泛化。\n"
        (root / "comparison.md").write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    evaluate(args.root, args.arm, args.checkpoint, args.smoke)
