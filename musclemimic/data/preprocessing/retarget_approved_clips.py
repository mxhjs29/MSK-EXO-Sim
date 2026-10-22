"""Retarget reviewed task clips at 100 Hz into an isolated, verifiable delivery."""
import argparse
import collections
import csv
import hashlib
import json
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from musclemimic.data.preprocessing import retarget_selected_myofullbody as engine
from musclemimic.data.preprocessing.dataset_paths import OUTPUT_ROOT, DATASET, SCRIPTS, CONFIG

ROOT = engine.ROOT
AUDIT = DATASET / 'dataset_audit'
OLD = DATASET / 'converted_dataset/approved_continuous_100hz/full/new'
STAGE = OUTPUT_ROOT / 'approved_100hz/converted_datasets/myofullbody_no_fingers/gmr'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(name, value):
    (STAGE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def init(output):
    engine.OUT = Path(output)
    engine.init_worker()

def run(row):
    return engine.convert(row['trajectory_path'], AUDIT / row['clip_path'], 100., row['source_start_frame'])

def selection():
    import numpy as np
    reviews = json.loads((AUDIT / 'motion_reviews.json').read_text())['reviews']
    candidates = {r['id']: r for r in map(json.loads, (AUDIT / 'selected.jsonl').read_text().splitlines())}
    rows = []
    for key, review in reviews.items():
        if review['status'] != 'pass':
            continue
        row = dict(candidates[key])
        for field in ('source_path', 'source_start_frame', 'source_end_frame_exclusive'):
            assert review[field] == row[field], (key, field)
        row['task'] = review.get('corrected_task') or row['task']
        row['subtype'] = review.get('corrected_subtype') or row['subtype']
        row['human_review'] = review
        row['trajectory_path'] = f"{row['task']}/{key}.npz"
        clip = AUDIT / row['clip_path']
        assert sha(clip) == row['clip_file_sha256'], key
        with np.load(clip, allow_pickle=False) as z, np.load(DATASET / row['source_path'], allow_pickle=False) as src:
            lo, hi = row['source_start_frame'], row['source_end_frame_exclusive']
            assert len(z['poses']) == hi - lo == row['clip_frames']
            assert np.array_equal(z['poses'], src['poses'][lo:hi])
            assert np.array_equal(z['trans'], src['trans'][lo:hi])
            assert float(z['mocap_framerate']) == row['fps']
            assert len(z['poses']) >= 2
        rows.append(row)
    return sorted(rows, key=lambda r: r['id'])

def verify(rows, frequency=100):
    import numpy as np
    import mujoco
    model = mujoco.MjModel.from_binary_path(str(STAGE / 'model.mjb'))
    data = mujoco.MjData(model)
    results, tasks = [], []
    for row in rows:
        path = STAGE / row['trajectory_path']
        result = json.loads(path.with_suffix('.json').read_text())
        assert result['status'] == 'converted'
        with np.load(path, allow_pickle=True) as z:
            q, v = z['qpos'], z['qvel']
            n = len(q)
            expected = row.get('sample_count', int(np.floor((row['clip_frames'] - 1) / row['fps'] * frequency + 1e-8)) + 1)
            assert n == expected and q.shape == (n, 89) and v.shape == (n, 88)
            assert float(z['frequency']) == frequency and list(z['split_points']) == [0, n]
            assert np.isfinite(q).all() and np.isfinite(v).all()
            assert np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-6)
            assert z['site_xpos'].shape == (n, 17, 3) and z['site_xmat'].shape == (n, 17, 9)
            assert np.isfinite(z['site_xpos']).all() and np.isfinite(z['site_xmat']).all()
            dv = np.empty((n - 1, 88))
            for i in range(n - 1):
                mujoco.mj_differentiatePos(model, dv[i], 1 / frequency, q[i], q[i+1])
            expected_v = np.vstack([dv[0], (dv[:-1] + dv[1:]) / 2, dv[-1]])
            assert np.allclose(v, expected_v, atol=1e-8)
            ids = [model.site(str(name)).id for name in z['site_names']]
            for i in (0, n // 2, n - 1):
                data.qpos[:] = q[i]
                mujoco.mj_kinematics(model, data)
                assert np.allclose(z['site_xpos'][i], data.site_xpos[ids], atol=1e-6)
        with np.load(path.with_name(path.stem + '_analysis.npz')) as a:
            times = a['source_times_s']
            assert len(times) == n and np.allclose(np.diff(times), 1 / frequency)
            assert abs(times[0] - row.get('sample_start_time_s', row['source_start_frame'] / row['fps'])) < 1e-9
            assert times[-1] <= (row['source_end_frame_exclusive'] - 1) / row['fps'] + 1e-8
            assert np.allclose(a['source_frame_coordinates'], times * row['fps'])
            assert np.isfinite(a['position_errors_m']).all()
        result['quality_flag'] = 'within_position_thresholds' if result['position_mean_m'] <= .10 and result['position_p95_m'] <= .20 else 'review_position_error'
        result['id'] = row['id']
        results.append(result)
        tasks.append({**row, 'retargeted': True, 'review_status': ('human_pass' if row.get('review_status') in ('pass','human_pass') else 'not_human_reviewed'), 'frequency': float(frequency), 'trajectory_frames': n,
                      'action': {'start_frame': 0, 'end_frame_exclusive': n},
                      'quality': result, 'dynamics_verified': False})
    actual = {str(p.relative_to(STAGE)) for p in STAGE.rglob('*.npz') if not p.stem.endswith('_analysis')}
    assert actual == {r['trajectory_path'] for r in rows}
    summary = {'clips': len(rows), 'unique_sources': len({r['source_path'] for r in rows}),
               'frequency': frequency, 'nq': 89, 'nv': 88, 'total_frames': sum(r['frames'] for r in results),
               'task_counts': dict(collections.Counter(r['task'] for r in rows)),
               'quality_counts': dict(collections.Counter(r['quality_flag'] for r in results)),
               'max_hinge_limit_violation_rad': max(r['max_hinge_limit_violation_rad'] for r in results),
               'max_slide_limit_violation_m': max(r['max_slide_limit_violation_m'] for r in results),
               'dynamics_verified': False, 'problems': []}
    dump('verification_summary.json', summary)
    dump('conversion_results.json', results)
    (STAGE / 'tasks.jsonl').write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in tasks))
    fields = ['id', 'task', 'subtype', 'trajectory_path', 'source_path', 'source_start_frame', 'source_end_frame_exclusive', 'fps', 'clip_frames', 'trajectory_frames', 'frequency']
    with (STAGE / f'task_index_{frequency:g}hz.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(tasks)
    with (STAGE / 'needs_review.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader()
        w.writerows(r for r in results if r['quality_flag'] != 'within_position_thresholds')
    (STAGE / 'README.md').write_text(f'''# 人工通过任务片段 → MyoFullBody（无手指）

{len(rows)} 条人工通过片段，{frequency:g} Hz，{summary['total_frames']} 帧，qpos=89、qvel=88。
仅转换选定原始帧区间；采用人工修正后的类别和动作内容。

从 `tasks.jsonl` 或 `task_index_{frequency:g}hz.csv` 读取清单，trajectory_path 相对于本目录。
`tasks.jsonl` 保留源文件、原始帧区间、人工记录、BABEL 语义证据及重映射质量。
BABEL 证据保留原始标注时间，可能超出当前片段，不代表整个片段的精确标签。
原始帧区间左闭右开；输出从片段首帧起每 {1 / frequency:.6g} 秒采样，截止于最后一个可插值的原始帧，不外推、不延长动作。
源帧率低于 {frequency:g} Hz 时插值不增加原始观测信息。SMPL 旋转使用 SLERP、平移使用线性插值，每个 {frequency:g} Hz 帧均求解 GMR。
速度由 MuJoCo mj_differentiatePos 按 {1 / frequency:.6g} 秒计算，内部帧取相邻差分均值；每条片段独立初始化。

输出是项目原生 Trajectory NPZ，含 qpos/qvel/site_xpos/site_xmat/frequency/split_points。
加载示例：`Trajectory.load(str(path), backend=numpy)`（`from loco_mujoco.trajectory import Trajectory`）。
模型为 model.mjb，关节定义为 joint_schema.json；保留手腕、不映射手指。
`*_analysis.npz` 提供原始时间坐标、拟合误差与地面修正；拟合误差是 GMR 求解后、额外地面穿透修正前的指标。
`verification_summary.json` 是完整性验证结果；needs_review.csv 列出平均位置误差 >0.10m 或 P95 >0.20m 的片段。
人工动作验收不等同于重映射后的动力学验收；尚未验证动力学，也不包含肌肉激活或物体接触监督。
训练划分请按源文件/受试者分组，避免同源片段泄漏。
''')
    import importlib.metadata
    dump('environment.json', {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'mujoco', 'torch', 'smplx', 'mink')})
    provenance = STAGE / 'reproduction'
    provenance.mkdir(exist_ok=True)
    for source in (Path(__file__), SCRIPTS / 'dataset_paths.py', SCRIPTS / 'retarget_selected_myofullbody.py', CONFIG):
        shutil.copy2(source, provenance / source.name)
    dump('delivery_manifest.json', {str(p.relative_to(STAGE)): sha(p) for p in sorted(STAGE.rglob('*')) if p.is_file() and p.name != 'delivery_manifest.json'})
    print(json.dumps(summary, ensure_ascii=False), flush=True)

def main():
    global STAGE
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=STAGE); p.add_argument('--workers', type=int, default=3); p.add_argument('--limit', type=int); p.add_argument('--verify-only', action='store_true'); args = p.parse_args()
    STAGE = args.output.resolve()
    rows = selection()
    STAGE.mkdir(parents=True, exist_ok=True)
    snapshot = {'selected_sha256': sha(AUDIT / 'selected.jsonl'), 'reviews_sha256': sha(AUDIT / 'motion_reviews.json'), 'frequency': 100, 'scope': 'human-approved clips only', 'clips': len(rows)}
    manifest = STAGE / 'run_manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == snapshot, 'Selection changed; use a new staging directory.'
    else:
        dump('run_manifest.json', snapshot)
        for name in ('model.mjb', 'model_metadata.json', 'joint_schema.json'):
            shutil.copy2(OLD / name, STAGE / name)
        shutil.copy2(AUDIT / 'motion_reviews.json', STAGE / 'motion_reviews_snapshot.json')
        (STAGE / 'selection_snapshot.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    if args.verify_only:
        verify(rows); return
    pending = []
    for row in rows:
        path = STAGE / row['trajectory_path']
        sidecar = path.with_suffix('.json')
        if sidecar.exists() and path.exists() and path.with_name(path.stem + '_analysis.npz').exists():
            if json.loads(sidecar.read_text()).get('frequency') == 100:
                continue
        pending.append(row)
    if args.limit:
        pending = pending[:args.limit]
    errors = []
    print(f'Approved {len(rows)}; pending this run {len(pending)}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(str(STAGE),)) as pool:
        futures = {pool.submit(run, r): r['id'] for r in pending}
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(count, len(pending), json.dumps(result, ensure_ascii=False), flush=True)
            if result['status'] != 'converted': errors.append(result)
    if errors:
        dump('errors.json', errors); raise SystemExit(1)
    if not args.limit:
        verify(rows)

if __name__ == '__main__':
    main()
