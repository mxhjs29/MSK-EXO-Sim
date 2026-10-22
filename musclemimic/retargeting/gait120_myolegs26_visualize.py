"""Software-rendered review artifacts for offline MyoLegs26 references."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np


def render(folder, model):
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from loco_mujoco.trajectory import Trajectory

    traj = Trajectory.load(folder/'trajectory.npz', backend=np)
    q = np.asarray(traj.data.qpos)
    data = mujoco.MjData(model)
    comparison = dict(np.load(folder/'comparison.npz'))
    labels = list(comparison['marker_labels'])
    markers = comparison['markers_m']
    pelvis = [labels.index(n) for n in ['LASI', 'RASI', 'LPSI', 'RPSI']]
    edges = []
    for chain in [['LASI', 'RASI', 'RPSI', 'LPSI', 'LASI'],
                  ['LASI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 'LTOE', 'LHEE', 'LANK'],
                  ['RASI', 'RTHI', 'RKNE', 'RTIB', 'RANK', 'RTOE', 'RHEE', 'RANK']]:
        edges.extend((labels.index(a), labels.index(b)) for a, b in zip(chain, chain[1:]))
    selected = sorted(set(i for edge in edges for i in edge))
    fig = plt.figure(figsize=(4, 4.64), dpi=100, facecolor='#f5f8fc')
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlim=(-.7, .7), ylim=(-.7, .7), zlim=(0, 1.35), xlabel='X (m)', ylabel='Y (m)')
    ax.set_box_aspect((1.4, 1.4, 1.35))
    ax.view_init(elev=12, azim=-60)
    ax.tick_params(labelsize=7)
    lines = [ax.plot([], [], [], lw=1.5, color='#4b718c')[0] for _ in edges]
    dots = ax.scatter([], [], [], s=17, c='#1670b9')
    fig.subplots_adjust(left=0, right=1, top=.95, bottom=.07)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 400)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 464)
    renderer = mujoco.Renderer(model, width=400, height=464)
    camera = mujoco.MjvCamera()
    camera.distance = 2.8
    camera.elevation = -8
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    frames = list(range(0, len(q), 4))
    try:
        with imageio.get_writer(folder/'reference.mp4', fps=25, codec='libx264', quality=8,
                               ffmpeg_params=['-threads', '1'], macro_block_size=16, ffmpeg_log_level='error') as writer:
            for index, i in enumerate(frames):
                center = markers[i, pelvis].mean(axis=0)
                center[2] = 0
                points = markers[i]-center
                for line, (a, b) in zip(lines, edges):
                    line.set_data_3d(points[[a, b], 0], points[[a, b], 1], points[[a, b], 2])
                dots._offsets3d = tuple(points[selected].T)
                fig.canvas.draw()
                panels = [np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()]
                data.qpos[:] = q[i]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.xpos[model.body('pelvis').id]
                camera.lookat[2] = .9
                for azimuth in [90, 145]:
                    camera.azimuth = azimuth
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    panels.append(renderer.render().copy())
                frame = Image.new('RGB', (1200, 528), '#f5f8fc')
                draw = ImageDraw.Draw(frame)
                draw.text((12, 7), f"{folder.name} | source t={comparison['source_time_s'][i]:.2f}s", font=font, fill='#19354c')
                draw.text((12, 33), 'Recorded lower-body markers', font=small, fill='#19354c')
                draw.text((410, 33), 'MyoLegs26 reference | view A', font=small, fill='#19354c')
                draw.text((810, 33), 'View B | kinematic replay, no policy', font=small, fill='#19354c')
                for j, panel in enumerate(panels):
                    frame.paste(Image.fromarray(panel), (j*400, 64))
                writer.append_data(np.asarray(frame))
                if index == len(frames)//2:
                    frame.save(folder/'preview.png')
    finally:
        renderer.close()
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout='constrained')
    time = comparison['source_time_s']-comparison['source_time_s'][0]
    for row, side in enumerate(['r', 'l']):
        for col, stem in enumerate(['hip_flexion', 'knee_angle', 'ankle_angle']):
            values = np.rad2deg(q[:, model.joint(stem+'_'+side).qposadr[0]])
            axes[row, col].plot(time, values)
            axes[row, col].set(title=stem+'_'+side, xlabel='Time (s)', ylabel='Model angle (deg)')
            axes[row, col].grid(alpha=.2)
    fig.suptitle('MyoLegs26 coordinates: knee flexion is negative; bilateral motion comes from mocap')
    fig.savefig(folder/'joint_angles.png', dpi=140)
    plt.close(fig)
    print('RENDERED', folder.name, flush=True)


def report(output):
    result = json.loads((output/'manifest.json').read_text())
    synergy = json.loads((output/'synergy/results.json').read_text())
    records = result['records']
    rows = ''.join(f"<tr><td>{html.escape(r['source']['id'])}</td><td>{r['split']}</td>"
                   f"<td>{r['metrics']['frames']}</td><td>{1000*r['metrics']['transfer_body_error_mean_m']:.2f}</td>"
                   f"<td>{r['metrics']['equality_residual_max']:.2e}</td></tr>" for r in records)
    options = ''.join(f'<option value="{r["source"]["id"]}">{r["source"]["id"]}</option>' for r in records)
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gait120 → MyoLegs26：肌肉协同与参考轨迹</title><style>
body{max-width:1200px;margin:auto;padding:24px;background:#f5f8fc;color:#19354c;font:16px/1.65 system-ui,sans-serif}video,img{width:100%;background:white;border-radius:8px}select,button{font:inherit;padding:7px;margin:6px}td,th{padding:7px;border:1px solid #c9d8e5}table{border-collapse:collapse}a{color:#1263a0}.note{padding:15px;background:#e4eff8;border-left:4px solid #1670b9}
</style><h1>Gait120 → MyoLegs26</h1>
<p>S101 六段平地行走；每侧六个 EMG 肌群，选择 K=RANK 个协同。右侧来自实测 EMG，左侧采用相同权重与半周期错相。</p>
<p class="note">本页展示运动学参考回放，不是肌肉策略训练结果。左右动作均来自实际双侧动捕；仅左侧 EMG 先验采用对称假设。肌肉线条显示几何，不表示激活强弱。</p>
<label>片段 <select id="clip">OPTIONS</select></label><button id="slow">播放速度 1×</button><a id="download">轨迹 NPZ</a>
<video id="video" controls loop muted playsinline></video><p>左：原始下肢标记；中、右：同一时刻的 MyoLegs26。各视图独立居中，轨迹保留根平移。轨迹为 100 Hz，视频为 25 fps。</p>
<img id="angles" alt="关节角度">
<h2>肌肉协同</h2><img src="synergy/weights_and_rank.png" alt="协同权重与选择"><img src="synergy/bilateral_phase_targets.png" alt="左右侧先验">
<p>W 左右相同，但左右系数为独立块；左侧参考系数按右侧半周期错相构造。循环插值仅对代理曲线端点取平均，右侧测量和拟合系数保留。仍有每侧七个执行器未获得目标，不能把缺测解释为零激活。</p>
<h2>重映射检查</h2><table><tr><th>片段</th><th>数据划分</th><th>帧数</th><th>骨段位置转移平均误差 mm</th><th>耦合残差</th></tr>ROWS</table>
<p>表中误差相对于已有完整人体拟合参考的对应骨段，不是直接对原始皮肤标记的误差。原拟合误差另存于 manifest.json。短片段保持独立，不用于声称已完成连续步行。</p>
<p class="note">运动学合格不等于动态步行合格：本次未强制站立期无滑动接触。逐帧足部碰撞网格与地面仍有正间隙，低位足部存在水平运动；尚未验证肌肉驱动后的接触、稳定性和持续步行。</p>
<p class="note">参考模型副本修正了依赖关节限位，使之包含原耦合多项式在主关节范围内的取值。主关节范围、耦合公式及肌肉参数保持原值。后续加载参考需要使用导出的配套模型；本次未修改 PPO 或已安装模型资产。</p>
<p><a href="model/range_corrections.json">限位修正明细</a> · <a href="model/myolegs26_reference.xml">配套模型 XML</a> · <a href="model/myolegs26_reference.mjb">精确编译模型 MJB</a> · <a href="manifest.json">完整指标</a> · <a href="completion_audit.json">最终检查记录</a> · <a href="levelwalking_train.npz">提取/训练集参考</a> · <a href="synergy/training_prior.npz">仅训练片段协同先验</a> · <a href="synergy/per_group_reconstruction.csv">逐组重建指标</a></p>
<script>const sel=document.getElementById('clip'),v=document.getElementById('video');function show(){const base='clips/'+sel.value+'/';v.src=base+'reference.mp4';v.poster=base+'preview.png';document.getElementById('angles').src=base+'joint_angles.png';document.getElementById('download').href=base+'trajectory.npz';v.playbackRate=1;}sel.onchange=show;document.getElementById('slow').onclick=()=>{v.playbackRate=v.playbackRate===1?.25:1;document.getElementById('slow').textContent='播放速度 '+v.playbackRate+'×';};show();</script></html>'''
    document = document.replace('RANK', str(synergy['selected_rank_per_leg'])).replace('OPTIONS', options).replace('ROWS', rows)
    (output/'index.html').write_text(document, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/gait120_myolegs26_synergy_v1'))
    parser.add_argument('--clip')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if not args.report_only:
        import mujoco
        model = mujoco.MjModel.from_binary_path(str(args.output/'model/myolegs26_reference.mjb'))
        manifest = json.loads((args.output/'manifest.json').read_text())
        for record in manifest['records']:
            name = record['source']['id']
            if not args.clip or name == args.clip:
                render(args.output/'clips'/name, model)
    report(args.output)
