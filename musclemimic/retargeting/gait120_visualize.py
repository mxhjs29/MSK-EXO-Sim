"""Render Gait120 marker fits and the exact retargeted musculoskeletal model."""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np


def render_clip(folder, env):
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from loco_mujoco.trajectory import Trajectory
    from musclemimic.web_viewer.c3d.markers import canonicalize_marker_label

    comparison=dict(np.load(folder / "comparison.npz"))
    traj=Trajectory.load(folder / "trajectory.npz",backend=np)
    q=np.asarray(traj.data.qpos); m=env._model; d=mujoco.MjData(m)
    labels=list(comparison["marker_labels"]); fitted_labels=list(comparison["fitted_marker_labels"])
    order=[fitted_labels.index(canonicalize_marker_label(n)) for n in labels]
    markers=comparison["markers_m"]; fitted=comparison["fitted_markers_m"][:,order]
    pelvis=[labels.index(n) for n in ["LASI","RASI","LPSI","RPSI"]]
    chains=[["LFHD","RFHD","RBHD","LBHD","LFHD"],
            ["C7","LSHO","LELB","LWRA","LFIN"], ["C7","RSHO","RELB","RWRA","RFIN"],
            ["C7","T10","LPSI","LASI","RASI","RPSI","LPSI"],
            ["LASI","LTHI","LKNE","LTIB","LANK","LTOE","LHEE","LANK"],
            ["RASI","RTHI","RKNE","RTIB","RANK","RTOE","RHEE","RANK"]]
    edges=[]
    for chain in chains:
        for a,b in zip(chain,chain[1:]):
            if a in labels and b in labels: edges.append((labels.index(a),labels.index(b)))
    width,height=400,464
    fig=plt.figure(figsize=(4,4.64),dpi=100,facecolor="#f5f8fc")
    ax=fig.add_subplot(111,projection="3d")
    ax.computed_zorder=False
    ax.set(xlim=(-.8,.8),ylim=(-.8,.8),zlim=(0,2),xlabel="X (m)",ylabel="Y (m)")
    ax.set_box_aspect((1.6,1.6,2)); ax.view_init(elev=12,azim=-60)
    ax.tick_params(labelsize=7); ax.set_title("Observed markers / fitted markers",fontsize=11)
    lines=[ax.plot([],[],[],c="#8ca2bc",lw=1)[0] for _ in edges]
    obs=ax.scatter([],[],[],s=14,c="#1670b9",depthshade=False,zorder=4)
    fit=ax.scatter([],[],[],s=18,c="#e58b24",marker="x",depthshade=False,zorder=5)
    fig.subplots_adjust(left=0,right=1,bottom=.055,top=.95)
    m.vis.global_.offwidth=max(m.vis.global_.offwidth,width)
    m.vis.global_.offheight=max(m.vis.global_.offheight,height)
    renderer=mujoco.Renderer(m,height=height,width=width)
    camera=mujoco.MjvCamera(); camera.distance=2.85; camera.elevation=-8
    option=mujoco.MjvOption(); option.sitegroup[:]=0
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",18)
    small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    samples=list(range(0,len(q),4))
    try:
        with imageio.get_writer(folder / "comparison.mp4",fps=25,codec="libx264",quality=8,
                                macro_block_size=16,ffmpeg_log_level="error") as writer:
            for j,i in enumerate(samples):
                center=markers[i,pelvis].mean(axis=0); center[2]=0
                a=markers[i]-center;b=fitted[i]-center
                obs._offsets3d=tuple(a.T);fit._offsets3d=tuple(b.T)
                for line,(s,t) in zip(lines,edges):
                    line.set_data_3d(a[[s,t],0],a[[s,t],1],a[[s,t],2])
                fig.canvas.draw(); source=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
                d.qpos[:]=q[i];mujoco.mj_forward(m,d)
                camera.lookat[:]=q[i,:3];camera.lookat[2]=.9
                images=[source]
                for angle in (90,145):
                    camera.azimuth=angle;renderer.update_scene(d,camera=camera,scene_option=option)
                    images.append(renderer.render().copy())
                frame=Image.new("RGB",(1200,528),"#f5f8fc")
                draw=ImageDraw.Draw(frame)
                draw.text((16,8),f"{folder.name}  |  source t={comparison['source_time_s'][i]:.2f}s",font=font,fill="#19354c")
                draw.text((16,34),"Blue: recorded markers   Orange: SMPL-H fit",font=small,fill="#334e68")
                draw.text((410,34),"MyoFullBody | view A",font=small,fill="#334e68")
                draw.text((810,34),"MyoFullBody | view B",font=small,fill="#334e68")
                for k,panel in enumerate(images):frame.paste(Image.fromarray(panel),(k*width,64))
                writer.append_data(np.asarray(frame))
                if j==len(samples)//2:frame.save(folder / "preview.png")
    finally:
        renderer.close();plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(12,6),layout="constrained")
    time=np.arange(len(q))/traj.info.frequency
    for row,side in enumerate(["r","l"]):
        for col,name in enumerate(["hip_flexion","knee_angle","ankle_angle"]):
            ax=axes[row,col];j=m.joint(f"{name}_{side}").qposadr[0]
            ax.plot(time,np.rad2deg(q[:,j]),c="#1670b9",lw=2)
            ax.set(title=f"{side.upper()} {name}",xlabel="Time within exported clip (s)",ylabel="Target angle (deg)")
            ax.grid(alpha=.2)
    fig.suptitle(folder.name+" | target model coordinates (no source-angle sign assumptions)")
    fig.savefig(folder / "joint_angles.png",dpi=130);plt.close(fig)
    print("RENDERED",folder.name,flush=True)


def build_report(output):
    from PIL import Image,ImageDraw,ImageFont
    manifest=json.loads((output / "manifest.json").read_text())
    records=manifest["records"]
    def embed(path,mime):return "data:"+mime+";base64,"+base64.b64encode(path.read_bytes()).decode()
    payload=[];rows=[]
    for r in records:
        clip=r["source"];m=r["metrics"];folder=output / "clips" / clip["id"]
        payload.append(dict(id=clip["id"],video=embed(folder/"comparison.mp4","video/mp4"),
                            poster=embed(folder/"preview.png","image/png"),plot=embed(folder/"joint_angles.png","image/png"),
                            trajectory=f"clips/{clip['id']}/trajectory.npz",source=clip["path"],metrics=m))
        rows.append(f"<tr><td>{clip['id']}</td><td>{m['frames']}</td><td>{m['marker_fit_mean_mm']:.2f}</td>"
                    f"<td>{m['marker_fit_p95_mm']:.2f}</td><td>{m['gmr_body_position_error_mean_m']*100:.2f}</td>"
                    f"<td>{m['floor_penetration_max_m']*1000:.4f}</td></tr>")
    options="".join(f'<option value="{i}">{r["source"]["id"]}</option>' for i,r in enumerate(records))
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gait120 平地行走 → 全身肌骨模型</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#19354c;background:#f5f8fc;max-width:1250px;margin:auto;padding:24px}
h1{font-size:30px}h2{margin-top:30px}video,img{width:100%;height:auto;border-radius:10px;background:#fff}select,button{font:inherit;padding:8px;margin:6px;border:1px solid #a6b9ca;border-radius:6px;background:white}a{color:#1263a0}
.note{padding:16px;background:#e8f1f8;border-left:4px solid #1670b9}.metrics{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}.metric{background:white;padding:10px 20px;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #cedae5;padding:7px;text-align:left}th{background:#e8f1f8}.table{overflow:auto}small{color:#536a80}
</style><body><h1>Gait120 平地行走 → 全身肌骨模型</h1>
<p>原始 TRC 标记 → SMPL-H 姿态拟合 → GMR → 关节耦合投影 → MyoFullBody。源时间尺度保留为 100 Hz。</p>
<div class="note">这是重映射参考轨迹的运动学回放。红色肌肉线条是模型几何显示，不代表肌肉激活；尚未运行肌肉策略训练或证明动态跟踪成功。</div>
<div class="metrics">METRIC_CARDS</div>
<label>片段 <select id="clip">OPTIONS</select></label><button id="speed">播放速度：1×</button><a id="download">下载当前轨迹</a>
<video id="video" controls loop muted playsinline preload="metadata"></video><p id="detail"></p>
<p><small>左：蓝色原始标记与橙色拟合标记；中、右：同一时刻的目标肌骨模型。相机跟随骨盆以便观察姿态，轨迹文件保留根部平移。源人体与目标模型体形不同，因此各视图独立居中。视频 25 fps，轨迹 100 Hz。</small></p>
<h2>目标关节轨迹</h2><img id="angles" alt="目标模型下肢关节角度曲线">
<h2>完整性与误差</h2><p>标记误差衡量 SMPL-H 对原始动捕的拟合；GMR 骨段位置误差衡量体形适配后的 IK 目标与目标模型之间的差异，两者不可混用。表中 GMR 误差为耦合精确投影前的求解器统计；投影引起的站点变化另存于 manifest.json。</p>
<div class="table"><table><thead><tr><th>片段</th><th>输出帧数</th><th>标记平均误差 mm</th><th>标记 P95 mm</th><th>GMR 平均骨段误差 cm</th><th>最大地板穿透 mm</th></tr></thead><tbody>ROWS</tbody></table></div>
<h2>使用范围</h2><p>CLIP_COUNT 个片段来自当前本地副本中可完整读取的 SUBJECT_LIST 数据，其余残缺文件已列入排除清单。每个片段保持独立，不把不同试次拼接，也不循环扩展动作。为计算中心差分速度，每段首尾各去掉一帧，保留的源时间与帧号写在 comparison.npz。</p>
<div class="note"><b>目标模型副本包含膝关节限位修正。</b>原模型左膝一个耦合平移的多项式为负，而限位只允许正值；其他若干依赖膝关节的限位也没有完全包含其多项式范围。本次保留原模型 GMR 结果，另将 CORRECTION_COUNT 个依赖膝关节限位按原耦合多项式的取值范围修正，并精确投影耦合关系。上方视频与主要轨迹使用修正副本；原模型资产、主膝屈曲范围、肌肉参数和既有训练权重未改动。训练时须使用此副本或等效配置。<a href="model/range_corrections.json">修正明细</a> · <a href="model/myofullbody_reference.xml">模型 XML</a> · <a href="levelwalking_original_model.npz">原模型 GMR 轨迹</a></div>
<p>输出检查包括有限数值、四元数、关节限位、关节耦合、地板穿透和训练环境加载。足底接触与滑移仍需结合后续动力学跟踪验证；当地面反力文件缺失时，低位足点速度只作为诊断量。</p>
<p><a href="levelwalking_trajectory.npz">下载全部轨迹（含独立片段边界）</a> · <a href="manifest.json">逐片段指标与源文件记录</a> · <a href="clips.csv">片段索引 CSV</a></p>
<script>const data=PAYLOAD;const sel=document.getElementById('clip'),video=document.getElementById('video');
function show(){const d=data[Number(sel.value)],m=d.metrics;video.src=d.video;video.poster=d.poster;video.playbackRate=1;document.getElementById('speed').textContent='播放速度：1×';document.getElementById('angles').src=d.plot;document.getElementById('download').href=d.trajectory;document.getElementById('detail').textContent=`${d.source} ｜ ${m.frames} 帧 / 100 Hz ｜ 标记平均误差 ${m.marker_fit_mean_mm.toFixed(2)} mm ｜ 目标：${m.nq} qpos / ${m.nv} qvel / ${m.nu} 执行器`;}
sel.addEventListener('change',show);document.getElementById('speed').addEventListener('click',()=>{video.playbackRate=video.playbackRate===1?.25:1;document.getElementById('speed').textContent='播放速度：'+video.playbackRate+'×';});show();</script></body></html>'''
    frames=sum(r["metrics"]["frames"] for r in records)
    mean=np.average([r["metrics"]["marker_fit_mean_mm"] for r in records],weights=[r["source"]["frames"] for r in records])
    cards=f'<div class="metric"><b>{len(records)}</b> 个完整片段</div><div class="metric"><b>{frames}</b> 帧参考轨迹</div><div class="metric"><b>{mean:.2f} mm</b> 平均标记拟合误差</div><div class="metric"><b>100 Hz</b> 输出频率</div>'
    corrections=json.loads((output/'model/range_corrections.json').read_text())
    document=document.replace('CLIP_COUNT',str(len(records))).replace('SUBJECT_LIST','、'.join(sorted({r['source']['subject'] for r in records}))).replace('CORRECTION_COUNT',str(len(corrections['changes'])))
    document=document.replace('METRIC_CARDS',cards).replace('OPTIONS',options).replace('ROWS',''.join(rows)).replace('PAYLOAD',json.dumps(payload,ensure_ascii=False))
    (output / "index.html").write_text(document)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    examples=[next(r for r in records if r['source']['subject']==s) for s in sorted({r['source']['subject'] for r in records})]
    sheet=Image.new('RGB',(1200,560*len(examples)),'#f5f8fc');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(examples):
        folder=output/'clips'/r['source']['id'];sheet.paste(Image.open(folder/'preview.png'),(0,560*i))
        draw.text((15,560*i+533),f"Marker fit mean: {r['metrics']['marker_fit_mean_mm']:.2f} mm | source and target at matched times",font=font,fill='#19354c')
    sheet.save(output/'overview.png')
    print('REPORT',output/'index.html',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('outputs/gait120_levelwalking'))
    p.add_argument('--clip');p.add_argument('--report-only',action='store_true');args=p.parse_args()
    if not args.report_only:
        from musclemimic.retargeting.gait120_retarget import make_reference_env
        env,_=make_reference_env()
        folders=[args.output/'clips'/args.clip] if args.clip else sorted((args.output/'clips').iterdir())
        for folder in folders:
            if (folder/'result.json').exists():render_clip(folder,env)
    if not args.clip:build_report(args.output)


if __name__=='__main__':main()
