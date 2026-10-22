"""Self-contained hand/box friction validation report and reference preview."""
import base64
import html
import hashlib
import json
import time
from pathlib import Path

from musclemimic.runner.workflows.box_friction import OUT, make_cpu, OmegaConf, np, mujoco, render, environment_signature


def uri(path,mime):
    return f'data:{mime};base64,'+base64.b64encode(Path(path).read_bytes()).decode()


def reference_media():
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    folder=OUT/'bend';e=make_cpu(OmegaConf.load(folder/'base_config.yaml'));m,d=e._model,e._data
    q=np.asarray(e.th.traj.data.qpos).copy();origin=q[0,:2].copy();q[:,:2]-=origin;q[:,-7:-5]-=origin
    for frame in [0,len(q)//3,2*len(q)//3,len(q)-1]:
        d.qpos[:]=q[frame];d.qvel[:]=0;mujoco.mj_forward(m,d)
        render(e,folder/'reference_images',[frame],policy=True)
    m.vis.global_.offwidth=960;m.vis.global_.offheight=720
    renderer=mujoco.Renderer(m,height=720,width=960)
    option=mujoco.MjvOption();option.sitegroup[:]=0;option.tendongroup[:]=0
    cam=mujoco.MjvCamera();cam.azimuth=135;cam.elevation=-12;cam.distance=2.6
    cam.lookat[:]=q[0,:3]+[0,0,.15]
    with imageio.get_writer(folder/'reference_preview.mp4',fps=25,codec='libx264',quality=7,macro_block_size=16) as writer:
        for frame in range(0,len(q),4):
            d.qpos[:]=q[frame];d.qvel[:]=0;mujoco.mj_forward(m,d)
            renderer.update_scene(d,camera=cam,scene_option=option)
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,960,36),fill='white')
            draw.text((12,10),f'Reference trajectory preview (NOT PPO rollout) | 5 kg | t={frame*.01:.2f}s',fill='black')
            writer.append_data(np.asarray(im))
    renderer.close();e.stop()


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reference_sha=hashlib.sha256((OUT/'bend/reference.npz').read_bytes()).hexdigest()
    signature=environment_signature()
    stamp=OUT/'bend/reference_preview.sha256'
    if not (OUT/'bend/reference_preview.mp4').exists() or not stamp.exists() or stamp.read_text()!=signature:
        reference_media();stamp.write_text(signature)
    physics=json.loads((OUT/'bend/physics/acceptance.json').read_text())
    ref=json.loads((OUT/'bend/reference_audit.json').read_text())
    status=json.loads((OUT/'audit/background_status.json').read_text()) if (OUT/'audit/background_status.json').exists() else {}
    progress=[]
    for root,task in [(OUT/'bend/training','箱子'),(OUT.parent/'payload_finetune_v1/lift/training','哑铃')]:
        for p in sorted(root.glob('*/metrics.jsonl')):
            if task=='箱子' and not (p.parent.name=='smoke_platform_final' or 'kg_v2_' in p.parent.name):continue
            if (p.parent/'invalid_run.json').exists():continue
            if task=='哑铃' and p.parent.name!='target_2kg_random_300':continue
            rows=[]
            for line in p.read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass  # concurrent append may end mid-record
            if rows:progress.append(dict(task=task,run=p.parent.name,logged_updates=len(rows),
                timesteps=rows[-1]['Current Timestep'],mean_return=rows[-1]['Mean Episode Return'],
                mean_episode_length=rows[-1]['Mean Episode Length'],ppo_loss=rows[-1]['PPO/total_loss']))
    fig,ax=plt.subplots(3,1,figsize=(10,7),sharex=True)
    for name,label in [('nominal','friction=0.8'),('zero_friction','friction=0')]:
        rows=json.loads((OUT/f'bend/physics/{name}.json').read_text())['trajectory'];t=[r['t'] for r in rows]
        ax[0].plot(t,[r['height_m'] for r in rows],label=label)
        ax[1].plot(t,[r['friction_vertical_n'] for r in rows],label=label)
        ax[2].plot(t,[r['penetration_m']*1000 for r in rows],label=label)
    for a,label in zip(ax,['Box center height (m)','Hand friction support (N)','Hand penetration (mm)']):
        a.set_ylabel(label);a.grid(alpha=.25);a.legend();a.axvline(2.3,color='gray',ls='--')
    ax[1].axhline(49.05,color='gray',ls=':');ax[-1].set_xlabel('Time (s); hands open at 2.3 s')
    fig.tight_layout();fig.savefig(OUT/'physics_comparison.png',dpi=130);plt.close(fig)
    body=['<header><h1>双手摩擦搬箱：训练前验收与训练进度</h1>',
        f'<p>快照时间：{time.strftime("%Y-%m-%d %H:%M:%S %z")}。实时状态见 audit/background_status.json。</p>',
        '<p><b>当前目标：</b>箱子为 5 kg 自由刚体，仅依靠手部碰撞、夹紧力及摩擦承重。哑铃继续使用已有连接约束方案。</p>',
        '<p>手与箱子之间没有连接等式、吸附力或直接施加给箱子的辅助力。双手原有碰撞几何保持启用，并补充薄的掌部软组织接触面；掌面没有新增质量或独立控制器。</p></header>',
        '<section><h2>碰撞几何修正</h2><p>原前臂使用恒定半径 30 mm 的胶囊体，末端过于粗大。箱子任务改为半径 30 → 24 → 17 mm 的重叠胶囊体，沿前臂向手腕逐渐变细；人体质量、惯量、关节及肌肉保持原值，碰撞持续开启。参考轨迹还对全身与箱子的距离加入避碰约束，并逐帧验收。该近似接触皮肤仍需通过后续真实手部几何标定进一步验证。</p></section>',
        '<section><h2>验收结论</h2>',f'<p>摩擦物理测试：<b>{"通过" if physics["passed"] else "未通过"}</b>；参考轨迹手箱最大穿透：{ref["max_hand_box_penetration_m"]*1000:.3f} mm；9 个起点重置已检查。</p>',
        '<p>这项通过表示物理模型与训练环境就绪。PPO 是否学会搬箱另行评估，不能用测试装置的成功代替策略成功。</p>',
        '<pre>'+html.escape(json.dumps(status,ensure_ascii=False,indent=2))+'</pre></section>',
        '<section><h2>训练计数快照</h2><p>以下来自实际 PPO 日志；回报提高不能单独证明承重成功，仍需独立接触与完整轨迹评估。</p><pre>'+html.escape(json.dumps(progress,ensure_ascii=False,indent=2))+'</pre></section>',
        '<section><h2>摩擦承重与反例</h2><p>独立测试装置使用实际手部碰撞几何，双手各由有限的 60 N 夹紧驱动，竖直运动由测试滑台驱动。箱子完全自由，只承受重力与碰撞力。此装置不是人体 PPO。</p>',
        '<p>保持条件为 μ(N右 + N左) ≥ mg。当前 μ=0.8、两手合计夹紧力约 120 N，对应摩擦上限约 96 N，超过 5 kg 箱子的 49.05 N 重量。实际竖直摩擦支撑由仿真接触力读取。</p>',
        f'<img src="{uri(OUT/"physics_comparison.png","image/png")}">',
        '<pre>'+html.escape(json.dumps(physics,ensure_ascii=False,indent=2))+'</pre></section>',
        '<section><h2>参考动作与手箱位置</h2>',
        f'<p>使用原 bend 第 {ref["source_start"]} 帧起的 {ref["frames"]} 帧。调整上肢、手掌朝向及箱子路径，使两个掌面与侧壁对齐；手指保持在箱体外。箱子尺寸 24×20×24 cm，初始置于约 {ref["support_height_m"]*100:.1f} cm 的低台面。当前覆盖双手接触后提起动作，尚未覆盖接近、放回、释放的完整搬运任务。</p>',
        f'<p>最大相邻帧上肢关节变化 {ref["max_arm_frame_change_deg"]:.2f}°。以下视频是参考轨迹回放，不是训练成功视频。</p>',
        f'<video controls preload="metadata" src="{uri(OUT/"bend/reference_preview.mp4","video/mp4")}"></video><div class="images">']
    for p in sorted((OUT/'bend/reference_images').glob('*full.png')):body.append(f'<img src="{uri(p,"image/png")}">')
    body.append('</div></section><section><h2>策略迁移与训练</h2><p>从原无负载 bend PPO 的 checkpoint_10000 迁移。保留 416 维肌肉动作，观测从 2808 维扩展至 2827 维，追加箱子相对位置、姿态、速度和双手相对箱子的位置。新增输入权重初始化为零，检查迁移时策略输出不变，优化器重新初始化。</p><p>奖励保留人体动作模仿，并加入掌面朝向、手箱接触位置、箱子位置姿态和手型跟踪。箱子下落超过参考 15 cm，或手箱数值穿透超过 3 mm 时终止回合，避免策略靠明显穿模完成任务。3 mm 是失败门槛，不是参考轨迹允许穿透量；参考轨迹的实际最大接触深度见上方逐帧验收结果。</p><p>完成 32 环境 smoke test 与独立接触检查后，扩大到 4096 环境，按 1 → 3 → 5 kg 逐级训练。每级需通过完整轨迹、双手接触、摩擦支撑和穿透检查才能增重。后台会保存 checkpoint、评估和日志；最多重复一次当前级别后转为待检查，避免无限训练。</p><table><tr><th>评估</th><th>质量达标</th><th>最大手箱穿透</th></tr>')
    for p in sorted((OUT/'bend/evaluations').glob('*/results.json')):
        a=json.loads(p.read_text());rows=a['episodes']
        if a.get('environment_signature')!=signature:continue
        body.append(f'<tr><td>{html.escape(p.parent.name)} · {a["mass_kg"]} kg</td><td>{sum(r["quality_pass"] for r in rows)}/{len(rows)}</td><td>{max(r["peak_penetration_m"] for r in rows)*1000:.3f} mm</td></tr>')
    body.append('</table></section>')
    lift_root=OUT.parent/'payload_finetune_v1'
    lift_state=lift_root/'audit/lift_continuation/background_status.json'
    if lift_state.exists():
        body.append('<section><h2>右手 2 kg 哑铃继续训练</h2><p>哑铃通过刚性连接随右手运动，已验证真实质量、重力力矩与重置。使用 GPU2、4096 个并行环境，从上一轮 2 kg checkpoint 继续训练 300 个 PPO 更新；改用随机起点，学习从动作各阶段承重。此前确定性四起点评估仅 1/4 达到关节误差质量阈值，不能宣称已经全部学会。GPU3 的箱子单独采用自由刚体摩擦接触任务。</p><pre>'+html.escape(lift_state.read_text())+'</pre></section>')
    body.append('<section><h2>配置与日志快照</h2>')
    files=[OUT/'bend/base_config.yaml',OUT/'audit/scaling_gate.json',OUT/'audit/full_geometry.json',OUT/'audit/warp_contact_sensor.json',OUT/'audit/physics_final.log',OUT/'audit/smoke_platform_final.log']
    files+=list((OUT/'bend/training').glob('*/transfer.json'))
    for p in files:
        if p.exists():body.append('<details><summary>'+html.escape(str(p.relative_to(OUT)))+'</summary><pre>'+html.escape(p.read_text())+'</pre></details>')
    body.append('<p>完整 checkpoint 为单独的 Orbax 目录，位于 bend/training/*/checkpoints；reference.npz、逐帧评估、物理 fixture、完整日志保存在本报告所在文件夹。旧的连接约束箱子结果保留在 payload_finetune_v1，仅作为历史基线。</p><p>接触模型依据 <a href="https://mujoco.readthedocs.io/en/3.4.0/modeling.html#contact-parameters">MuJoCo 3.4 接触参数说明</a>。</p></section>')
    document='<!doctype html><html lang="zh"><meta charset="utf-8"><title>摩擦搬箱验收</title><style>body{font:17px/1.7 system-ui;color:#183046;background:#eef2f5;max-width:1200px;margin:30px auto;padding:0 20px}header,section{background:white;padding:25px;margin:20px 0;border-radius:12px}img,video{width:100%}.images{display:grid;grid-template-columns:1fr 1fr;gap:12px}pre{white-space:pre-wrap;max-height:400px;overflow:auto;background:#f3f5f7;padding:15px;font:12px/1.5 monospace}td,th{padding:10px;border-bottom:1px solid #ddd}table{width:100%;border-collapse:collapse}h1,h2{line-height:1.3}</style>'+''.join(body)+'</html>'
    path=OUT/'report.html';tmp=path.with_suffix('.tmp');tmp.write_text(document);tmp.replace(path)
    print('REPORT',path,flush=True)


if __name__=='__main__':main()
