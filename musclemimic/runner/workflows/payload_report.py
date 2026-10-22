"""Offline review artifact for loaded PPO transfer: physical checks and rollouts."""
from musclemimic.runner.workflows.payload_finetune import *
import base64
import html


def video(task,label):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    folder=OUT/task/'evaluations'/label
    result=json.loads((folder/'results.json').read_text())
    cfg=OmegaConf.load(OUT/task/'base_config.yaml');cfg.experiment.env_params.payload_mass_kg=result['mass_kg']
    e=make_cpu(cfg);m,d=e._model,e._data
    q=np.load(folder/'start_0_rollout.npz')['qpos']
    m.vis.global_.offwidth=960;m.vis.global_.offheight=720
    renderer=mujoco.Renderer(m,height=720,width=960)
    opt=mujoco.MjvOption();opt.sitegroup[:]=0;opt.tendongroup[:]=0
    cam=mujoco.MjvCamera();cam.azimuth=135;cam.elevation=-12;cam.distance=2.5
    cam.lookat[:]=q[0,:3]+[0,0,.10]
    with imageio.get_writer(folder/'rollout.mp4',fps=25,codec='libx264',quality=8,macro_block_size=16) as writer:
        for frame in range(0,len(q),4):
            d.qpos[:]=q[frame];d.qvel[:]=0;mujoco.mj_forward(m,d)
            renderer.update_scene(d,camera=cam,scene_option=opt)
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,960,36),fill='white')
            draw.text((12,10),f'{task} | {result["mass_kg"]} kg | {label} | t={frame*.01:.2f}s | ideal attachment',fill='black')
            writer.append_data(np.asarray(im))
    renderer.close();e.stop()


def uri(path,mime):return f'data:{mime};base64,'+base64.b64encode(Path(path).read_bytes()).decode()


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    summary={};sections=[]
    box_superseded=(OUT/'bend/scope_superseded.json').exists()
    if box_superseded:
        sections.append('<section><h2>箱子方案已变更：本页箱子结果仅为旧约束基线</h2><p>用户现要求箱子依靠双手接触与摩擦搬起。下方箱子模型使用连接约束，其通过结果不代表新要求已通过。摩擦接触方案在 outputs/payload_box_friction_v2 中独立验证与训练。</p></section>')
    selection=OUT/'final_selection.json'
    if selection.exists():
        chosen=json.loads(selection.read_text())
        if box_superseded and 'bend' in chosen:
            chosen['bend']['all_quality_pass']=False
        sections.append('<section><h2>最终模型与验收结论</h2><table><tr><th>任务</th><th>四个起点全部达标</th><th>checkpoint</th></tr>')
        for task,value in chosen.items():
            sections.append(f'<tr><td>{task}</td><td>{"通过" if value["all_quality_pass"] else "未通过，仍需改进"}</td><td><code>{html.escape(value["checkpoint"])}</code></td></tr>')
        sections.append('</table><p>这些是同一参考动作上的四个确定性重置起点，验证完整剩余轨迹；不代表未见动作、质量或真实抓握的泛化测试。</p></section>')
    else:
        sections.append('<section><h2>阶段性报告：训练与评估仍在进行</h2><p>尚未完成最终模型选择，以下结果包含基线与中间 checkpoint，不能作为最终验收结论。后台流程完成后会更新本文件；实时状态见 audit/background_status.json。</p></section>')
    sections.append('''<section><h2>物理建模、参考轨迹与迁移方法</h2>
<p>动力学为 M(q)q̈ + C(q,q̇) + g(q) = τ<sub>muscle</sub> + J<sup>T</sup>f<sub>contact</sub>。刚性持哑铃后，新增质量和惯量直接进入人体质量矩阵与重力项；额外重力广义负载为 Δg(q) = J<sub>COM</sub><sup>T</sup>[0,0,mg]。箱子保持独立自由刚体，两手通过柔性连接约束传力，静止时两手竖直支撑力之和应约等于 5×9.81 = 49.05 N。</p>
<p>哑铃沿用 324 帧 lift 参考，调整握持手型及安装偏移以减少几何穿插。箱子为 18.2×20×16 cm，握点间距 21 cm；取 bend 原始参考第 140–480 帧（341 帧）的起身搬举段，只调整上肢，使双手握点间距一致，并加入时间平滑。调整后的参考分别用于微调前后比较。</p>
<p>观测仍为 2808 维，肌肉动作仍为 416 维。Actor/Critic 各为 5 层、每层 1024 单元的 SiLU 网络，带 LayerNorm。迁移原 checkpoint 的网络参数和运行归一化统计，重新初始化优化器；使用 4096 环境、64 步 rollout、128 个 minibatch、1 个 PPO epoch，每次更新采集 262144 个环境步。初始微调学习率 3×10⁻⁵，必要的后续精调使用 1×10⁻⁵。</p>
<p>先在 32 环境做训练 smoke test，并独立验证 9 个起始相位的重置、物体质量、承重传递和落地接触，再扩展到 4096 环境。lift 先进行 0.5 kg 适应，取第 20 次更新作为后续 1 kg 与 2 kg 分支的共同起点。发现完整动作后半程漂移后，后续 lift 改为从动作开头重置，增强骨盆位置与肘关节跟踪。bend 直接从原无负载 PPO 迁移到 5 kg。</p>
<p>奖励保留原动作模仿项。后续 lift 额外加入 0.3 exp(−e<sub>elbow</sub>²/0.20²) 和手型项 0.1 exp(−mean(e<sub>finger</sub>²)/0.5²)，角度单位为 rad；骨盆位置奖励权重由 0.1 提到 0.3。箱子增加 0.05 exp(−mean(‖e<sub>grip</sub>‖²)/0.025²)、0.1 exp(−‖e<sub>pos</sub>‖²/0.10²) 和 0.1 exp(−[1−〈q,q*〉²]/0.2)，位置单位为 m，q 为单位四元数。理想连接已承担握持，因此握点奖励主要限制柔性约束偏差，不意味着学会摩擦抓握。</p>
<p>本阶段没有肌肉激活惩罚：先确认增加真实负载后仍能完成动作。此前的 0.5 kg 阶段没有额外肘关节奖励，因此不同训练阶段的总奖励不能直接横向比较。</p></section>''')
    for task,title in [('lift','右手哑铃 · 2 kg'),('bend','双手持箱 · 5 kg')]:
        p=OUT/task;sections.append(f'<section><h2>{title}</h2>')
        physics=p/'physics/acceptance.json'
        if physics.exists():
            a=json.loads(physics.read_text());sections.append('<h3>物理验收</h3><pre>'+html.escape(json.dumps(a,ensure_ascii=False,indent=2))+'</pre>')
        if task=='bend' and (p/'evaluations/final_5kg/results.json').exists():
            a=json.loads((p/'evaluations/final_5kg/results.json').read_text())
            rows=a['records'][0]['trajectory']
            if all('hand_support_world_n' in x for x in rows):
                force=np.asarray([x['hand_support_world_n'] for x in rows])
                fz=force[:,:,2].mean(axis=0)
                sections.append(f'<p>完整起身轨迹中实际约束支撑力：右手竖直均值 {fz[0]:.2f} N，左手 {fz[1]:.2f} N，合计 {fz.sum():.2f} N。动态力包含物体加速度和启动瞬态，不要求逐帧等于静态重量 49.05 N。</p>')
        prep=p/'preparation.json'
        if prep.exists():
            a=json.loads(prep.read_text());sections.append('<div class="images">')
            for path in a['images'][:4]:sections.append(f'<img src="{uri(path,"image/png")}">')
            sections.append('</div>')
        records=[];series=[]
        for f in sorted((p/'evaluations').glob('*/results.json')):
            a=json.loads(f.read_text());label=f.parent.name
            trajectories=[v['trajectory'] for v in a['records']];rows=[s for tr in trajectories for s in tr]
            if not rows:continue
            r=dict(label=label,mass_kg=a['mass_kg'],complete=sum(v['termination']=='reference_end' for v in a['records']),total=len(a['records']),
                quality_pass=int(sum(v['termination']=='reference_end' and np.mean([x.get('joint_error_deg',1e3) for x in v['trajectory']])<=20 and max(x.get('root_error_m',1e3) for x in v['trajectory'])<=.25 and v['peak_grip_error_m']<=.03 for v in a['records'])),
                joint_mae_deg=float(np.mean([x.get('joint_error_deg',0) for x in rows])),
                root_error_cm=float(np.mean([x.get('root_error_m',0) for x in rows])*100),
                mean_activation=float(np.mean([x.get('elbow8_activation_rms',x['activation']) for x in rows])),
                peak_grip_mm=max(v['peak_grip_error_m'] for v in a['records'])*1000,checkpoint=a['checkpoint'])
            records.append(r);series.append((label,a['mass_kg'],a['records'][0]['trajectory']))
            sections.append(f'<details><summary>{html.escape(label)}：逐起点记录与 checkpoint</summary><pre>'+html.escape(json.dumps(dict(checkpoint=a['checkpoint'],episodes=[{k:v for k,v in x.items() if k not in ['trajectory','images']} for x in a['records']]),ensure_ascii=False,indent=2))+'</pre></details>')
        summary[task]=records
        sections.append('<table><tr><th>评估</th><th>负载 kg</th><th>完成 / 质量达标</th><th>关节 MAE °</th><th>骨盆误差 cm</th><th>激活 RMS</th><th>最大握点误差 mm</th></tr>')
        for r in records:sections.append(f'<tr><td>{r["label"]}</td><td>{r["mass_kg"]}</td><td>{r["complete"]}/{r["total"]} · {r["quality_pass"]}/{r["total"]}</td><td>{r["joint_mae_deg"]:.2f}</td><td>{r["root_error_cm"]:.2f}</td><td>{r["mean_activation"]:.3f}</td><td>{r["peak_grip_mm"]:.2f}</td></tr>')
        sections.append('</table><p class="note">质量门槛：完整结束、任务关节 MAE ≤20°、骨盆峰值误差 ≤25 cm、握点峰值误差 ≤3 cm。误差与激活统计使用各回合实际运行帧，提前失败会缩短统计时长。不同负载、不同截断时长不作为严格省力对比。lift 激活使用右肘 8 肌；bend 此处为全身激活。</p>')
        selected=[s for s in series if s[0] in ['original_2kg','original_5kg','final_2kg','final_5kg']]
        if selected:
            fig,ax=plt.subplots(3,1,figsize=(10,7),sharex=True)
            for label,mass,rows in selected:
                t=[x['t'] for x in rows]
                ax[0].plot(t,np.rad2deg([x['joint_angle_rad'] for x in rows]),label=label)
                ax[1].plot(t,[x['muscle_torque_nm'] for x in rows],label=label)
                ax[2].plot(t,[x.get('elbow8_activation_rms',x['activation']) for x in rows],label=label)
            ref=selected[-1][2];ax[0].plot([x['t'] for x in ref],np.rad2deg([x['reference_angle_rad'] for x in ref]),'k--',label='reference')
            for a,y in zip(ax,['Joint angle (deg)','Muscle torque (Nm)','Activation RMS']):a.set_ylabel(y);a.grid(alpha=.25);a.legend()
            ax[-1].set_xlabel('Time (s)');fig.tight_layout();plot=p/'comparison.png';fig.savefig(plot,dpi=140);plt.close(fig)
            sections.append(f'<img class="plot" src="{uri(plot,"image/png")}">')
        sections.append('<div class="videos">')
        for label,_,_ in selected:
            v=p/'evaluations'/label/'rollout.mp4'
            if v.exists():sections.append(f'<div><h3>{label}</h3><video controls preload="metadata" src="{uri(v,"video/mp4")}"></video></div>')
        sections.append('</div></section>')
    stages=[]
    for p in sorted(OUT.glob('*/training/*/completed.json')):
        a=json.loads(p.read_text());a['path']=str(p.parent);stages.append(a)
    sections.append('<section><h2>训练记录与复现文件</h2><pre>'+html.escape(json.dumps(stages,ensure_ascii=False,indent=2))+'</pre>')
    for p in sorted(OUT.glob('*/training/*/stage_transition.json'))+sorted(OUT.glob('*/training/*/invalid_run.json')):
        sections.append('<details><summary>'+html.escape(str(p.relative_to(OUT)))+'</summary><pre>'+html.escape(p.read_text())+'</pre></details>')
    sections.append('<p>0.5 kg 原计划 80 次更新，实际提前结束，仅将 checkpoint_20 用于迁移。早期 curriculum_1kg_fullstart 因终止条件的轨迹缓存索引错误导致每回合仅一步，已排除，修复后的 v2 从有效的 0.5 kg checkpoint 重新迁移。有效单元测试 33 项通过；另 11 项旧集成测试因测试 fixture 使用未注册的旧环境名而未执行，详见 audit/test_scope.json。</p>')
    for p in sorted(OUT.glob('*/base_config.yaml')):sections.append('<details><summary>'+str(p.relative_to(OUT))+'</summary><pre>'+html.escape(p.read_text())+'</pre></details>')
    sections.append('<p>checkpoint 为 Orbax 目录，独立保存在上表 path/checkpoints 下。原始轨迹、逐帧指标、日志、物理 fixture 保存在本报告同目录对应任务子目录和 audit 中。</p></section>')
    document='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>负载动作 PPO 微调验证</title><style>
body{font:17px/1.6 system-ui;background:#eef2f5;color:#183046;max-width:1280px;margin:35px auto;padding:0 20px}section,header{background:white;padding:26px;border-radius:15px;margin:24px 0}h1,h2{line-height:1.3}pre{font:12px/1.5 monospace;white-space:pre-wrap;max-height:430px;overflow:auto;background:#f3f6f8;padding:15px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}.images,.videos{display:grid;grid-template-columns:1fr 1fr;gap:15px}.images img,video,.plot{width:100%}.note{font-size:14px;color:#526575}details{margin:12px 0}@media print{section{break-before:page}video{max-height:280px}}
</style><header><h1>从无负载 PPO 到持物动作</h1><p>右手哑铃 2 kg → 双手持箱 5 kg；小批物理验证后，扩展到 4096 个并行环境。</p><p>加载已有 PPO 的网络权重和观测归一化统计，重新建立优化器，以较低学习率、逐级增重方式微调。动作仍是原有 416 维肌肉控制，环境增加物体真实质量、惯量与碰撞。</p><p><b>当前范围：</b>预先持物的负载动作。哑铃通过手部刚性连接承重，箱子通过两个柔性连接约束承重。握持处用理想连接代替摩擦抓握；尚未训练自主抓取、松手或手指防滑。没有世界固定支撑，没有逐帧把物体移动到参考位置。</p><p>原动作模仿奖励继续约束人体运动；哑铃补充手型、肘关节和骨盆跟踪，箱子补充物体位置、姿态和握点对齐奖励。暂不增加肌肉激活惩罚，优先学会稳定承重。</p></header>'''+''.join(sections)+'</html>'
    (OUT/'report.html').write_text(document);save_json(OUT/'evaluation_summary.json',summary)
    print('REPORT',OUT/'report.html')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['lift','bend']);p.add_argument('--label');a=p.parse_args()
    if a.task and a.label:video(a.task,a.label)
    else:report()
