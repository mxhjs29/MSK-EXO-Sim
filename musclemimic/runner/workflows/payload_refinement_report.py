"""Self-contained, phase-aligned reference/before/after videos and force curves."""
import base64
import fcntl
import hashlib
import html
import json
import time
from pathlib import Path
import numpy as np
import mujoco
from omegaconf import OmegaConf
from musclemimic.runner.workflows.payload_refinement import OUT,make_cpu,evaluate


def uri(path,mime):return f'data:{mime};base64,'+base64.b64encode(Path(path).read_bytes()).decode()


def comparison_video(task,start,env,before,after,old_folder,new_folder,path):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    m,d=env._model,env._data
    old=np.load(old_folder/f'start_{start}_rollout.npz')['qpos']
    new=np.load(new_folder/f'start_{start}_rollout.npz')['qpos'] if after else None
    reference=np.asarray(env.th.traj.data.qpos).copy();origin=reference[start,:2].copy()
    reference[:,:2]-=origin
    if task=='bend':reference[:,-7:-5]-=origin
    reference=reference[start+1:]
    width,scene_height,height=480,544,624
    m.vis.global_.offwidth=width;m.vis.global_.offheight=scene_height
    renderer=mujoco.Renderer(m,height=scene_height,width=width)
    option=mujoco.MjvOption();option.sitegroup[:]=0;option.tendongroup[:]=0
    cam=mujoco.MjvCamera();cam.azimuth=135;cam.elevation=-12;cam.distance=2.8
    anchor=reference[0,:3]+[0,0,.12]
    labels=['REFERENCE','BEFORE','AFTER','AFTER: GRIP DETAIL']
    previews=[]
    pending=path.with_name(path.stem+'.working.mp4')
    with imageio.get_writer(pending,fps=25,codec='libx264',quality=7,macro_block_size=16) as writer:
        for i in range(0,len(reference),4):
            panels=[]
            for panel in range(4):
                if panel>=2 and new is None:
                    im=Image.new('RGB',(width,height),'#eef2f5');draw=ImageDraw.Draw(im)
                    draw.text((45,240),'New checkpoint evaluation pending',fill='black');panels.append(np.asarray(im));continue
                poses=reference if panel==0 else old if panel==1 else new
                index=min(i,len(poses)-1);d.qpos[:]=poses[index];d.qvel[:]=0;mujoco.mj_forward(m,d)
                cam.lookat[:]=anchor;cam.distance=2.8;cam.azimuth=135;cam.elevation=-12
                if panel==3:
                    cam.lookat[:]=d.xpos[env.payload_body_id]
                    cam.azimuth=235;cam.elevation=-10;cam.distance=.72 if task=='bend' else .52
                renderer.update_scene(d,camera=cam,scene_option=option)
                im=Image.new('RGB',(width,height),'white')
                im.paste(Image.fromarray(renderer.render()),(0,0));draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,width,58),fill='white')
                draw.text((12,10),labels[panel],fill='black')
                draw.text((12,31),f'{task} | {env.payload_mass_kg:g} kg | start={start} | phase t={(start+i+1)*.01:.2f}s',fill='black')
                if panel:
                    record=before if panel==1 else after;rows=record['trajectory'];row=rows[index]
                    draw.rectangle((0,height-76,width,height),fill='white')
                    if task=='lift':
                        text=f"Elbow error: {row['joint_error_deg']:.1f} deg\nElbow8 activation RMS: {row['elbow8_activation_rms']:.3f}"
                    else:
                        text=f"Normal R/L: {row['normal_n'][0]:.1f} / {row['normal_n'][1]:.1f} N\nFriction Z: {sum(row['friction_vertical_n']):.1f} N | Non-hand Z: {row['other_vertical_n']:.1f} N\nBox position error: {row['object_error_m']*100:.1f} cm"
                    draw.multiline_text((12,height-66),text,fill='black',spacing=5)
                    if i>=len(poses) and record['termination']!='reference_end':
                        draw.rectangle((0,58,width,91),fill='#a21b2d')
                        draw.text((12,68),f"TERMINATED at {record['steps']*.01:.2f}s; final pose held",fill='white')
                panels.append(np.asarray(im))
            combined=np.concatenate(panels,axis=1);writer.append_data(combined)
            if i in [0,4*(len(reference)//8)]:
                picture=path.with_name(path.stem+f'_frame_{i}.png');Image.fromarray(combined).save(picture);previews.append(str(picture))
    renderer.close();pending.replace(path);return previews


def plot_comparison(task,start,env,before,after,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(12,7));ax=ax.ravel()
    m=env._model;ref=np.asarray(env.th.traj.data.qpos);tref=np.arange(start+1,len(ref))*.01
    if task=='lift':
        adr=int(m.jnt_qposadr[m.joint('elbow_flexion_r').id]);ax[0].plot(tref,np.rad2deg(ref[start+1:,adr]),'k:',label='Reference')
    else:ax[0].plot(tref,ref[start+1:,-5],'k:',label='Reference')
    for record,label,color,style in [(before,'Before','#d36729','--'),(after,'After','#1678a4','-')]:
        if record is None:continue
        rows=record['trajectory'];t=[r['phase_t'] for r in rows]
        if task=='lift':
            vals=[[np.rad2deg(r['joint_angle_rad']) for r in rows],[r['joint_error_deg'] for r in rows],
                  [r['elbow8_activation_rms'] for r in rows],[r['muscle_torque_nm'] for r in rows]]
            for a,v in zip(ax,vals):a.plot(t,v,style,color=color,label=label)
        else:
            ax[0].plot(t,[r['height_m'] for r in rows],style,color=color,label=label)
            for side,side_color in [(0,'#1678a4'),(1,'#aa4295')]:
                ax[1].plot(t,[r['normal_n'][side] for r in rows],style,color=side_color,alpha=.7,label=label+(' R' if side==0 else ' L'))
            ax[2].plot(t,[sum(r['friction_vertical_n']) for r in rows],style,color=color,label=label+' hand friction')
            ax[2].plot(t,[r['other_vertical_n'] for r in rows],style,color='#8a286c' if label=='After' else '#777',alpha=.7,label=label+' non-hand')
            ax[3].plot(t,[r['object_error_m']*100 for r in rows],style,color=color,label=label)
            if all('object_orientation_error_deg' in r for r in rows):
                if 'pose_angle_axis' not in locals():
                    pose_angle_axis=ax[3].twinx()
                pose_angle_axis.plot(t,[r['object_orientation_error_deg'] for r in rows],style,
                                     color='#30977a' if label=='After' else '#a48225',alpha=.75,label=label+' angle')
    if task=='lift':
        titles=['Elbow angle (deg)','Instantaneous absolute error (deg)','Elbow8 activation RMS','Muscle elbow torque (Nm)']
        ax[1].axhline(20,color='gray',ls=':',label='20 deg (MAE gate, not pointwise)')
    else:
        titles=['Box center height (m)','Hand normal force (N)','Vertical support force (N)','Box position error (cm)']
        ax[1].axhline(1.5*env.payload_mass_kg*9.81/(2*.8),color='gray',ls=':',label='Squeeze target per hand')
        ax[2].axhline(env.payload_mass_kg*9.81,color='gray',ls=':',label='Box weight')
        ax[3].axhline(10,color='gray',ls=':',label='10 cm gate')
        if 'pose_angle_axis' in locals():
            pose_angle_axis.set_ylabel('Box orientation error (deg)')
            pose_angle_axis.legend(fontsize=7,loc='upper left')
    for a,title in zip(ax,titles):
        a.set_title(title);a.set_xlabel('Reference phase time (s)');a.grid(alpha=.2);a.legend(fontsize=7)
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


def common_window(task,before,after,env):
    """Compare shared simulated phases without filling failed trajectories."""
    count=min(len(before['trajectory']),len(after['trajectory']))
    b=before['trajectory'][:count];a=after['trajectory'][:count]
    assert all(abs(x['phase_t']-y['phase_t'])<1e-8 for x,y in zip(b,a))
    key='joint_error_deg' if task=='lift' else 'object_error_m'
    scale=1 if task=='lift' else 100
    result=dict(shared_steps=count,shared_seconds=count*.01,
        before_mean_error=float(np.mean([r[key] for r in b]))*scale,
        after_mean_error=float(np.mean([r[key] for r in a]))*scale)
    if task=='bend':
        surface=env.box_support_pos[2]+env.box_support_half_size[2]+env.box_half_size[2]
        pairs=[(x,y) for x,y in zip(b,a) if min(x['height_m'],y['height_m'])>surface+.04]
        result['shared_air_seconds']=len(pairs)*.01
        for side,label in [(0,'before'),(1,'after')]:
            result[label+'_nonhand_vertical_n']=float(np.mean([abs(pair[side]['other_vertical_n']) for pair in pairs])) if pairs else None
            result[label+'_bilateral_fraction']=float(np.mean([min(pair[side]['normal_n'])>1. for pair in pairs])) if pairs else None
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'report.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    body=['<header><span class="tag">负载微调 · 第三版</span><h1>哑铃后段跟踪与双手摩擦搬箱</h1>',
        f'<p>更新于 {time.strftime("%Y-%m-%d %H:%M:%S %z")}。视频、曲线和原始评估均已内嵌，可离线打开。</p>',
        '<p>视频依次显示参考动作、改进前策略、改进后策略及改进后手部近景。相同测试起点、相同物体质量、相同参考相位进行对齐。提前失败后保持最终姿态并明确标红；该段不是后续成功运动。</p>',
        '<p>哑铃使用右手刚性连接，完整保留 2 kg 负载。箱子保持自由刚体，依靠手部碰撞与摩擦承重；没有连接、吸附、外加箱子支撑或独立募集层。箱子仍从约 31 cm 的窄台面、双手已就位的参考动作开始。</p></header>',
        '<section><h2>为什么调整</h2><p>哑铃原肘角奖励为 0.3 exp(−e²/0.20²)，在较大误差下奖励很弱。保留该精细跟踪项，新增 0.7w/(1+(e/0.35)²) 和角速度项 0.1/(1+(ė/2)²)；后段权重 w 平滑升至 1.5。采样采用 40% 全动作、60% 第 160–259 帧。</p>',
        '<p>箱子原奖励重视掌面位置，对持续夹紧和前臂承重缺少直接约束。新增接触力观测：双手世界系合力、法向力及非手部合力共 11 维，输入从 2827 扩为 2838，新增输入权重初始化为零并验证初始策略输出不变，肌肉动作仍为 416 维。</p>',
        '<p>每手目标夹紧力 N* = 1.5mg/(2μ)，μ=0.8；双手奖励为 0.75√[clip(N右/N*,0,1)clip(N左/N*,0,1)]。另加箱子位置、速度的宽范围跟踪和真实摩擦支撑奖励，并惩罚非手部合力及过量夹紧。采样采用 30% 全动作、40% 提起前段、30% 第 110–159 帧。</p>',
        '<p>非手部合力由完整人体接触合力减去左右手合力得到；当前模型的全身根节点为 Full Body，pelvis 不包含上肢。CPU 原始接触力和 GPU 独立世界读数均已校验。传感器约定见 <a href="https://mujoco.readthedocs.io/en/3.4.0/XMLreference.html#sensor-contact">MuJoCo 官方说明</a>。</p>',
        '<p>非手部竖直力曲线保留符号：正值表示向上承托箱子，负值表示向下压迫箱子。验收使用绝对值的均值，因此同时约束这两类接触；前臂压箱也会使手部需要提供更大的向上摩擦力。</p>',
        '<p>保持原验收标准：哑铃完整运行、肘关节 MAE ≤20°、根部偏差 ≤25 cm；箱子完整运行、位置误差 &lt;10 cm、空中双手接触比例 &gt;85%、平均摩擦支撑 &gt;0.7mg、非手部竖直支撑 &lt;0.1mg、手箱穿透 &lt;3 mm。训练回报因奖励变化不能直接作为新旧效果比较。</p></section>']
    raw={};media=OUT/'presentation';media.mkdir(exist_ok=True)
    for task,title in [('lift','右手 2 kg 哑铃'),('bend','双手摩擦搬箱')]:
        folder=OUT/task;baseline=folder/'evaluations/baseline';selection=folder/'latest_evaluation.json'
        newfolder=None;after=None
        if selection.exists():
            selected=json.loads(selection.read_text());newfolder=folder/'evaluations'/selected['label']
            after=json.loads((newfolder/'results.json').read_text())
            matched=folder/'evaluations'/f'baseline_{after["mass_kg"]:g}kg'
            original=json.loads((baseline/'results.json').read_text())
            if original['mass_kg']!=after['mass_kg'] and not (matched/'results.json').exists():
                evaluate(task,matched.name,original['checkpoint'],after['mass_kg'])
            if matched.exists():baseline=matched
        if not (baseline/'results.json').exists():continue
        before=json.loads((baseline/'results.json').read_text())
        if after:assert before['mass_kg']==after['mass_kg'],'Before/after mass must match'
        raw[task]=dict(before=before,after=after)
        cfg=OmegaConf.load(folder/'base_config.yaml');cfg.experiment.env_params.payload_mass_kg=before['mass_kg'];env=make_cpu(cfg)
        if task=='bend' and cfg.experiment.env_params.reward_type=='BoxPoseRefinementReward':
            body.append('<p>本版本另加箱子四元数姿态奖励：0.8 / (1 + (姿态角误差 / 0.40 rad)²)。位置和姿态都按每一帧参考轨迹评估；姿态误差曲线使用左右轴显示，不改变既有质量门槛。</p>')
        state=json.loads((folder/'background_status.json').read_text()) if (folder/'background_status.json').exists() else {}
        body.append(f'<section><h2>{title} · {before["mass_kg"]:g} kg</h2>')
        body.append(f'<p>已评估模型：{html.escape(after["label"]) if after else "新模型待评估"}。改进前完成 {before["complete_count"]}/4、达标 {before["quality_count"]}/4；'+
            (f'改进后完成 {after["complete_count"]}/4、达标 {after["quality_count"]}/4。' if after else '当前先展示真实基线回放。')+'</p>')
        body.append('<details><summary>后台状态与 checkpoint</summary><pre>'+html.escape(json.dumps(dict(status=state,before=before['checkpoint'],after=after['checkpoint'] if after else None),ensure_ascii=False,indent=2))+'</pre></details>')
        if after:
            key='joint_mae_deg' if task=='lift' else 'peak_object_error_m'
            scale=1 if task=='lift' else 100
            title_metric='肘角 MAE (°)' if task=='lift' else '最大箱位误差 (cm)'
            body.append(f'<table><tr><th>起始帧</th><th>完成步数：前 → 后 / 预期</th><th>{title_metric}：前 → 后</th><th>达标：前 → 后</th></tr>')
            for b in before['records']:
                a=next(r for r in after['records'] if r['start']==b['start'])
                body.append(f'<tr><td>{b["start"]}</td><td>{b["steps"]} → {a["steps"]} / {a["expected_steps"]}</td><td>{b[key]*scale:.2f} → {a[key]*scale:.2f}</td><td>{"是" if b["quality_pass"] else "否"} → {"是" if a["quality_pass"] else "否"}</td></tr>')
            body.append('</table><p>各轨迹指标按实际运行段计算；发生提前失败时，不能仅凭平均误差或支撑力判断改善。下方同时列出双方都实际运行的共同相位区间。</p>')
        body.append(f'<label>测试起点 <select onchange="choose(\'{task}\',this.value)">')
        for r in before['records']:body.append(f'<option value="{r["start"]}">第 {r["start"]} 帧</option>')
        body.append('</select></label>')
        for idx,b in enumerate(before['records']):
            start=b['start'];a=next((r for r in after['records'] if r['start']==start),None) if after else None
            token=hashlib.sha256(json.dumps(['footer-outside-v2',before['checkpoint'],after['checkpoint'] if after else None,start]).encode()).hexdigest()[:10]
            video=media/f'{task}_{start}_{token}.mp4';plot=media/f'{task}_{start}_{token}.png'
            if not video.exists():comparison_video(task,start,env,b,a,baseline,newfolder,video)
            if not plot.exists():plot_comparison(task,start,env,b,a,plot)
            body.append(f'<article class="episode {task}" data-start="{start}" style="display:{"block" if idx==0 else "none"}">')
            body.append(f'<video controls preload="metadata" src="{uri(video,"video/mp4")}"></video><img src="{uri(plot,"image/png")}">')
            body.append('<table><tr><th>指标</th><th>改进前</th><th>改进后</th></tr>')
            fields=[('完成状态','termination'),('实际运行步数','steps')]
            fields+=([('肘关节 MAE (°)','joint_mae_deg'),('根部最大偏差 (m)','peak_root_error_m')] if task=='lift' else
                [('箱子最大位置误差 (m)','peak_object_error_m'),('空中双手接触比例','bilateral_contact_fraction'),
                 ('空中平均竖直摩擦力 (N)','airborne_mean_friction_vertical_n'),('空中平均 |非手部竖直力| (N)','airborne_other_body_vertical_n'),('最大手箱穿透 (m)','peak_penetration_m')])
            if task=='bend':fields += [('箱子姿态平均误差 (°)','mean_object_orientation_error_deg'),('箱子姿态最大误差 (°)','peak_object_orientation_error_deg')]
            fields+=[('质量达标','quality_pass')]
            def fmt(x):return f'{x:.5g}' if isinstance(x,float) else html.escape(str(x))
            for label,key in fields:body.append(f'<tr><td>{label}</td><td>{fmt(b[key])}</td><td>{fmt(a[key]) if a else "待评估"}</td></tr>')
            body.append('</table>')
            if a:
                failures=[]
                if a['termination']!='reference_end':failures.append('提前结束')
                if task=='lift':
                    if a['joint_mae_deg']>20:failures.append('肘角 MAE 超过 20°')
                    if a['peak_root_error_m']>.25:failures.append('根部偏差超过 25 cm')
                else:
                    if a['peak_object_error_m']>=.1:failures.append('箱子位置误差达到或超过 10 cm')
                    if a['bilateral_contact_fraction']<=.85:failures.append('双手持续接触不足')
                    if a['airborne_mean_friction_vertical_n']<=.7*after['mass_kg']*9.81:failures.append('手部摩擦承重不足')
                    if a['airborne_other_body_vertical_n']>=.1*after['mass_kg']*9.81:failures.append('非手部竖直支撑过多')
                    if a['peak_penetration_m']>=.003:failures.append('手箱穿透达到或超过 3 mm')
                body.append('<p><strong>当前评估：'+('尚未达标；'+ '、'.join(failures) if failures else '该起点通过全部原验收标准')+'。</strong></p>')
                shared=common_window(task,b,a,env)
                unit='°' if task=='lift' else 'cm'
                body.append(f'<p>共同实际运行区间 {shared["shared_seconds"]:.2f} s，平均跟踪误差：{shared["before_mean_error"]:.2f} → {shared["after_mean_error"]:.2f} {unit}。</p>')
                if task=='bend':
                    body.append(f'<p>共同区间内双方均离台的时长 {shared["shared_air_seconds"]:.2f} s；')
                    if shared['shared_air_seconds']:
                        body.append(f'平均非手部竖直支撑：{shared["before_nonhand_vertical_n"]:.2f} → {shared["after_nonhand_vertical_n"]:.2f} N；双手持续接触比例：{shared["before_bilateral_fraction"]:.1%} → {shared["after_bilateral_fraction"]:.1%}。')
                    else:body.append('无共同离台样本，支撑指标不作对比。')
                    body.append('</p>')
            body.append('</article>')
        env.stop();body.append('</section>')
    body.append('<section><h2>原始结果与复现材料</h2>')
    for name,data in raw.items():body.append(f'<details><summary>{name} 完整逐帧数据</summary><pre>'+html.escape(json.dumps(data,ensure_ascii=False,indent=2))+'</pre></details>')
    for p in [OUT/'lift/base_config.yaml',OUT/'bend/base_config.yaml',OUT/'audit/warp_contact_feedback.json',OUT/'audit/sampling_tests.log']:
        if p.exists():body.append('<details><summary>'+html.escape(str(p.relative_to(OUT)))+'</summary><pre>'+html.escape(p.read_text())+'</pre></details>')
    body.append('<p>未内嵌的大文件：各任务 training/*/checkpoints 下的 Orbax checkpoint、reference.npz 和 evaluations/* 的逐帧状态 NPZ。HTML 所在目录另存源码快照及完整训练日志。后台运行说明见 OPERATIONS.txt。</p></section>')
    css='body{font:16px/1.7 system-ui;color:#17324a;background:#edf2f5;max-width:1500px;margin:30px auto;padding:0 24px}section,header{background:white;padding:28px;margin-bottom:24px;border-radius:12px}h1,h2{line-height:1.25}video,img{width:100%;margin:12px 0}pre{white-space:pre-wrap;max-height:380px;overflow:auto;font:12px/1.5 monospace;background:#f2f5f8;padding:18px}table{width:100%;border-collapse:collapse}th,td{padding:8px;text-align:left;border-bottom:1px solid #dce3e9}select{font:inherit;margin:12px;padding:6px}.tag{color:#237dad}details{margin:12px 0}'
    js="function choose(task,start){document.querySelectorAll('.episode.'+task).forEach(e=>{e.style.display=e.dataset.start===start?'block':'none';e.querySelectorAll('video').forEach(v=>v.pause())})}"
    doc='<!doctype html><html lang="zh"><meta charset="utf-8"><title>负载微调：策略回放与接触力对比</title><style>'+css+'</style>'+''.join(body)+'<script>'+js+'</script></html>'
    target=OUT/'report.html';temp=target.with_suffix('.tmp');temp.write_text(doc);temp.replace(target)
    print('REPORT',target,len(doc),flush=True)


if __name__=='__main__':main()
