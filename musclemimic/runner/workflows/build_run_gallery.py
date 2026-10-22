from musclemimic.utils.paths import workspace_root, config_root
from pathlib import Path
import json
import numpy as np
import subprocess
root=workspace_root() / 'outputs/fullbody_run_h/reference_data'
reports=[]
for summary in sorted(root.glob('*/reference_summary.json')):
    label=summary.parent.name
    report=json.loads((root/label/'reference_summary.json').read_text())
    analysis=next((root/'amass_cache/msk').rglob(report.get('retarget_source',report['source'])+'_analysis.npz'))
    errors=np.load(analysis)['pos_error']
    report['retarget_mean_error_cm']=float(errors.mean()*100)
    report['retarget_p95_error_cm']=float(np.percentile(errors,95)*100)
    (root/label/'reference_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    reports.append(report)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(root/label/'reference.mp4'),'-vf','select=eq(n\\,8)','-frames:v','1',str(root/label/'preview.jpg')],check=True)
cards=''.join(f'''<article><h2>{r['label']} · {r['source']}</h2><p>{r.get('description', '直线跑')} · 原始区间 {r.get('source_start_s', 0):.2f}–{r.get('source_end_s', r['duration_s']):.2f} 秒</p><p>平均速度 {r['mean_speed_m_s']:.2f} m/s · 重映射 {r['duration_s']:.2f} 秒 · {r['frames']} 帧</p><video controls loop muted playsinline preload="metadata" poster="{r['label']}/preview.jpg" src="{r['label']}/reference.mp4"></video><p>重映射点位误差：平均 {r['retarget_mean_error_cm']:.2f} cm / P95 {r['retarget_p95_error_cm']:.2f} cm</p><a href="{r['label']}/reference.mp4">打开单条视频</a></article>''' for r in sorted(reports, key=lambda r: (r["label"] < "E", r["label"])))
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>跑步参考轨迹</title><style>body{font:16px system-ui;background:#111a27;color:#eef3fa;margin:24px auto;max-width:1500px;padding:0 20px}h1{margin-bottom:8px}p{color:#bfccda;line-height:1.6}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}article{background:#1b2a3b;padding:18px;border-radius:12px}h2{font-size:20px}video{width:100%;border-radius:8px}button,select{font:inherit;padding:9px 14px;margin:8px 8px 18px 0}a{color:#88c8ff}@media(max-width:800px){main{grid-template-columns:1fr}}</style><h1>跑步参考动作 H</h1><p>已选定 H：KIT 向前跑，含起步和停下。全身肌骨模型，无手指关节、无外骨骼，保留腕关节。每条视频展示两个视角。<br>这是重映射参考回放，尚未训练。轨迹由原始连续动作截取，未经循环拼接。播放器循环会在片尾跳回起点，动作相位未对齐。</p><button onclick="restart()">全部从头播放</button><button onclick="videos.forEach(v=>v.pause())">全部暂停</button><label>速度 <select onchange="videos.forEach(v=>v.playbackRate=Number(this.value))"><option value="1">正常速度</option><option value="0.5">0.5 倍慢放</option><option value="0.25">0.25 倍慢放</option></select></label><main>'''+cards+'''</main><p>训练配置已固定为 H。</p><script>const videos=[...document.querySelectorAll('video')];function restart(){videos.forEach(v=>{v.currentTime=0;v.play().catch(()=>{})})}</script></html>'''
(root/'index.html').write_text(html)
(root/'candidates.json').write_text(json.dumps(reports,indent=2)+'\n')
for r in reports: print(r['label'],r['source'],r['frames'],r['retarget_mean_error_cm'])
