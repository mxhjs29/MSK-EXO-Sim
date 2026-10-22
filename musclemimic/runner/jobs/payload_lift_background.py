"""Continue the validated 2 kg dumbbell task in a detached tmux session."""
import fcntl
import json
import os
import time
import traceback
from pathlib import Path
from musclemimic.runner.jobs import payload_background as worker

OUT=worker.OUT
AUDIT=OUT/'audit/lift_continuation'
LABEL='target_2kg_random_300'
SOURCE=OUT/'lift/training/target_2kg_refine_3/checkpoints/target_2kg_refine_3/checkpoint_10'


def main():
    os.chdir(worker.ROOT);AUDIT.mkdir(parents=True,exist_ok=True)
    worker.AUDIT=AUDIT
    lock=(AUDIT/'background.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    worker.status('supervisor',phase='running',pid=os.getpid(),session='payload_lift_v2',
                  mass_kg=2,num_envs=4096,updates=300,start_mode='random',source=str(SOURCE))
    folder=OUT/'lift/training'/LABEL
    args=['musclemimic.runner.workflows.payload_finetune','train','--task','lift',
          '--mass','2','--envs','4096','--updates','300','--label',LABEL,
          '--start-mode','random','--checkpoint',str(SOURCE),'--lr','1e-5']
    for attempt in range(3):
        if (folder/'completed.json').exists():break
        if (AUDIT/'STOP_BACKGROUND').exists():
            worker.status('supervisor',phase='stopped');return
        if worker.existing_trainers(LABEL):
            raise RuntimeError('Duplicate trainer detected; refusing to launch')
        try:worker.command(args,LABEL,gpu=2);break
        except RuntimeError:
            if attempt==2:raise
            time.sleep(30)
    cp=folder/'checkpoints'/LABEL/'checkpoint_300'
    label=LABEL+'_evaluation'
    worker.command(['musclemimic.runner.workflows.payload_finetune','evaluate','--task','lift',
        '--mass','2','--checkpoint',str(cp),'--label',label],'evaluation')
    result=json.loads((OUT/'lift/evaluations'/label/'results.json').read_text())
    worker.command(['musclemimic.runner.workflows.payload_report','--task','lift','--label',label],'video')
    worker.command(['musclemimic.runner.workflows.payload_report'],'report')
    worker.status('supervisor',phase='complete' if result['all_quality_pass'] else 'needs_quality_review',
                  checkpoint=str(cp),evaluation=str(OUT/'lift/evaluations'/label/'results.json'))


if __name__=='__main__':
    try:main()
    except Exception:
        worker.status('supervisor',phase='error',traceback=traceback.format_exc());raise
