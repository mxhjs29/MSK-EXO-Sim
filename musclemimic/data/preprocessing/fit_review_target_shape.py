"""Fit SMPL-H shape to the exact delivered no-fingers MyoFullBody binary."""
from musclemimic.data.preprocessing import retarget_selected_myofullbody as engine
from musclemimic.data.preprocessing.dataset_paths import OUTPUT_ROOT, DATASET, SCRIPTS, CONFIG
import json
from pathlib import Path
ROOT=engine.ROOT
OUT=OUTPUT_ROOT/'shape_fit_100hz/gmr'
BASE=DATASET/'converted_dataset/approved_continuous_100hz/full/new'

def main():
    import torch,numpy as np,mujoco
    from unittest.mock import patch
    from loco_mujoco.smpl.retargeting import _ensure_gmr_imports,get_smpl_model_path
    _ensure_gmr_imports()
    from general_motion_retargeting.utils.smpl import SMPLH_Parser
    from general_motion_retargeting.utils import shape_fitting as fit
    torch.set_num_threads(1)
    body=SMPLH_Parser(model_path=get_smpl_model_path(),gender='neutral',use_pca=False,num_betas=16).eval()
    model=mujoco.MjModel.from_binary_path(str(BASE/'model.mjb'))
    class Factory:
        @staticmethod
        def from_xml_path(path):return model
    class Adapter:
        MjModel=Factory
        def __getattr__(self,k):return getattr(mujoco,k)
    cfg=json.loads((CONFIG).read_text())
    with patch.object(fit,'mj',Adapter()):
        targets,joint_names,rotations,human_names=fit.get_robot_tpose_targets(str(BASE/'model.mjb'),'myofullbody',cfg,include_rotations=True)
    indices=fit.get_smpl_tpose_indices(joint_names)
    shape,scale,pos,rot,offset_z,height_scale,metrics=fit.fit_smpl_shape_to_robot(body,targets,indices,target_rotations=rotations,iterations=1000,lr=.001,device='cpu',verbose=False,robot_type='myofullbody')
    assert np.isfinite(shape.numpy()).all() and np.isfinite(scale.numpy()).all() and float(scale[0])>0 and height_scale>0
    assert metrics['final_loss']<metrics['initial_loss']
    offsets=fit.compute_alignment_offsets(body,shape,scale,indices,human_names,targets,rotations,robot_type='myofullbody')
    OUT.mkdir(parents=True,exist_ok=True)
    fit.save_fitted_shape(shape,scale,pos,rot,offset_z,height_scale,metrics,str(OUT/'fitted_shape.pkl'),human_joint_names=human_names,local_offsets=offsets)
    report={'iterations':1000,'initial_mean_squared_position_error_m2':metrics['initial_loss'],'final_mean_squared_position_error_m2':metrics['final_loss'],'initial_rms_position_error_m':float(np.sqrt(metrics['initial_loss'])),'final_rms_position_error_m':float(np.sqrt(metrics['final_loss'])),'scale':float(scale[0]),'height_scale':height_scale,'offset_z_m':offset_z,'betas':shape.numpy().tolist(),'target':'exact delivered model.mjb; library myofullbody T-pose','note':'Library final_loss is squared distance in m^2, not meters; reference-pose fit, not motion validation.'}
    (OUT/'shape_fit_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
