import numpy as np
from musclemimic.core.goals.trajectory import GoalTrajMimic

class PayloadGoalTrajMimic(GoalTrajMimic):
    def __init__(self, *args, source_start_frame=0, source_frame_count=0, **kwargs):
        self.source_start_frame=int(source_start_frame)
        self.source_frame_count=int(source_frame_count)
        super().__init__(*args, **kwargs)

    def get_obs_and_update_state(self, *args, **kwargs):
        goal, carry=super().get_obs_and_update_state(*args, **kwargs)
        if self.enable_motion_phase and self.source_frame_count:
            phase=(carry.traj_state.subtraj_step_no+self.source_start_frame)/self.source_frame_count
            if isinstance(goal,np.ndarray):
                goal=goal.copy();goal[-1]=phase
            else:goal=goal.at[-1].set(phase)
        return goal,carry

PayloadGoalTrajMimic.register()
