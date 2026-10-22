from musclemimic.tasks.box_lift.goals import BoxFrictionGoal
from musclemimic.core.sensors.hand_contact import ContactReadout

class BoxRefinementGoal(BoxFrictionGoal):
    def _init_from_mj(self,env,model,data,current_obs_size):
        super()._init_from_mj(env,model,data,current_obs_size)
        self.contact_readout=ContactReadout(model);self.weight=env.payload_mass_kg*9.81

    @property
    def dim(self):return super().dim+11

    def get_obs_and_update_state(self,env,model,data,carry,backend):
        obs,carry=super().get_obs_and_update_state(env,model,data,carry,backend)
        f=self.contact_readout(data,backend)
        values=backend.concatenate((f['hand'].reshape(-1),f['normal'],f['other']))/self.weight
        return backend.concatenate((obs,backend.clip(values,-10.,10.))),carry

BoxRefinementGoal.register()
