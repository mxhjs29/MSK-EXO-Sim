from musclemimic.tasks.box_lift.terminal import BoxFrictionTerminal
from musclemimic.core.sensors.hand_contact import ContactReadout, CONTACT_SLOTS

class BoxRefinementTerminal(BoxFrictionTerminal):
    def __init__(self,env,**kwargs):
        super().__init__(env,**kwargs);self.contact_readout=ContactReadout(env._model)

    def _is_absorbing_compat(self,env,obs,info,data,carry,backend):
        done,carry=super()._is_absorbing_compat(env,obs,info,data,carry,backend)
        overflow=backend.any(self.contact_readout(data,backend)['count']>CONTACT_SLOTS)
        return done|overflow,carry

BoxRefinementTerminal.register()
