from musclemimic.environments.humanoids.myofullbody_assist import MyoFullBodyAssistEnv, MjxMyoFullBodyAssistEnv
from musclemimic.retargeting.gait120_myolegs26 import correct_dependent_ranges

class MyoFullBodyGait120SynergyEnv(MyoFullBodyAssistEnv):
    # Keep the maintained family name: legacy site mapping dispatches on it.
    def __init__(self, **kwargs):
        if kwargs.get('msk_key', 'myolegs26') != 'myolegs26':
            raise ValueError('This experiment only supports myolegs26')
        kwargs['msk_key'] = 'myolegs26'
        super().__init__(**kwargs)

    def _apply_spec_changes(self, spec):
        spec = super()._apply_spec_changes(spec)
        self.synergy_range_corrections = correct_dependent_ranges(spec)
        return spec

class MjxMyoFullBodyGait120SynergyEnv(MyoFullBodyGait120SynergyEnv, MjxMyoFullBodyAssistEnv):
    mjx_enabled = True

MyoFullBodyGait120SynergyEnv.register()

MjxMyoFullBodyGait120SynergyEnv.register()
