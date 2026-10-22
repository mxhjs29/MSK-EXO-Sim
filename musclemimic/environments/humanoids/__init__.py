from .myofullbody_assist import MyoFullBodyAssistEnv, MjxMyoFullBodyAssistEnv


# Register only the maintained Assist environments.  ``myofullbody.py`` is
# still used as an implementation base by ``myofullbody_assist.py``; the
# standalone legacy environments are intentionally no longer public registry
# entries.
MyoFullBodyAssistEnv.register()
MjxMyoFullBodyAssistEnv.register()

from .myofullbody_payload import MyoFullBodyPayloadEnv, MjxMyoFullBodyPayloadEnv
MyoFullBodyPayloadEnv.register()
MjxMyoFullBodyPayloadEnv.register()

from .myofullbody_box_friction import MyoFullBodyBoxFrictionEnv, MjxMyoFullBodyBoxFrictionEnv
from . import myofullbody_payload_refinement
