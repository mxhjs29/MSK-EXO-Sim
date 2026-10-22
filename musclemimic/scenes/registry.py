"""Object builders can be registered without changing PPO or rewards."""
from types import SimpleNamespace
from .objects.payload import add_payload
from .objects.box import add_friction_box

OBJECT_BUILDERS = {}


def register_object(name, builder):
    if name in OBJECT_BUILDERS:
        raise ValueError(f"Object builder already registered: {name}")
    OBJECT_BUILDERS[name] = builder


def _payload(kind, mass, **parameters):
    values = dict(payload_kind=kind, payload_mass_kg=mass, payload_grip_span_m=.21,
                  payload_mount_offset=(0., 0., 0.), payload_finger_posture={})
    values.update(parameters)
    return SimpleNamespace(**values)


register_object("dumbbell", lambda spec, p: add_payload(spec, _payload("dumbbell", 2., **p)))
register_object("ideal_box", lambda spec, p: add_payload(spec, _payload("box", 5., **p)))


def _box(spec, parameters):
    defaults = dict(box_half_size=(.12, .10, .12), hand_friction=.8,
                    box_support_pos=(0., 0., .1), box_support_half_size=(.20, .20, .1))
    defaults.update(parameters)
    return add_friction_box(spec, _payload("box", 5., **defaults))


register_object("friction_box", _box)
