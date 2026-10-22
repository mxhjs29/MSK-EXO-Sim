"""MSK/device/terrain/object composition with compiled ownership metadata."""
from dataclasses import dataclass
from .composition import build_msk_device_terrain_spec, ComponentNames
from .ownership import resolve_ownership
from .registry import OBJECT_BUILDERS
from .validation import validate_scene


@dataclass
class BuiltScene:
    composition: object
    model: object
    ownership: dict


def build_scene(scene):
    validate_scene(scene)
    human = next(e for e in scene.entities if e.kind == "msk")
    device = next((e for e in scene.entities if e.kind in {"device", "robot"}), None)
    composition = build_msk_device_terrain_spec(human.model_key, device.model_key if device else None, scene.terrain)
    names = {human.entity_id: composition.components["msk"]}
    if device:
        # Legacy ownership groups all non-MSK joints under robot. In a warehouse
        # this also includes free cartons; keep those out of the device agent.
        original = composition.components["robot"]
        prefix = scene.terrain.get("terrain", "flat") + "_"
        names[device.entity_id] = ComponentNames(**{
            field: tuple(n for n in getattr(original, field) if not n.startswith(prefix))
            for field in ("bodies", "joints", "actuators", "sensors")
        })
    names["terrain"] = ComponentNames(**{
        field: tuple(v.name for v in getattr(composition.spec, field)
                     if v.name and v.name != "world" and not any(v.name in getattr(n, field) for n in names.values()))
        for field in ("bodies", "joints", "actuators", "sensors")
    })
    for entity in scene.entities:
        if entity.kind != "object":
            continue
        if entity.model_key not in OBJECT_BUILDERS:
            raise ValueError(f"Unknown object builder: {entity.model_key}")
        spec = composition.spec
        fields = {k: {v.name for v in getattr(spec, k)} for k in ("bodies", "joints", "actuators", "sensors")}
        OBJECT_BUILDERS[entity.model_key](spec, entity.parameters)
        names[entity.entity_id] = ComponentNames(**{k: tuple(v.name for v in getattr(spec, k) if v.name and v.name not in prior) for k, prior in fields.items()})
    for key, value in scene.physics_settings.items():
        if not hasattr(composition.spec.option, key):
            raise ValueError(f"Unknown physics option: {key}")
        setattr(composition.spec.option, key, value)
    model = composition.spec.compile()
    return BuiltScene(composition, model, {key: resolve_ownership(model, key, value) for key, value in names.items()})
