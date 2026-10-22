"""Reject unsupported composition instead of silently dropping entities."""
import math


def validate_scene(scene):
    ids = [e.entity_id for e in scene.entities]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Scene entity IDs must be nonempty and unique")
    if any(not i for i in ids):
        raise ValueError("Empty entity ID")
    if "terrain" in ids:
        raise ValueError("The terrain entity ID is reserved for scene.terrain")
    if sum(e.kind == "msk" for e in scene.entities) != 1:
        raise ValueError("Current builder requires exactly one MSK; multi-human composition is a future extension")
    if sum(e.kind in {"device", "robot"} for e in scene.entities) > 1:
        raise ValueError("Current builder supports at most one device/robot")
    if sum(e.kind == "object" for e in scene.entities) > 1:
        raise ValueError("Built-in payloads use fixed names; at most one payload per scene")
    if any(e.kind not in {"msk", "device", "robot", "object"} for e in scene.entities):
        raise ValueError("Unknown entity kind; terrain is configured by scene.terrain")
    for entity in scene.entities:
        if entity.kind != "object" and entity.parameters:
            raise ValueError("Non-object parameters must use model registry keys or physics_settings")
        if "payload_mass_kg" in entity.parameters:
            mass = entity.parameters["payload_mass_kg"]
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError("Payload mass must be finite and positive")
    for attachment in scene.attachments:
        if attachment.parent not in ids or attachment.child not in ids or attachment.parent == attachment.child:
            raise ValueError("Attachment references unknown or identical entities")
        if attachment.mode != "model_defined":
            raise ValueError("Only model-defined attachments are supported; friction boxes remain free bodies")
        by_id = {e.entity_id: e for e in scene.entities}
        if by_id[attachment.parent].kind != "msk" or by_id[attachment.child].kind not in {"device", "object"}:
            raise ValueError("Model-defined attachments must describe an MSK and its device or payload")
