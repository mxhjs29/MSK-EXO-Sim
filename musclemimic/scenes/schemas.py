"""Declarative physical composition; entities do not imply learning agents."""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EntitySpec:
    entity_id: str
    kind: str
    model_key: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AttachmentSpec:
    parent: str
    child: str
    mode: str = "model_defined"


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    entities: tuple[EntitySpec, ...]
    terrain: dict[str, Any] = field(default_factory=lambda: {"terrain": "flat"})
    attachments: tuple[AttachmentSpec, ...] = ()
    physics_settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["entities"] = tuple(EntitySpec(**e) for e in value["entities"])
        value["attachments"] = tuple(AttachmentSpec(**a) for a in value.get("attachments", ()))
        return cls(**value)
