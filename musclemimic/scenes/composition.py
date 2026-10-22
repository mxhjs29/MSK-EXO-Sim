"""Headless composition of MSK, optional device, and terrain models."""

from __future__ import annotations

import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping as ABCMapping
from typing import Any, Mapping

import assist_sim
import mujoco
import myo_sim
from assist_sim import load_combined, load_msk
from myoassist_terrains import build_terrain
from myoassist_terrains.config import ExternalTerrainConfig, config_from_dict


_STANDALONE_ROBOT_MODELS = {
    "unitree_g1": "unitree_g1/g1.xml",
    "unitree_go2": "unitree_go2/go2.xml",
    "franka_emika_panda": "franka_emika_panda/franka_panda_lift.xml",
}


@dataclass(frozen=True)
class ComponentNames:
    """Names owned by one component after composition."""

    joints: tuple[str, ...] = ()
    actuators: tuple[str, ...] = ()
    sensors: tuple[str, ...] = ()
    bodies: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompositionResult:
    """An editable final spec plus immutable component ownership metadata."""

    spec: mujoco.MjSpec
    components: Mapping[str, ComponentNames]
    name_map: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    _scratch: tempfile.TemporaryDirectory = field(repr=False, compare=False, default=None)


def _names(spec: mujoco.MjSpec) -> ComponentNames:
    return ComponentNames(
        joints=tuple(item.name for item in spec.joints if item.name),
        actuators=tuple(item.name for item in spec.actuators if item.name),
        sensors=tuple(item.name for item in spec.sensors if item.name),
        bodies=tuple(item.name for item in spec.bodies if item.name and item.name != "world"),
    )


def _absolutize_files(root: ET.Element, base_dir: Path) -> None:
    compiler = root.find("compiler")
    mesh_dir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
    texture_dir = Path(compiler.get("texturedir", "")) if compiler is not None else Path()
    for element in root.iter():
        raw = element.get("file")
        if not raw:
            continue
        path = Path(raw)
        if path.is_absolute():
            element.set("file", path.as_posix())
            continue
        candidates = []
        if element.tag == "texture" and texture_dir != Path():
            candidates.append((base_dir / texture_dir / path).resolve())
        if element.tag == "mesh" and mesh_dir != Path():
            candidates.append((base_dir / mesh_dir / path).resolve())
        candidates.append((base_dir / path).resolve())
        for candidate in candidates:
            if candidate.exists():
                element.set("file", candidate.as_posix())
                break


def _namespace_external_scene(scene_root: ET.Element, prefix: str) -> None:
    """Namespace an external MJCF scene before copying it into the model."""
    for element in scene_root.iter():
        name = element.get("name")
        if name:
            element.set("name", f"{prefix}_{name}")

    reference_attributes = (
        "body1", "body2", "joint", "joint1", "joint2", "site", "mesh",
        "material", "texture", "hfield", "skin", "tendon", "geom1",
        "geom2", "camera", "actuator", "body", "objname",
    )
    for element in scene_root.iter():
        for attribute in reference_attributes:
            value = element.get(attribute)
            if value:
                element.set(attribute, f"{prefix}_{value}")
        for attribute in ("class", "childclass"):
            value = element.get(attribute)
            if value:
                element.set(attribute, f"{prefix}_{value}")


def _translate_external_scene(scene_root: ET.Element, offset: tuple[float, float, float]) -> None:
    """Translate every top-level object in an external scene."""
    worldbody = scene_root.find("worldbody")
    if worldbody is None:
        raise ValueError("External terrain scene has no <worldbody> section.")

    def shifted_pos(element: ET.Element) -> None:
        raw = element.get("pos", "0 0 0")
        values = [float(value) for value in raw.split()]
        if len(values) != 3:
            raise ValueError(f"Expected a 3D pos on <{element.tag}>, got {raw!r}")
        element.set("pos", " ".join(f"{value + delta:.12g}" for value, delta in zip(values, offset)))

    for child in worldbody:
        if child.tag in {"body", "geom", "site", "camera", "light"}:
            shifted_pos(child)


def _make_noncentral_warehouse_boxes_visual(scene_root: ET.Element) -> None:
    """Keep only shelf0 cartons movable in the composed warehouse.

    ``shelf_box_00`` through ``shelf_box_05`` are the six cartons on the
    central shelf.  All other cartons remain visible but are welded static,
    massless, and non-colliding like rack decoration.
    """
    worldbody = scene_root.find("worldbody")
    if worldbody is None:
        raise ValueError("Warehouse scene has no <worldbody> section.")
    for body in worldbody.findall("body"):
        name = body.get("name", "")
        if not name.startswith("shelf_box_"):
            continue
        try:
            box_index = int(name.rsplit("_", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid warehouse box name: {name!r}") from exc
        if box_index < 6:
            continue
        for joint in list(body.findall("joint")):
            body.remove(joint)
        geom = body.find("geom")
        if geom is None:
            raise ValueError(f"Warehouse box {name!r} has no visual geom.")
        geom.set("mass", "0")
        geom.set("contype", "0")
        geom.set("conaffinity", "0")


def _append_qpos0_values(model_root: ET.Element, external_model: mujoco.MjModel) -> None:
    """Keep external free-body poses when the base model has keyframes."""
    keyframe = model_root.find("keyframe")
    if keyframe is None or external_model.nq == 0:
        return
    values = " ".join(f"{value:.12g}" for value in external_model.qpos0)
    for key in keyframe.findall("key"):
        key.set("qpos", " ".join((*key.get("qpos", "").split(), *values.split())))


def _namespace_standalone_robot(robot_root: ET.Element, prefix: str) -> None:
    """Make standalone robot names and references safe in a shared MJCF."""
    # MuJoCo permits an unnamed ``<freejoint/>``.  A composed model needs a
    # stable name for every joint so trajectory metadata can stay string-keyed.
    for index, freejoint in enumerate(robot_root.iter("freejoint")):
        if freejoint.get("name") is None:
            freejoint.set("name", f"freejoint_{index}")

    asset = robot_root.find("asset")
    if asset is not None:
        for element in asset:
            if element.get("name") is None and element.get("file"):
                element.set("name", Path(element.get("file", "")).stem)

    collision_classes: set[str] = set()

    def collect_collision_classes(element: ET.Element, under_collision: bool = False) -> None:
        current_class = element.get("class")
        is_collision = under_collision or current_class == "collision" or (
            current_class is not None and "collision" in current_class
        )
        if current_class and is_collision:
            collision_classes.add(current_class)
        for child in element.findall("default"):
            collect_collision_classes(child, is_collision)

    defaults = robot_root.find("default")
    if defaults is not None:
        collect_collision_classes(defaults)
    for geom in robot_root.iter("geom"):
        if geom.get("group") == "3" or geom.get("class") in collision_classes:
            # A human model can provide a global non-colliding geom default.
            # Explicit flags keep the standalone robot collidable after merge.
            geom.set("contype", "1")
            geom.set("conaffinity", "1")

    for element in robot_root.iter():
        if element.get("name"):
            element.set("name", f"{prefix}_{element.get('name')}")
        for attribute in ("class", "childclass"):
            if element.get(attribute):
                element.set(attribute, f"{prefix}_{element.get(attribute)}")

    reference_attributes = (
        "body1", "body2", "joint", "joint1", "joint2", "site", "mesh",
        "material", "texture", "hfield", "skin", "tendon", "geom1", "geom2",
        "camera", "actuator",
    )
    for element in robot_root.iter():
        for attribute in reference_attributes:
            value = element.get(attribute)
            if value:
                element.set(attribute, f"{prefix}_{value}")


def _append_robot_keyframe_values(model_root: ET.Element, robot_root: ET.Element) -> None:
    """Extend MSK keyframes with the robot's initial qpos and controls."""
    robot_key = robot_root.find("keyframe/key")
    if robot_key is None:
        # Complete exported scenes such as the robosuite Panda + table model
        # can rely on MuJoCo's qpos0/ctrl0 initialization instead of a named
        # keyframe.  Their body poses and joint refs still define the correct
        # initial state, so there is nothing to append to the MSK keyframes.
        return
    robot_qpos = robot_key.get("qpos", "").split()
    robot_ctrl = robot_key.get("ctrl", "").split()
    keyframe = model_root.find("keyframe")
    if keyframe is None:
        return
    for key in keyframe.findall("key"):
        key.set("qpos", " ".join((*key.get("qpos", "").split(), *robot_qpos)))
        if robot_ctrl:
            key.set("ctrl", " ".join((*key.get("ctrl", "").split(), *robot_ctrl)))


def _inject_standalone_robot(model_root: ET.Element, robot_path: Path, robot_key: str) -> None:
    """Insert a robot as an independent world-body root, without welds."""
    robot_root = ET.parse(robot_path).getroot()
    _absolutize_files(robot_root, robot_path.parent)
    _append_robot_keyframe_values(model_root, robot_root)
    _namespace_standalone_robot(robot_root, robot_key)

    model_default = _find_or_make(model_root, "default")
    for child in robot_root.findall("default/*"):
        model_default.append(child)
    model_asset = _find_or_make(model_root, "asset")
    for child in robot_root.findall("asset/*"):
        model_asset.append(child)
    model_worldbody = model_root.find("worldbody")
    if model_worldbody is None:
        raise ValueError("Composed model has no <worldbody> section.")
    for child in robot_root.findall("worldbody/*"):
        model_worldbody.append(child)
    for section in ("actuator", "sensor", "tendon", "equality", "contact"):
        source = robot_root.find(section)
        if source is None:
            continue
        target = _find_or_make(model_root, section)
        for child in source:
            target.append(child)


def _find_or_make(root: ET.Element, tag: str, index: int = 0) -> ET.Element:
    element = root.find(tag)
    if element is None:
        element = ET.Element(tag)
        root.insert(index, element)
    return element


def _inject_terrain(model_root: ET.Element, terrain_root: ET.Element) -> None:
    model_asset = _find_or_make(model_root, "asset")
    terrain_asset = terrain_root.find("asset")
    if terrain_asset is not None:
        for child in list(terrain_asset):
            model_asset.append(child)

    model_worldbody = model_root.find("worldbody")
    terrain_worldbody = terrain_root.find("worldbody")
    if model_worldbody is None or terrain_worldbody is None:
        raise ValueError("Both the composed model and terrain must contain <worldbody>.")
    for element in terrain_root.iter("geom"):
        element.set("contype", "1")
        element.set("conaffinity", "1")
    for child in list(terrain_worldbody):
        model_worldbody.append(child)


def _owned(final: ComponentNames, baseline: ComponentNames, *, baseline_side: bool) -> ComponentNames:
    def select(values: tuple[str, ...], base_values: tuple[str, ...]) -> tuple[str, ...]:
        base = set(base_values)
        return tuple(value for value in values if (value in base) == baseline_side)

    return ComponentNames(
        joints=select(final.joints, baseline.joints),
        actuators=select(final.actuators, baseline.actuators),
        sensors=select(final.sensors, baseline.sensors),
        bodies=select(final.bodies, baseline.bodies),
    )


def build_msk_device_terrain_spec(
    msk_key: str,
    device_key: str | None = None,
    terrain: dict[str, Any] | None = None,
) -> CompositionResult:
    """Compose a training model using the same ordering as ``compose_viewer.py``.

    ``device_key=None`` is a supported MSK+terrain configuration.  MSK names
    always come from ``myo_sim``; final names not owned by that baseline are
    assigned to the optional robot/device component.
    """
    device_key = device_key or None
    terrain = {"terrain": "flat"} if terrain is None else terrain
    # Hydra/OmegaConf supplies ``DictConfig`` objects, which implement the
    # mapping protocol but are not plain ``dict`` instances.  Normalize all
    # mapping-like configs before passing them to the terrain builder.
    if not isinstance(terrain, ABCMapping):
        raise TypeError("terrain must be a mapping or None")
    terrain = dict(terrain)

    baseline = _names(myo_sim.load_spec(msk_key))
    scratch = tempfile.TemporaryDirectory(prefix="musclemimic_compose_")
    scratch_path = Path(scratch.name)
    base_xml = scratch_path / "msk_device.xml"
    terrain_assets = scratch_path / "terrain_assets"
    final_xml = scratch_path / "composed.xml"

    try:
        if device_key is None or device_key in _STANDALONE_ROBOT_MODELS:
            load_msk(msk_key, export_xml=str(base_xml))
        else:
            load_combined(msk_key, device_key, export_xml=str(base_xml))

        model_root = ET.parse(base_xml).getroot()
        _absolutize_files(model_root, base_xml.parent)
        if device_key in _STANDALONE_ROBOT_MODELS:
            assist_models = Path(assist_sim.__file__).resolve().parent / "models"
            _inject_standalone_robot(
                model_root,
                assist_models / _STANDALONE_ROBOT_MODELS[device_key],
                device_key,
            )

        terrain_config = config_from_dict(terrain)
        terrain_spec = build_terrain(terrain_config, output_dir=terrain_assets)
        terrain_root = ET.fromstring(terrain_spec.to_xml())
        if isinstance(terrain_config, ExternalTerrainConfig):
            _absolutize_files(terrain_root, terrain_config.scene_file.parent)
            if terrain_config.terrain == "warehouse":
                _make_noncentral_warehouse_boxes_visual(terrain_root)
            _namespace_external_scene(terrain_root, terrain_config.terrain)
            _translate_external_scene(terrain_root, terrain_config.offset)
            external_model = mujoco.MjModel.from_xml_string(ET.tostring(terrain_root, encoding="unicode"))
            _append_qpos0_values(model_root, external_model)
        else:
            terrain_spec.compile()
            _absolutize_files(terrain_root, terrain_assets)
        _inject_terrain(model_root, terrain_root)

        ET.indent(model_root, space="  ")
        final_xml.write_text(ET.tostring(model_root, encoding="unicode"), encoding="utf-8")
        spec = mujoco.MjSpec.from_file(str(final_xml))
        final = _names(spec)
        msk = _owned(final, baseline, baseline_side=True)
        robot = _owned(final, baseline, baseline_side=False) if device_key else ComponentNames()
        components = MappingProxyType(
            {"msk": msk, "robot": robot, "terrain": ComponentNames()}
        )
        return CompositionResult(
            spec=spec,
            components=components,
            name_map=MappingProxyType({"msk": MappingProxyType({})}),
            _scratch=scratch,
        )
    except Exception:
        scratch.cleanup()
        raise


def export_composition(result: CompositionResult, target: str | Path) -> None:
    """Export a composition for diagnostics without changing its live spec."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    source = Path(result._scratch.name) / "composed.xml"
    shutil.copyfile(source, target)
