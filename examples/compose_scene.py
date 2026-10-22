"""Run from the repository root: python -m examples.compose_scene."""
from musclemimic.scenes import SceneSpec, EntitySpec, build_scene


def main():
    scene = build_scene(SceneSpec(
        scene_id="human_with_hippo",
        entities=(EntitySpec("human", "msk", "myofullbody"), EntitySpec("device", "device", "Hippo_L1")),
        terrain={"terrain": "flat"},
    ))
    print("nq, nv, nu:", scene.model.nq, scene.model.nv, scene.model.nu)
    print("human actuators:", scene.ownership["human"].actuator_indices)
    print("device actuators:", scene.ownership["device"].actuator_indices)


if __name__ == "__main__":
    main()
