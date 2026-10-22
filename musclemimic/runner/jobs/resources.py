"""Machine-specific resource assignments live in local config or environment."""
import json
import os
from musclemimic.utils.paths import workspace_root


def gpu_for(role, default):
    override = os.environ.get("MUSCLEMIMIC_GPU_" + role.upper())
    if override is not None:
        return override
    path = workspace_root() / "local/configs/resources.json"
    if path.exists():
        return str(json.loads(path.read_text()).get("gpus", {}).get(role, default))
    return str(default)


def cpu_set_for(role):
    value = os.environ.get("MUSCLEMIMIC_CPUS_" + role.upper())
    path = workspace_root() / "local/configs/resources.json"
    if value is None and path.exists():
        value = json.loads(path.read_text()).get("cpus", {}).get(role)
    allowed = set(os.sched_getaffinity(0))
    if value is None:
        return ",".join(map(str, sorted(allowed)))
    requested = set()
    for item in str(value).split(","):
        bounds = item.split("-")
        if len(bounds) == 1:
            requested.add(int(bounds[0]))
        elif len(bounds) == 2:
            requested.update(range(int(bounds[0]), int(bounds[1]) + 1))
        else:
            raise ValueError(f"Invalid CPU set for {role}: {value}")
    if not requested or not requested <= allowed:
        raise ValueError(f"CPU set for {role} is outside the available affinity: {value}")
    return str(value)
