import hashlib
from pathlib import Path
import numpy as np

def digest(path):
    path = Path(path)
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    for p in files:
        if path.is_dir():
            h.update(str(p.relative_to(path)).encode())
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
    return h.hexdigest()


def model_fingerprint(model):
    # Integration timestep is tracked separately; retain actual topology and mechanics.
    h = hashlib.sha256()
    for name in ['names', 'body_pos', 'body_quat', 'body_mass', 'body_inertia',
                 'body_parentid', 'jnt_type', 'jnt_axis', 'jnt_pos', 'jnt_range',
                 'dof_damping', 'dof_armature', 'eq_data', 'eq_type', 'eq_obj1id',
                 'eq_obj2id', 'actuator_gainprm', 'actuator_gear', 'actuator_trnid']:
        value = np.asarray(getattr(model, name))
        h.update(name.encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()
