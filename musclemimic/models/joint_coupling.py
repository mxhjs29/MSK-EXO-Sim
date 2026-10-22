"""Explicit, opt-in range correction for the Gait120 target model variant."""
import mujoco
import numpy as np
from numpy.polynomial import Polynomial


def correct_knee_coupling_ranges(spec):
    """Keep dependent knee limits consistent with their exact joint equations.

    Mutates only the supplied spec. Primary joint limits and installed assets
    are untouched. Must run before CPU/MJX model construction.
    """
    model = spec.compile()
    changes = []
    for eid in range(model.neq):
        if model.eq_type[eid] != mujoco.mjtEq.mjEQ_JOINT or not model.eq_active0[eid]:
            continue
        child, parent = int(model.eq_obj1id[eid]), int(model.eq_obj2id[eid])
        if parent < 0 or model.joint(parent).name not in {"knee_angle_r", "knee_angle_l"}:
            continue
        if not model.jnt_limited[child]:
            continue
        ref_child = model.qpos0[model.jnt_qposadr[child]]
        lo, hi = model.jnt_range[parent] - model.qpos0[model.jnt_qposadr[parent]]
        polynomial = Polynomial(model.eq_data[eid, :5])
        points = [lo, hi] + [r.real for r in polynomial.deriv().roots()
                             if abs(r.imag) < 1e-8 and lo < r.real < hi]
        values = polynomial(points) + ref_child
        expected = np.array([min(values), max(values)])
        old = model.jnt_range[child].copy()
        if expected[0] >= old[0] - 1e-10 and expected[1] <= old[1] + 1e-10:
            continue
        new = expected + np.array([-1e-9, 1e-9])
        name = model.joint(child).name
        spec.joint(name).range = new
        changes.append(dict(joint=name, parent=model.joint(parent).name,
                            old_range=old.tolist(), new_range=new.tolist()))
    return changes
