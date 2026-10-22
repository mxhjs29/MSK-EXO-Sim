# Third-party components

This repository retains a modified LocoMuJoCo implementation and the `myo_sim`,
`assist_sim`, and `myoassist_terrains` source/model packages. Preserve attribution,
license files and notices within each component when redistributing it. The root
license does not replace separate terms attached to model meshes or dependencies.

SMPL/MANO assets under `local/assets/smpl/` are included in the Git publication
scope at the user’s request, with PKL/NPZ models managed by Git LFS. Their bundled
license notices remain in place and state the permissions required for third-party
redistribution. These assets are excluded from Python wheel/sdist archives.
Original and processed motion datasets and learned checkpoints remain excluded
from public source and distributions.
This migration does not grant new rights to external datasets or model assets.

The GMR parameter adapter in `musclemimic/retargeting/gmr_params.py` is derived
from the pinned GMR backend (commit `1b4eb9864360fa822b1290f34592da6393a4d496`).
Copyright 2025 Yanjieze; its MIT license is retained in
`musclemimic/retargeting/GMR_LICENSE.txt`. The GMR solver remains an optional
external dependency; the musculoskeletal target is provided by `myo_sim`.
