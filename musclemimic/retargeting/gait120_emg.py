"""Read the published Gait120 MATLAB tables without MATLAB.

The decoder is deliberately restricted to Gait120's MCOS v4 table layout;
unknown layouts fail closed. Downloaded tables are never modified.
"""
from pathlib import Path
import io
import json
import hashlib
import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import varmats_from_mat

CHANNELS = ["VL", "RF", "VM", "TA", "BF", "ST", "GM", "GL", "SM", "SL", "PL", "PB"]
MUSCLES = ["vaslat_r", "recfem_r", "vasmed_r", "tibant_r", "bflh_r", "semiten_r",
           "gasmed_r", "gaslat_r", "soleus_r", "soleus_r", "perlong_r", "perbrev_r"]


class Gait120Mat:
    def __init__(self, path):
        self.path = Path(path)
        self.data = loadmat(self.path, simplify_cells=True)
        workspace = self.data["__function_workspace__"].tobytes()
        header = self.path.read_bytes()[:124] + workspace[:4]
        variables = varmats_from_mat(io.BytesIO(header + workspace[8:]))
        wrapper = loadmat(variables[0][1], simplify_cells=True)["__function_workspace__"][0, 0]
        self.pool = wrapper.MCOS["arr"][0]
        metadata = self.pool[0].tobytes()
        offsets = np.frombuffer(metadata[:40], dtype="<u4")
        if tuple(offsets[:2]) != (4, 8):
            raise ValueError("Unsupported MCOS version")
        expected = b"data\x00ndims\x00nrows\x00rownames\x00nvars\x00varnames\x00props\x00table\x00"
        if not metadata[40:int(offsets[2])].startswith(expected):
            raise ValueError("Unsupported MCOS property schema")
        self.objects = np.frombuffer(metadata[int(offsets[4]):int(offsets[5])], "<u4").reshape(-1, 6)
        self.properties = np.frombuffer(metadata[int(offsets[5]):int(offsets[6])], "<u4")

    def table(self, opaque):
        reference = np.asarray(opaque["arr"][0]).ravel()
        if reference.size != 6 or tuple(reference[1:4]) != (2, 1, 1):
            raise ValueError("Unsupported table reference")
        object_id = int(reference[4])
        row = self.objects[object_id]
        if row[0] != 1 or row[4] != row[5] or row[4] != object_id:
            raise ValueError("Unsupported table object record")
        start = 2 + (int(row[4]) - 1) * 22
        record = self.properties[start:start + 22]
        if record.size != 22 or record[0] != 7:
            raise ValueError("Unsupported table property record")
        triples = record[1:].reshape(7, 3)
        np.testing.assert_array_equal(triples[:, 0], np.arange(1, 8))
        np.testing.assert_array_equal(triples[:, 1], np.ones(7))
        values = [self.pool[int(index) + 2] for index in triples[:, 2]]
        columns, ndim, nrows, _, nvars, names, _ = values
        array = np.column_stack(columns)
        if ndim != 2 or array.shape != (int(nrows), int(nvars)) or not np.isfinite(array).all():
            raise ValueError("Invalid table dimensions or values")
        return array, list(names)


def prepare(root):
    root = Path(root)
    raw = Gait120Mat(root / "data/RawData.mat")
    processed = Gait120Mat(root / "data/ProcessedData.mat")
    trial = raw.data["LevelWalking"]["Trial01"]
    proc = processed.data["LevelWalking"]["Trial01"]
    bounds = trial["Step01"]["TargetFrame"]
    emg_raw, _ = raw.table(trial["EMGs_raw"])
    envelope, names = processed.table(proc["Step01"]["EMGs_env"])
    normalized, names2 = processed.table(proc["Step01"]["EMGs_norm"])
    assert names == names2 and len(names) == 12
    np.testing.assert_allclose(normalized, envelope / proc["MVCs"], rtol=1e-12, atol=1e-12)
    fs = raw.data["EMGs_info"]["fs"]
    fs_marker = raw.data["Markers_info"]["fs"]
    assert fs == 2000 and fs_marker == 100
    assert len(envelope) == (bounds[1] - bounds[0] + 1) * (fs // fs_marker)
    assert len(emg_raw) == (trial["TotalFrame"][1] - trial["TotalFrame"][0] + 1) * (fs // fs_marker)
    # TRC convention: marker frame 485 is t=4.84 s. Keep physical sample time.
    times = (int(bounds[0]) - 1) / fs_marker + np.arange(len(envelope)) / fs
    np.savez_compressed(root / "data/emg_reference.npz", source_time_s=times,
                        envelope=envelope, normalized=normalized,
                        channels=CHANNELS, muscle_names=MUSCLES, mvc=proc["MVCs"],
                        raw_trial=emg_raw, marker_target_frames=bounds)
    record = dict(subject="S101", task="LevelWalking", trial="Trial01", step="Step01",
                  sample_rate_hz=fs, samples=len(envelope), channels=CHANNELS,
                  source_variable_names=names, muscle_names=MUSCLES,
                  target_marker_frames=bounds.tolist(), total_marker_frames=trial["TotalFrame"].tolist(),
                  mcos_table_ids={k:int(proc["Step01"][k]["arr"][0].ravel()[4])
                                  for k in ("EMGs_env", "EMGs_norm")},
                  mvc_consistency_passed=True, primary_lag_s=0,
                  primary_groups=11, soleus_rule="mean of MVC-normalized SM and SL; one anatomical vote",
                  bf_rule="bflh_r (surface biceps femoris); bfsh is not pooled",
                  selected_by="first intact retargeted walking clip, before any policy EMG results",
                  sources={name:hashlib.sha256((root / "data" / name).read_bytes()).hexdigest()
                           for name in ("RawData.mat", "ProcessedData.mat")})
    (root / "data/emg_alignment.json").write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    import sys
    print(json.dumps(prepare(sys.argv[1]), indent=2))
