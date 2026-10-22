"""CPU-only, trial-held-out NMF audit of S101 walking EMG (no policy training).

E[muscle, time] ~= W[muscle, synergy] @ H[synergy, time].
NMF uses train-only RMS channel weighting; saved W is in MVC envelope units.
Held-out H is fitted to held-out EMG with W fixed: this measures reconstruction,
not open-loop EMG prediction or the performance of a walking policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from musclemimic.retargeting.gait120_emg import CHANNELS, MUSCLES, Gait120Mat

GROUP_NAMES = list(dict.fromkeys(MUSCLES))
GROUP_LABELS = ["VL", "RF", "VM", "TA", "BF", "ST", "GM", "GL", "SOL", "PL", "PB"]
SPLITS = {"train": ["Trial01", "Trial02", "Trial03"],
          "validation": ["Trial04"], "test": ["Trial05"]}


def project_nonnegative(basis, signal):
    """Fit only coefficients, keeping the supplied muscle basis fixed."""
    return np.column_stack([nnls(basis, col, maxiter=1000)[0] for col in signal.T])


def fit_nmf(signal, rank, *, seed=42, max_iter=4000, tol=1e-8):
    """Hierarchical alternating least squares for a nonnegative M x T matrix."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or not np.isfinite(signal).all() or (signal < 0).any():
        raise ValueError("NMF requires a finite, nonnegative matrix")
    if not 1 <= rank <= min(signal.shape) or signal.max() <= 0:
        raise ValueError("Invalid rank or empty signal")
    rng = np.random.default_rng(seed)
    scale = np.sqrt(signal.mean() / rank)
    basis = (rng.random((signal.shape[0], rank)) + .1) * scale
    coeff = (rng.random((rank, signal.shape[1])) + .1) * scale
    previous = np.inf
    converged = False
    for iteration in range(1, max_iter + 1):
        gram, cross = basis.T @ basis, basis.T @ signal
        for k in range(rank):
            coeff[k] = np.maximum(0, coeff[k] + (cross[k] - gram[k] @ coeff) / max(gram[k, k], 1e-20))
        gram, cross = coeff @ coeff.T, signal @ coeff.T
        for k in range(rank):
            basis[:, k] = np.maximum(0, basis[:, k] + (cross[:, k] - basis @ gram[:, k]) / max(gram[k, k], 1e-20))
        norms = np.maximum(np.linalg.norm(basis, axis=0), 1e-20)
        basis /= norms
        coeff *= norms[:, None]
        if iteration % 20 == 0:
            loss = float(np.sum((signal - basis @ coeff) ** 2))
            if iteration >= 200 and np.isfinite(previous) and abs(previous - loss) <= tol * max(previous, 1e-12):
                converged = True
                break
            previous = loss
    # Exact NNLS coefficient projection gives the same evaluation rule on each split.
    coeff = project_nonnegative(basis, signal)
    loss = float(np.sum((signal - basis @ coeff) ** 2))
    return basis, coeff, {"seed": seed, "iterations": iteration, "converged": converged, "sse": loss}


def reconstruction_metrics(signal, reconstruction):
    error = np.sum((signal - reconstruction) ** 2, axis=1)
    energy = np.sum(signal ** 2, axis=1)
    centered_energy = np.sum((signal - signal.mean(axis=1, keepdims=True)) ** 2, axis=1)
    if np.any(energy <= 0):
        raise ValueError("Zero-energy EMG channel")
    correlations = []
    for measured, fitted in zip(signal, reconstruction):
        correlations.append(float(np.corrcoef(measured, fitted)[0, 1])
                            if min(np.std(measured), np.std(fitted)) > 1e-12 else None)
    return {
        "global_uncentered_vaf": float(1 - error.sum() / energy.sum()),
        "minimum_muscle_uncentered_vaf": float(np.min(1 - error / energy)),
        "per_muscle_uncentered_vaf": (1 - error / energy).tolist(),
        "per_muscle_centered_r2": [float(1-e/v) if v > 1e-20 else None for e, v in zip(error, centered_energy)],
        "per_muscle_pearson_r": correlations,
        "mean_pearson_r": float(np.mean(correlations)) if all(x is not None for x in correlations) else None,
        "per_muscle_rmse_mvc": np.sqrt(error / signal.shape[1]).tolist(),
    }


def choose_rank(rows, *, global_threshold=.90, muscle_threshold=.75):
    """Test metrics deliberately play no role in rank selection."""
    for row in sorted(rows, key=lambda x: x["rank"]):
        if all(row[split]["global_uncentered_vaf"] >= global_threshold and
               row[split]["minimum_muscle_uncentered_vaf"] >= muscle_threshold
               for split in ("train", "validation")):
            return row["rank"]
    return None


def load_clips(source_dir, phase_samples):
    processed = Gait120Mat(source_dir / "ProcessedData.mat")
    raw = Gait120Mat(source_dir / "RawData.mat")
    fs, marker_fs = raw.data["EMGs_info"]["fs"], raw.data["Markers_info"]["fs"]
    phase = np.linspace(0, 1, phase_samples)
    clips, arrays, source_names = [], [], None
    for trial_name in sorted(name for name in processed.data["LevelWalking"] if name.startswith("Trial")):
        trial = processed.data["LevelWalking"][trial_name]
        split = next(key for key, trials in SPLITS.items() if trial_name in trials)
        for step_name in sorted(name for name in trial if name.startswith("Step")):
            values, names = processed.table(trial[step_name]["EMGs_norm"])
            envelope, env_names = processed.table(trial[step_name]["EMGs_env"])
            np.testing.assert_allclose(values, envelope / trial["MVCs"], rtol=1e-12, atol=1e-12)
            if len(names) != len(CHANNELS) or env_names != names or (source_names is not None and source_names != names):
                raise ValueError("Unexpected EMG channel ordering")
            source_names = names
            if np.any(values < 0):
                raise ValueError("Negative EMG envelope")
            bounds = np.asarray(raw.data["LevelWalking"][trial_name][step_name]["TargetFrame"], dtype=int)
            if len(values) != (bounds[1] - bounds[0] + 1) * (fs / marker_fs):
                raise ValueError("EMG/marker duration mismatch")
            grouped = np.column_stack([values[:, [i for i, muscle in enumerate(MUSCLES) if muscle == name]].mean(axis=1)
                                       for name in GROUP_NAMES])
            arrays.append(np.vstack([np.interp(phase, np.linspace(0, 1, len(values)), column) for column in grouped.T]))
            clips.append({"id": f"S101_{trial_name}_{step_name}", "trial": trial_name, "step": step_name,
                          "split": split, "raw_samples": len(values), "marker_target_frames": bounds.tolist(),
                          "start_time_s": float((bounds[0]-1)/marker_fs),
                          "last_emg_time_s": float((bounds[0]-1)/marker_fs + (len(values)-1)/fs)})
    return np.stack(arrays), clips, source_names, float(fs)


def model_audit():
    import mujoco
    from musclemimic.models import build_msk_device_terrain_spec
    composition = build_msk_device_terrain_spec("myolegs", None, {"terrain": "flat"})
    model = composition.spec.compile()
    names = [model.actuator(i).name for i in range(model.nu)]
    mapping = []
    for muscle in GROUP_NAMES:
        index = names.index(muscle)
        mapping.append({"muscle": muscle, "actuator_id": index,
                        "activation_address": int(model.actuator_actadr[index]),
                        "activation_dimension": int(model.actuator_actnum[index])})
    return {"msk_key": "myolegs", "nq": model.nq, "nv": model.nv, "nu": model.nu, "na": model.na,
            "muscle_actuators": int(np.sum(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)),
            "measured_mapping": mapping, "unmeasured_actuators": [name for name in names if name not in GROUP_NAMES],
            "scope": "Compiled standalone model and checked names; no walking rollout, policy training or new retargeting."}


def make_plots(output, rows, rank, basis, coeffs, arrays, clips):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ranks = [row["rank"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for split in ("train", "validation"):
        for ax, metric in zip(axes, ("global_uncentered_vaf", "minimum_muscle_uncentered_vaf")):
            ax.plot(ranks, [row[split][metric] * 100 for row in rows], "o-", label=split)
            ax.set(xlabel="Number of synergies", ylabel="Uncentered VAF (%)", xticks=ranks)
            ax.grid(alpha=.25)
    for ax, threshold in zip(axes, (90, 75)):
        ax.axhline(threshold, color="grey", ls="--")
        if rank is not None:
            ax.axvline(rank, color="black", ls=":", label=f"Selected K={rank}")
        ax.legend()
    axes[0].set_title("Global reconstruction")
    axes[1].set_title("Worst reconstructed muscle")
    fig.savefig(output / "rank_selection.png", dpi=160)
    plt.close(fig)
    if rank is None:
        return
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(basis, aspect="auto", cmap="viridis", vmin=0)
    ax.set(yticks=range(11), yticklabels=GROUP_LABELS, xticks=range(rank),
           xticklabels=[f"S{i+1}" for i in range(rank)], title="S101 right-leg NMF basis (MVC units; column max = 1)")
    fig.colorbar(im, ax=ax, label="Relative muscle weight")
    fig.savefig(output / "synergy_weights.png", dpi=160)
    plt.close(fig)
    test_index = next(i for i, clip in enumerate(clips) if clip["split"] == "test")
    observed, predicted = arrays[test_index], basis @ coeffs[test_index]
    phase = np.linspace(0, 100, observed.shape[1])
    fig, axes = plt.subplots(4, 3, figsize=(13, 9), constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        if i == 11:
            ax.axis("off")
            continue
        ax.plot(phase, observed[i], label="Measured envelope")
        ax.plot(phase, predicted[i], "--", label="Fixed-W reconstruction")
        ax.set(title=GROUP_LABELS[i], xlabel="Clip phase (%)", ylabel="MVC fraction")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Held-out Trial05 / K={rank}; coefficients fitted to test EMG, NOT policy predictions")
    fig.savefig(output / "heldout_reconstruction.png", dpi=160)
    plt.close(fig)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    source_dir = Path(args.source_dir)
    protocol = {
        "subject": "S101", "task": "LevelWalking", "splits": SPLITS,
        "rank_candidates": list(range(1, 9)), "restarts": args.restarts, "seed": 42,
        "phase_samples_per_clip": 101, "solver": "HALS + fixed-basis SciPy NNLS projection",
        "solver_max_iter": 4000, "solver_relative_loss_tolerance": 1e-8,
        "selection": "Smallest K with train AND validation global uncentered VAF >= .90 AND every-muscle VAF >= .75; no test-based selection.",
        "preprocessing": "Published MVC-normalized envelopes; average SM/SL; linearly resample each clip to 101 equal phase points; equal clip weighting; divide channels by training-only RMS during fitting.",
        "phase_definition": "Normalized clip interval, not independently verified heel-strike phase; retain physical timestamps for future trajectory alignment.",
        "split_limitation": "This split is independent within this NMF audit; previous PPO experiments already used Trial01 and inspected subject S101.",
        "interpretation": "Reconstruction only: validation/test coefficients use their own EMG, with training W and RMS fixed. This is not EMG forecasting or a policy evaluation.",
        "training_started": False,
        "sources": {name: {"path": str(source_dir / name), "sha256": hashlib.sha256((source_dir/name).read_bytes()).hexdigest()}
                    for name in ("ProcessedData.mat", "RawData.mat")},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    # Record the protocol before observing any rank-sweep or test results.
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    arrays, clips, names, fs = load_clips(source_dir, 101)
    matrices = {split: np.concatenate([arrays[i] for i, clip in enumerate(clips) if clip["split"] == split], axis=1)
                for split in SPLITS}
    rms = np.sqrt(np.mean(matrices["train"] ** 2, axis=1))
    if np.any(rms < 1e-12):
        raise ValueError("Constant-zero training channel")
    train_scaled = matrices["train"] / rms[:, None]
    rows, bases = [], {}
    for rank in protocol["rank_candidates"]:
        best, restart_records = None, []
        for restart in range(args.restarts):
            basis, coeff, record = fit_nmf(train_scaled, rank, seed=42 + 100 * rank + restart)
            restart_records.append(record)
            if best is None or record["sse"] < best[2]["sse"]:
                best = (basis, coeff, record)
        scaled_basis = best[0]
        basis_mvc = rms[:, None] * scaled_basis
        basis_mvc /= np.maximum(basis_mvc.max(axis=0), 1e-20)
        # Deterministic presentation order by dominant muscle (NMF labels have no inherent order).
        basis_mvc = basis_mvc[:, np.argsort(np.argmax(basis_mvc, axis=0), kind="stable")]
        row = {"rank": rank, "restart_records": restart_records, "best_fit": best[2]}
        for split in ("train", "validation"):
            coeff = project_nonnegative(basis_mvc / rms[:, None], matrices[split] / rms[:, None])
            row[split] = reconstruction_metrics(matrices[split], basis_mvc @ coeff)
        rows.append(row)
        bases[rank] = basis_mvc
        print(json.dumps({"rank": rank, "train_vaf": row["train"]["global_uncentered_vaf"],
                          "val_vaf": row["validation"]["global_uncentered_vaf"],
                          "val_min_muscle_vaf": row["validation"]["minimum_muscle_uncentered_vaf"]}), flush=True)
    selected = choose_rank(rows)
    metrics, coeffs, basis = {}, None, None
    if selected is not None:
        basis = bases[selected]
        coeffs = np.stack([project_nonnegative(basis / rms[:, None], array / rms[:, None]) for array in arrays])
        for split in SPLITS:
            ids = [i for i, clip in enumerate(clips) if clip["split"] == split]
            metrics[split] = reconstruction_metrics(matrices[split], np.concatenate([basis @ coeffs[i] for i in ids], axis=1))
        np.savez_compressed(output / "synergy_prior.npz", W_mvc=basis, H_by_clip=coeffs,
                            emg_by_clip=arrays, train_rms=rms, clip_phase=np.linspace(0, 1, 101),
                            clip_ids=[clip["id"] for clip in clips], splits=[clip["split"] for clip in clips],
                            muscle_names=GROUP_NAMES, muscle_labels=GROUP_LABELS)
    # A fixed training-mean envelope quantifies how much uncentered VAF can come from the DC level.
    baseline = {split: reconstruction_metrics(matrix, np.broadcast_to(matrices["train"].mean(axis=1, keepdims=True), matrix.shape))
                for split, matrix in matrices.items()}
    result = {"selected_rank": selected, "training_started": False, "clips": clips,
              "source_channel_names": names, "source_sample_rate_hz": fs,
              "muscle_names": GROUP_NAMES, "muscle_labels": GROUP_LABELS,
              "rank_sweep": rows, "selected_metrics": metrics, "training_mean_baseline": baseline,
              "model_audit": model_audit(),
              "limitations": ["One subject, six short clips; not a population or long-horizon walking validation.",
                              "Only 11 measured right-leg muscles; 69 of the 80 lower-body actuators are unmeasured.",
                              "EMG envelopes are excitation proxies, not directly calibrated MuJoCo activation states.",
                              "High uncentered VAF is not proof of correct phase/correlation; centered R2 and per-muscle r are included.",
                              "Zero activation lies in every nonnegative synergy cone; a cone-distance reward alone cannot prevent zero-activation collapse.",
                              "Test H was reconstructed from test EMG, not produced by PPO; these r values cannot be ranked against previous policy-EMG r.",
                              "No lower-body reference conversion, policy training, dynamics rollout or neural-synergy identification claim."]}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    make_plots(output, rows, selected, basis, coeffs, arrays, clips)
    print(json.dumps({"output": str(output), "selected_rank": selected, "selected_metrics": metrics}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="outputs/gait120_emg_comparison_v1/data")
    parser.add_argument("--output", default="outputs/gait120_synergy_feasibility_v1")
    parser.add_argument("--restarts", type=int, default=8)
    args = parser.parse_args()
    if args.restarts < 1:
        parser.error("--restarts must be positive")
    run(args)
