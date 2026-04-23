"""Parallel scan over (f, epsilon) to find an excitable regime.

Neumann BC on both axes (default simulator.step behaviour), single planar
bolus IC, rigid mesh. For each (f, ε) combination we integrate up to
`--total` T_0 and record four classifiers:

    u_max_trace   : max u over space at each dumped time
    v_max_trace   : max v over space at each dumped time
    u_mid_xt      : u along the horizontal midline (time × x)
    label         : one of
        'dead'    — wave decays quickly, u_max < 0.1 after the first dip
        'excite'  — single traveling wave then true rest (u_max below 0.2
                    after the first pass AND v_max drops below 0.1)
        'train'   — repeated traveling wave fronts visible as diagonal
                    stripes in the x-t map (2+ distinct peaks in u_max,
                    front positions advance in time)
        'uniform' — synchronous oscillation across the whole domain
                    (u_max fluctuates but spatial RMS of u at any frame
                    is << u_max, meaning the field is nearly flat)

A composite image is saved: one row per f, one column per ε, showing the
x-t midline of u plus the detected label.

Usage:
    python runs/scan_fe.py --n-grid 64 --total 100 --f 0.8 0.9 1.0 1.1 1.2 \
                           --epsilon 0.05 0.1 0.2 0.354 0.5
"""
from __future__ import annotations
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from glsm.params import DEFAULT
from glsm import simulator as sim
from generate_datasets import IC_BUILDERS, make_rigid_state, LAMBDA_IN, DT_OUTER, REACT_SUB


def classify(u_traj: np.ndarray, v_traj: np.ndarray,
             t_arr: np.ndarray) -> str:
    """Rough label for the dynamics of a single trajectory."""
    u_max = u_traj.max(axis=(1, 2))
    v_max = v_traj.max(axis=(1, 2))
    # Spatial flatness at each frame: spatial std / spatial range, averaged.
    u_spatial_range = u_traj.max(axis=(1, 2)) - u_traj.min(axis=(1, 2))
    u_spatial_std = u_traj.std(axis=(1, 2))
    # Only measure after first bolus has dissipated (skip t < 5 T_0)
    mask = t_arr > 5.0
    if not mask.any():
        return "dead"
    # Dead: u_max essentially decays and stays below 0.1
    if u_max[mask].max() < 0.1:
        return "dead"
    # Count "peaks" in u_max (local maxima above 0.3) after t=5
    peaks = 0
    for i in range(1, len(u_max) - 1):
        if t_arr[i] < 5.0:
            continue
        if u_max[i] > u_max[i - 1] and u_max[i] >= u_max[i + 1] and u_max[i] > 0.3:
            peaks += 1
    # Uniform vs traveling: average "flatness" during high-u frames
    high_frames = u_max > 0.3
    if high_frames.any():
        # If the field is nearly spatially uniform when excited, it's
        # synchronous oscillation.
        ratio = np.median(u_spatial_std[high_frames] /
                          (u_spatial_range[high_frames] + 1e-9))
    else:
        ratio = 0.0
    # Excitable: 1 peak + drops back to rest (u_max final < 0.15, v_max < 0.15)
    if peaks <= 1 and u_max[-1] < 0.15 and v_max[-1] < 0.15:
        return "excite"
    # Distinguish traveling wave train from synchronous
    if ratio > 0.15:
        return "train"       # spatial structure present at peak → traveling
    else:
        return "uniform"     # spatially flat at peak → synchronous osc


def run_one(f: float, eps: float, Ny: int, Nx: int, n_dumps: int,
            dump_every: int, seed: int) -> dict:
    """Run a single (f, ε) bolus trajectory and return summary arrays."""
    p = replace(DEFAULT, f=f, epsilon=eps)
    u0, v0 = IC_BUILDERS["planar"](Ny, Nx, seed=seed)
    state = make_rigid_state(Ny, Nx, u0, v0, p=p, lambda_init=LAMBDA_IN)
    t0 = time.time()
    snaps = sim.run(state, p, dt=DT_OUTER, n_steps=n_dumps * dump_every,
                    snapshot_every=dump_every, reaction_substeps=REACT_SUB,
                    pin_left_wall=False, block_left_u_flux=False,
                    rigid_mesh=True, d_scale=1.0)
    u_traj = snaps["u"].astype(np.float32)
    v_traj = snaps["v"].astype(np.float32)
    label = classify(u_traj, v_traj, snaps["t"])
    return dict(
        f=f, epsilon=eps,
        u_max=u_traj.max(axis=(1, 2)),
        v_max=v_traj.max(axis=(1, 2)),
        u_midline=u_traj[:, Ny // 2, :],
        t=snaps["t"],
        label=label,
        elapsed=time.time() - t0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", nargs="+", type=float,
                    default=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5])
    ap.add_argument("--epsilon", nargs="+", type=float,
                    default=[0.03, 0.06, 0.1, 0.2, 0.354, 0.5])
    ap.add_argument("--n-grid", type=int, default=64, dest="n_grid")
    ap.add_argument("--total", type=float, default=100.0)
    ap.add_argument("--save-dt", type=float, default=1.0, dest="save_dt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=str,
                    default=str(ROOT / "plots" / "scan_fe.png"))
    args = ap.parse_args()

    Ny = Nx = args.n_grid
    dump_every = int(round(args.save_dt / DT_OUTER))
    n_dumps = int(round(args.total / args.save_dt))

    combos = [(f, eps) for f in args.f for eps in args.epsilon]
    print(f"Scanning {len(combos)} (f, ε) combos on {Ny}×{Nx} grid "
          f"for {args.total} T_0 ({n_dumps+1} frames each), "
          f"{args.workers} workers")

    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, f, eps, Ny, Nx, n_dumps, dump_every,
                             args.seed): (f, eps) for f, eps in combos}
        for fut in as_completed(futs):
            r = fut.result()
            results[(r["f"], r["epsilon"])] = r
            print(f"  f={r['f']:.2f} ε={r['epsilon']:.3f} → {r['label']:8s} "
                  f"({r['elapsed']:.1f}s)", flush=True)
    print(f"total elapsed {time.time()-t0:.1f}s")

    # Plot grid of midline x-t maps
    f_vals = sorted(set(k[0] for k in results))
    eps_vals = sorted(set(k[1] for k in results))
    nr, nc = len(f_vals), len(eps_vals)
    fig, axes = plt.subplots(nr, nc, figsize=(2.2 * nc, 1.9 * nr))
    if nr == 1: axes = axes[None, :]
    if nc == 1: axes = axes[:, None]
    label_color = {"dead": "#888", "excite": "#2ca02c",
                   "train": "#1f77b4", "uniform": "#d62728"}
    for i, f in enumerate(f_vals):
        for j, eps in enumerate(eps_vals):
            ax = axes[i, j]
            r = results[(f, eps)]
            ax.imshow(r["u_midline"].T, aspect="auto", cmap="inferno",
                      extent=[0, r["t"][-1], Nx, 0], vmin=0, vmax=0.92)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"f={f:.2f} ε={eps:.2g}\n[{r['label']}]",
                          fontsize=9,
                          color=label_color.get(r["label"], "k"))
    fig.suptitle(f"Yashin-Oreg rigid-mesh bolus, Neumann BC — "
                 f"{Ny}×{Nx}, {args.total} T_0", fontsize=12)
    fig.tight_layout()
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=115, bbox_inches="tight")
    print(f"wrote {out_path}")

    # Text summary
    print("\n--- classification grid ---")
    header = "  f\\ε | " + " | ".join(f"{e:.3g}" for e in eps_vals)
    print(header)
    print("-" * len(header))
    for f in f_vals:
        row = f"{f:.2f} | " + " | ".join(
            f"{results[(f, e)]['label']:7s}" for e in eps_vals)
        print(row)


if __name__ == "__main__":
    main()
