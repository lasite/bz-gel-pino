"""Build warm-start states for spiral-type initial conditions.

Analogue of CardiacEP-PINOS `.roe` start states in
`Data/OpenCARP_simulation_files/datasets_start_states/` — the reference repo
ships TWO such files (stable spiral at t=1000 ms and chaotic breakup at
t=1100 ms); subsequent dataset runs **start from those mature fields** rather
than from a raw broken-wavefront IC, so the recorded trajectory is entirely
in the steady/quasi-steady regime.

We do the same here: for each `sim_type` in {spiral_stable, chaotic_double},
run the rigid-mesh reaction-diffusion for a *warm-up* window (default 40 T_0
— 8× the spiral period, enough to wash out IC transient), then snapshot the
state at several phases and save each as `<name>_<t>T0.npz`. Each file stores
`{nodes, u, v, t}` and can be loaded via `sim.load_state(path)`.

Usage:
    python runs/build_start_states.py --ic spiral --n-phases 4 --warmup 40
    python runs/build_start_states.py --ic chaotic --n-phases 4 --warmup 40
"""
from __future__ import annotations
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from glsm.params import DEFAULT
from glsm import simulator as sim

from generate_datasets import (IC_BUILDERS, make_rigid_state,
                               LAMBDA_IN, DT_OUTER, REACT_SUB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", choices=["spiral", "chaotic"], default="spiral",
                    help="IC builder to warm up (only spiral / chaotic "
                         "self-sustain without pacemaker)")
    ap.add_argument("--n-grid", type=int, default=101, dest="n_grid")
    ap.add_argument("--warmup", type=float, default=40.0,
                    help="warm-up duration in T_0 (default 40.0)")
    ap.add_argument("--n-phases", type=int, default=4, dest="n_phases",
                    help="how many start states to save, spaced evenly in the "
                         "second half of the warm-up window so they capture "
                         "different spiral phases (default 4)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--f", type=float, default=0.9)
    ap.add_argument("--epsilon", type=float, default=None,
                    help="override Oregonator ε (default: Table I value 0.354)")
    ap.add_argument("--d-scale", type=float, default=1.0,
                    help="u-flux scaling (conmul analogue); same value "
                         "should be used for any downstream run that loads "
                         "the resulting state")
    ap.add_argument("--out-dir", type=str,
                    default=str(ROOT / "start_states"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    over = {"f": args.f}
    if args.epsilon is not None:
        over["epsilon"] = args.epsilon
    p = replace(DEFAULT, **over)
    Ny = Nx = args.n_grid
    print(f"Oregonator overrides: {over}")

    # Build IC, then run up to `warmup` T_0 and snapshot at phases t_k.
    ic_fn = IC_BUILDERS[args.ic]
    u0, v0 = ic_fn(Ny, Nx, seed=args.seed)
    state = make_rigid_state(Ny, Nx, u0, v0, p=p, lambda_init=LAMBDA_IN)

    # Snapshot times: evenly spaced in the last half of warmup so we skip
    # the bootstrap transient (tail of IC getting absorbed into the spiral).
    phase_times = np.linspace(0.5 * args.warmup, args.warmup, args.n_phases)
    # Convert to outer-step indices
    dump_every = int(round(1.0 / DT_OUTER))   # dump every 1 T_0 just for the walk
    n_steps_total = int(round(args.warmup / DT_OUTER))

    print(f"warmup {args.warmup} T_0  grid {Ny}×{Nx}  f={args.f}  "
          f"d_scale={args.d_scale}  n_phases={args.n_phases}")

    # Walk the trajectory, saving at the phase times.
    phase_step_idx = {int(round(t / DT_OUTER)): t for t in phase_times}
    t0 = time.time()
    cur = state
    for i in range(1, n_steps_total + 1):
        cur = sim.step(cur, DT_OUTER, p,
                       reaction_substeps=REACT_SUB,
                       pin_left_wall=False, block_left_u_flux=False,
                       rigid_mesh=True, d_scale=args.d_scale)
        if i in phase_step_idx:
            eps_tag = f"_ep{p.epsilon:g}"
            tag = f"{args.ic}_f{args.f}{eps_tag}_d{args.d_scale}_t{phase_step_idx[i]:.1f}T0"
            path = out_dir / f"{tag}.npz"
            sim.save_state(cur, path)
            print(f"  saved {path.name}  t={cur.t:.2f}  "
                  f"u∈[{cur.u.min():.3f},{cur.u.max():.3f}]  "
                  f"v∈[{cur.v.min():.3f},{cur.v.max():.3f}]  "
                  f"({time.time()-t0:.1f}s)")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
