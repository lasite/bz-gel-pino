"""Generate a clean-spiral warm-start .npz for the Δ=0.5 deformable run.

Problem at Δ=0.5 with mechanics on from t=0: the sharp U_HIGH=0.85 step IC
creates an enormous instant osmotic gradient that, combined with the 4×
stronger nodal mobility at Δ=0.5 (4/Δ² vs 4 at Δ=1), explodes the mesh
before the spiral can organise. Fix: run rigid-mesh first so the chemistry
develops a clean spiral at Δ=0.5, then hand off to the deformable solver
from that settled state.

Writes start_states/spiral_f{f}_dx{dx}_t{t}T0.npz so generate_datasets.py
can load it via --start-state.

Usage (produces the Δ=0.5 f=0.9 warm-start used for the 4.8-rotation clean
spiral dataset in dataset/spiral_deform/Train_361_*):

    python runs/make_warm_start_dx0p5.py \\
        --rigid-dx 1.0 --dx 0.5 --settle-t 40 --f 0.9 --dt 0.001

The --rigid-dx 1.0 runs the rigid-phase spiral organisation at Δ=1 (where
the 1/Δ² flux coefficient is 4× gentler than at Δ=0.5 and single-step RK4
stays comfortably stable), then regrids node positions down to Δ=0.5 at
save time. Reaction terms are Δ-independent and u, v are element-centred,
so nothing about the chemistry changes — only the mesh spacing rescales.
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

LAMBDA_IN = 1.1
BG = 1e-4


def build_spiral_ic(Ny: int, Nx: int, u_high: float, v_high: float,
                    seed: int = 42, smooth: int = 3):
    """Broken-wavefront spiral IC: activator-excited left half + refractory
    lower half — tip at their crossing curls into a spiral.

    `smooth` controls the width (in cells) of a linear ramp at each
    excitation edge. Sharp 0/u_high steps of width 1 cell produce unphysical
    grid-scale overshoot at Δ=0.5 (16× faster effective diffusion compared to
    Δ=1 at the same grid count). A 3-cell ramp kills the highest unstable
    wavenumber while keeping the spiral tip sharp enough to curl.
    """
    rng = np.random.default_rng(seed)
    u = np.full((Ny, Nx), BG, dtype=np.float64)
    v = np.full((Ny, Nx), BG, dtype=np.float64)
    # u left half + smoothed right edge
    xl = Nx // 2
    u[:, :xl] = u_high
    for k in range(1, smooth + 1):
        if xl + k - 1 < Nx:
            u[:, xl + k - 1] = u_high * (smooth - k + 1) / (smooth + 1)
    # v bottom half + smoothed top edge
    yt = Ny // 2
    v[yt:, :] = v_high
    for k in range(1, smooth + 1):
        if yt - k >= 0:
            v[yt - k, :] = v_high * (smooth - k + 1) / (smooth + 1)
    u = np.clip(u + 0.02 * rng.standard_normal(u.shape) * u_high, 0.0, 1.5)
    return u, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", type=int, default=101)
    ap.add_argument("--dx", type=float, default=0.5,
                    help="target undeformed element edge Δ for the warm-start "
                         "node positions (i.e. the Δ that the downstream "
                         "deformable run will use).")
    ap.add_argument("--rigid-dx", type=float, default=None, dest="rigid_dx",
                    help="if set, run the rigid phase at this Δ and only "
                         "regrid node positions to --dx at the end. Use "
                         "--rigid-dx 1.0 when --dx 0.5 with f≥0.9 pushes the "
                         "rigid step past the explicit-RK4 stability edge "
                         "and hits the u,v∈[0,1.5] clamp. Reaction terms F,G "
                         "are Δ-independent, and rigid-mode diffusion only "
                         "sees Δ through the flux coefficient ∝ 1/Δ², so a "
                         "warm-start organised at Δ=1 and then regridded to "
                         "Δ=0.5 gives identical physics minus the numerical "
                         "blow-up.")
    ap.add_argument("--settle-t", type=float, default=50.0,
                    help="T₀ of rigid-mesh integration to let spiral organise")
    ap.add_argument("--dt", type=float, default=0.001,
                    help="outer dt in T₀ for the rigid phase. With the "
                         "Fortran-strict single-step RK4 reaction integrator "
                         "and f=0.9 (more excitable than the paper default "
                         "0.7), dt≥0.002 drops below the stability margin at "
                         "sharp BZ wavefronts and pins v at the 1.5 clamp. "
                         "dt=0.001 leaves a clean state (u≤0.33, v≤0.22 at "
                         "t=40 T₀); 0.002 was fine for f=0.7 but NOT for "
                         "f=0.9, so the default drops to 0.001.")
    ap.add_argument("--u-high", type=float, default=0.55,
                    help="initial activator amplitude. The paper's stationary "
                         "u* at φ=0.1045 is 0.24, and F(u,v,φ)≈(u(1-u) - f·v·(u-q)/(u+q))/ε "
                         "is sharply non-linear; starting at 0.85 (near the "
                         "(1−φ)² ≈ 0.80 saturation for u_max) triggers a "
                         "Δ⁻²-scaled osmotic kick that explodes the Δ=0.5 "
                         "deformable mesh before a spiral forms. 0.55 keeps "
                         "the seed clearly above u* but safely below the "
                         "shock regime.")
    ap.add_argument("--v-high", type=float, default=0.3)
    ap.add_argument("--f", type=float, default=None, dest="f_override",
                    help="override Oregonator stoichiometric factor. "
                         "Default (None) keeps Parameters.f=0.7. The clean "
                         "4.8-rotation Δ=0.5 dataset was produced with "
                         "--f 0.9, because 0.9 is more excitable than 0.7 "
                         "and yields a spiral tip that survives the "
                         "mechanics-induced volume oscillations.")
    ap.add_argument("--epsilon", type=float, default=None, dest="epsilon_override",
                    help="override ε (default 0.354 from Table I).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rigid_dx = args.rigid_dx if args.rigid_dx is not None else args.dx

    overrides = {"dx": rigid_dx}
    if args.f_override is not None:       overrides["f"] = args.f_override
    if args.epsilon_override is not None: overrides["epsilon"] = args.epsilon_override
    p = replace(DEFAULT, **overrides)
    Ny = Nx = args.n_grid

    u0, v0 = build_spiral_ic(Ny, Nx, u_high=args.u_high, v_high=args.v_high)
    spacing_rigid = LAMBDA_IN * rigid_dx
    ks, ls = np.meshgrid(np.arange(Ny + 1), np.arange(Nx + 1), indexing="ij")
    nodes = np.stack([ls * spacing_rigid, ks * spacing_rigid],
                     axis=-1).astype(np.float64)
    st = sim.State(nodes=nodes, u=u0.copy(), v=v0.copy())

    n_steps = int(round(args.settle_t / args.dt))
    print(f"Rigid-phase: Δ={rigid_dx}  dt={args.dt}  n_steps={n_steps}  "
          f"total={args.settle_t} T₀  grid {Ny}×{Nx}  "
          f"(target output Δ={args.dx})")
    t0 = time.time()
    snaps = sim.run(st, p, dt=args.dt, n_steps=n_steps,
                    snapshot_every=max(1, n_steps // 20),
                    reaction_substeps=10, mech_substeps=1,
                    pin_left_wall=False, block_left_u_flux=False,
                    rigid_mesh=True)
    print(f"  done in {time.time()-t0:.1f}s — "
          f"u∈[{snaps['u'].min():.3f},{snaps['u'].max():.3f}]  "
          f"v∈[{snaps['v'].min():.3f},{snaps['v'].max():.3f}]")

    # Regrid node positions from rigid_dx → args.dx. u, v are element-centred
    # and Δ-independent, so they transplant unchanged; only node coordinates
    # rescale to the target spacing λ_init·args.dx.
    if rigid_dx != args.dx:
        spacing_out = LAMBDA_IN * args.dx
        nodes_out = np.stack([ls * spacing_out, ks * spacing_out],
                             axis=-1).astype(np.float64)
        print(f"  regridded node spacing {spacing_rigid:.4f} → "
              f"{spacing_out:.4f} for output Δ={args.dx}")
    else:
        nodes_out = snaps["nodes"][-1].copy()

    # Final state → warm start
    final = sim.State(nodes=nodes_out,
                      u=snaps["u"][-1].copy(),
                      v=snaps["v"][-1].copy(),
                      t=float(snaps["t"][-1]))
    if args.out is None:
        dx_tag = str(args.dx).replace('.', 'p')
        f_tag = (f"_f{str(p.f).replace('.', 'p')}"
                 if args.f_override is not None else "")
        eps_tag = (f"_ep{str(p.epsilon).replace('.', 'p')}"
                   if args.epsilon_override is not None else "")
        tag = f"spiral{f_tag}{eps_tag}_dx{dx_tag}_t{args.settle_t:.1f}T0.npz"
        out = ROOT / "start_states" / tag
    else:
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sim.save_state(final, out)
    print(f"Wrote warm-start {out}")


if __name__ == "__main__":
    main()
