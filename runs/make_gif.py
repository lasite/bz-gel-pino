"""Render a long trajectory of the rigid-mesh Yashin-Oregonator as a GIF.

Usage:
    python runs/make_gif.py --ic spiral --total 200 --f 0.9
    python runs/make_gif.py --ic spiral --total 200 --fps 20 --save-dt 0.5
"""
from __future__ import annotations
import argparse
import io
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from glsm.params import DEFAULT
from glsm import simulator as sim
from glsm import reaction as rxn

# reuse IC builders / rigid-mesh helper
from generate_datasets import (IC_BUILDERS, make_rigid_state,
                               LAMBDA_IN, DT_OUTER, REACT_SUB)


def render_frame(u: np.ndarray, v: np.ndarray, t: float, vmax_u: float,
                 vmax_v: float, ic_label: str) -> np.ndarray:
    """Render a single (u, v) frame to an RGB numpy array via matplotlib."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), dpi=90)
    im_u = axes[0].imshow(u, cmap="inferno", vmin=0.0, vmax=vmax_u,
                           origin="lower", interpolation="bilinear")
    axes[0].set_title(f"u   t = {t:6.1f} T0")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    fig.colorbar(im_u, ax=axes[0], shrink=0.75)

    im_v = axes[1].imshow(v, cmap="viridis", vmin=0.0, vmax=vmax_v,
                           origin="lower", interpolation="bilinear")
    axes[1].set_title(f"v   t = {t:6.1f} T0")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.colorbar(im_v, ax=axes[1], shrink=0.75)

    fig.suptitle(f"Yashin mod-Oregonator rigid mesh — IC: {ic_label}")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", choices=list(IC_BUILDERS.keys()), default="spiral")
    ap.add_argument("--total", type=float, default=200.0,
                    help="total simulation time in T_0 (default 200)")
    ap.add_argument("--save-dt", type=float, default=1.0, dest="save_dt",
                    help="dump interval in T_0 (default 1.0 → 200 frames)")
    ap.add_argument("--n-grid", type=int, default=101, dest="n_grid")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--f", type=float, default=0.9)
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--q", type=float, default=None)
    ap.add_argument("--out", type=str, default=None,
                    help="output path (default plots/<ic>_<total>T0.gif)")
    ap.add_argument("--pacemaker", action="store_true",
                    help="enable periodic re-excitation (planar: left strip, "
                         "centrifugal: centre disc, spiral/chaotic: no-op)")
    ap.add_argument("--pace-period", type=float, default=5.0,
                    help="pacemaker firing period in T_0 (default 5.0)")
    ap.add_argument("--d-scale", type=float, default=1.0,
                    help="analogue of `conmul` — scales the u-flux by this factor")
    ap.add_argument("--start-state", type=str, default=None,
                    help="load warm-start state (.npz) instead of ic builder")
    ap.add_argument("--clamp", action="store_true",
                    help="continuous Dirichlet clamp on left strip (planar) "
                         "or corner patch (centrifugal) until --clamp-until; "
                         "stabilises the wave train before release")
    ap.add_argument("--clamp-until", type=float, default=100.0,
                    help="release time in T_0 (default 100)")
    ap.add_argument("--clamp-u", type=float, default=0.9)
    ap.add_argument("--clamp-v", type=float, default=0.0)
    ap.add_argument("--periodic", type=str, default="", choices=["", "x", "y", "xy"],
                    help="periodic BC axes (rigid mesh only). Default: Neumann.")
    args = ap.parse_args()

    p = DEFAULT
    over = {"f": args.f}
    if args.epsilon is not None: over["epsilon"] = args.epsilon
    if args.q       is not None: over["q"] = args.q
    p = replace(p, **over)
    print(f"Oregonator overrides: {over}")

    # load from start state (warm restart) or build fresh IC
    if args.start_state is not None:
        state = sim.load_state(args.start_state)
        Ny, Nx = state.u.shape
        print(f"loaded start state {args.start_state}  grid {Ny}×{Nx}  "
              f"t0={state.t:.2f}  u∈[{state.u.min():.3f},{state.u.max():.3f}]  "
              f"v∈[{state.v.min():.3f},{state.v.max():.3f}]")
    else:
        Ny = Nx = args.n_grid
        ic_fn = IC_BUILDERS[args.ic]
        u0, v0 = ic_fn(Ny, Nx, seed=args.seed)
        state = make_rigid_state(Ny, Nx, u0, v0, p=p, lambda_init=LAMBDA_IN)

    dump_every = int(round(args.save_dt / DT_OUTER))
    assert abs(dump_every * DT_OUTER - args.save_dt) < 1e-9, \
        f"save_dt {args.save_dt} must be multiple of dt_outer {DT_OUTER}"
    n_dumps = int(round(args.total / args.save_dt))
    print(f"grid {Ny}×{Nx}  total {args.total} T0  dt_save {args.save_dt} T0  "
          f"n_dumps {n_dumps}  dump_every {dump_every} outer steps  "
          f"d_scale={args.d_scale}")

    phi = p.phi_0 / (p.lambda_perp * LAMBDA_IN ** 2)
    u_s, v_s = rxn.stationary_uniform(phi, p)
    print(f"φ={phi:.4f}  stationary (u*,v*)=({u_s:.4f},{v_s:.4f})")

    # Pacemaker mask: only meaningful for planar (left strip) and centrifugal
    # (centre disc). Spiral / chaotic ICs self-sustain and don't need it.
    pacemaker = None
    if args.pacemaker:
        if args.ic == "planar":
            w = max(3, int(0.05 * Nx))
            mask = np.zeros((Ny, Nx), dtype=bool)
            mask[:, :w] = True
            pacemaker = {"mask": mask, "period": args.pace_period,
                         "u_set": 0.9, "v_set": 0.0, "gate_v": 0.10}
        elif args.ic == "centrifugal":
            R = max(4, int(0.06 * min(Ny, Nx)))
            cx, cy = Nx // 2, Ny // 2
            k_idx, l_idx = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing="ij")
            mask = (l_idx - cx) ** 2 + (k_idx - cy) ** 2 < R * R
            pacemaker = {"mask": mask, "period": args.pace_period,
                         "u_set": 0.9, "v_set": 0.0, "gate_v": 0.10}
        else:
            print(f"[info] pacemaker requested for IC '{args.ic}' — ignored "
                  f"(only planar/centrifugal use it)")

    clamp = None
    if args.clamp:
        if args.ic == "planar":
            w = max(3, int(0.03 * Nx))
            m = np.zeros((Ny, Nx), dtype=bool); m[:, :w] = True
            clamp = {"mask": m, "u_value": args.clamp_u, "v_value": args.clamp_v,
                     "release_at": args.clamp_until}
        elif args.ic == "centrifugal":
            side = max(6, Ny // 4)
            m = np.zeros((Ny, Nx), dtype=bool); m[:side, :side] = True
            clamp = {"mask": m, "u_value": args.clamp_u, "v_value": args.clamp_v,
                     "release_at": args.clamp_until}
        else:
            print(f"[info] clamp requested for IC '{args.ic}' — ignored")
        if clamp is not None:
            print(f"  clamp: {clamp['mask'].sum()} cells held at "
                  f"u={clamp['u_value']}, v={clamp['v_value']} until "
                  f"t={clamp['release_at']} T_0")

    t0 = time.time()
    snaps = sim.run(state, p, dt=DT_OUTER, n_steps=n_dumps * dump_every,
                    snapshot_every=dump_every,
                    reaction_substeps=REACT_SUB,
                    pin_left_wall=False, block_left_u_flux=False,
                    rigid_mesh=True,
                    d_scale=args.d_scale,
                    pacemaker=pacemaker,
                    clamp=clamp,
                    periodic=args.periodic)
    print(f"sim done {time.time()-t0:.1f}s  "
          f"u∈[{snaps['u'].min():.3f},{snaps['u'].max():.3f}]  "
          f"v∈[{snaps['v'].min():.3f},{snaps['v'].max():.3f}]")

    # Render frames
    vmax_u = max(0.9, float(snaps["u"].max()))
    vmax_v = max(0.5, float(snaps["v"].max()))
    n_frames = snaps["u"].shape[0]
    frames = []
    t_render0 = time.time()
    for i in range(n_frames):
        frame = render_frame(snaps["u"][i], snaps["v"][i],
                             float(snaps["t"][i]), vmax_u, vmax_v, args.ic)
        frames.append(Image.fromarray(frame))
        if (i + 1) % 50 == 0 or i == n_frames - 1:
            print(f"  rendered {i+1}/{n_frames}  "
                  f"({time.time()-t_render0:.1f}s elapsed)", flush=True)

    # Encode GIF
    out_dir = ROOT / "plots"
    out_dir.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else \
        out_dir / f"{args.ic}_{int(round(args.total))}T0.gif"
    duration_ms = int(round(1000.0 / args.fps))
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    sz_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path}  "
          f"({n_frames} frames, {args.fps} fps, {sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
