"""Rigid-mesh dataset generator — 4 IC types, CardiacEP-PINOS `.pt` format.

This script reuses the already-working Yashin-Balazs 2007 2D gLSM simulator but
switches off gel deformation (rigid_mesh=True), so only the **polymer-modified
Oregonator reaction-diffusion** dynamics are integrated on a fixed uniform
λ=λ_init square lattice.

Initial conditions mirror the four types used in the CardiacEP-PINOS / Lydon
work (centrifugal / planar / spiral / chaotic). For each IC type the script
produces (n_traj + n_test) trajectories, each of length N_DUMPS+1 snapshots,
then sliding-windows them into (t_in → t_out) input/output pairs and writes

    dataset/<ic_type>/Train_<Nt>_frames_<t_in>_inputsteps_<t_out>_outputsteps/
        2D_Oreg_train_<res>_<cm>.pt      # {"x": [N,2,T_in,H,W], "y":[N,2,T_out,H,W]}
        2D_Oreg_test_<res>_<cm>.pt       # same layout
        dataset_info_<res>_<cm>.txt      # parsed by PINO_Train_Oreg.py
        sanity_<ic_type>.png

Run:
    python runs/generate_datasets.py --ic spiral --sanity
    python runs/generate_datasets.py --ic spiral
"""
from __future__ import annotations
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from glsm.params import DEFAULT
from glsm import simulator as sim
from glsm import reaction as rxn

CJK = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
                     size=10)

# -------------------------------------------------------------------
# Generator defaults (can be overridden on the CLI)
# -------------------------------------------------------------------
N_GRID     = 64          # (Ny, Nx) element grid — 64 is fast on CPU
LAMBDA_IN  = 1.1         # uniform mesh swelling; matches DEFAULT.lambda_perp
DT_OUTER   = 0.005       # T_0 — outer solver step
REACT_SUB  = 10          # reaction sub-steps per outer step
DUMP_EVERY = 100         # outer steps per snapshot  → dt_dump = 0.5 T_0
N_DUMPS    = 100         # 50 T_0 of dynamics — several wave passages

# IC amplitudes — ported from temp/Oregonator 2D.py (Tyson-style sharp-step IC).
# u excited near saturation of (1-φ)² ≈ 0.80, v refractory ≈ stationary v_s level.
U_HIGH = 0.85
V_HIGH = 0.25
BACKGROUND = 1e-4   # ≈ q — the temp-script uses sim.q as background floor


# -------------------------------------------------------------------
# Initial-condition builders
#
# Electrode geometry and stimulus kind match CardiacEP-PINOS
# `2D_Waves_AP.py` (planar: thin left strip; centrifugal: bottom-left square
# corner patch); the stimulus is applied once at t=0 as a sharp-edged IC and
# then the reaction-diffusion is integrated without further injection. This
# is exactly what the reference GIFs show: one bolus → single wave → decay.
#
# Spiral / chaotic use broken-wavefront patterns — for these our Yashin-gel
# Oregonator is self-sustaining (already verified 200 T_0 spiral rotation).
#
# Array convention: (Ny, Nx), axis 0 = k (row / y), axis 1 = l (col / x).
# -------------------------------------------------------------------
def ic_spiral(Ny: int, Nx: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Broken-wavefront spiral (left half u excited + bottom half v refractory).
    The tip at the crossing curls into a spiral. Self-sustaining."""
    rng = np.random.default_rng(seed)
    u = np.full((Ny, Nx), BACKGROUND, dtype=np.float64)
    v = np.full((Ny, Nx), BACKGROUND, dtype=np.float64)
    u[:, :Nx // 2] = U_HIGH
    v[Ny // 2:, :] = V_HIGH
    jitter = 0.02 * rng.standard_normal(u.shape)
    u = np.clip(u + jitter * U_HIGH, 0.0, 1.5)
    return u, v


def ic_centrifugal(Ny: int, Nx: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bottom-left corner square patch (CardiacEP-PINOS centrifugal electrode).

    Original electrode is 25 mm × 25 mm on a 100 mm × 100 mm mesh ⇒ 25% on
    each side. On our (Ny, Nx) grid we excite a ~25% × 25% corner patch at
    (0:Ny/4, 0:Nx/4), with a small seed-dependent offset for trajectory
    diversity. One pulse; no re-excitation — the expanding quarter-ring
    propagates, decays through the refractory tail, and the domain recovers.
    """
    rng = np.random.default_rng(seed)
    u = np.full((Ny, Nx), BACKGROUND, dtype=np.float64)
    v = np.full((Ny, Nx), BACKGROUND, dtype=np.float64)
    # 25% × 25% corner, with small random offset so each trajectory differs
    side = max(6, Ny // 4)
    dy = int((rng.random() - 0.5) * 0.05 * Ny)
    dx = int((rng.random() - 0.5) * 0.05 * Nx)
    y_lo = max(0, dy);                  y_hi = min(Ny, y_lo + side)
    x_lo = max(0, dx);                  x_hi = min(Nx, x_lo + side)
    u[y_lo:y_hi, x_lo:x_hi] = 0.9
    v[y_lo:y_hi, x_lo:x_hi] = 0.0
    return u, v


def ic_planar(Ny: int, Nx: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Thin left-wall strip (CardiacEP-PINOS planar electrode).

    Original electrode is 1 mm × 100 mm on a 100 mm × 100 mm mesh ⇒ 1%
    thick full-height strip at the left wall. On 101×101 this maps to 1–2
    cells, which is too thin for our slower BZ wave to properly launch, so
    we use 3 cells to give the wave a more robust seed. Full domain height.
    One pulse; no re-excitation — the single planar front propagates toward
    +x, refractory tail follows, domain recovers.
    """
    rng = np.random.default_rng(seed)
    u = np.full((Ny, Nx), BACKGROUND, dtype=np.float64)
    v = np.full((Ny, Nx), BACKGROUND, dtype=np.float64)
    w = max(3, int(0.03 * Nx))
    dx = int((rng.random() - 0.5) * 0.02 * Nx)
    lo = max(0, dx)
    hi = min(Nx, lo + w)
    u[:, lo:hi] = 0.9
    v[:, lo:hi] = 0.0
    return u, v


def ic_uniform_noise(Ny: int, Nx: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform stationary state + ±50% multiplicative noise — mirrors the
    3D src/ 'setGoonValue' layout (gelSystem.cu:190-197). This is the *only*
    IC that remains stable when mechanics is turned on from t=0, because it
    keeps the pressure gradient small in the initial transient. Fully-developed
    patterns (ic_spiral's broken wavefront, a warm-rigid spiral state) have
    sharp v-fronts that immediately create huge χ*·v·φ pressure jumps, which
    explicit Euler can't handle (the 3D code uses VODE / stiff implicit).
    """
    rng = np.random.default_rng(seed)
    # Stationary state at phi_init assuming lambda=lambda_init (caller-enforced).
    phi_init = DEFAULT.phi_0 / (DEFAULT.lambda_perp * LAMBDA_IN ** 2)
    u_s, v_s = rxn.stationary_uniform(phi_init, DEFAULT)
    u = u_s * (1 + 0.5 * (2 * rng.random((Ny, Nx)) - 1))
    v = v_s * (1 + 0.5 * (2 * rng.random((Ny, Nx)) - 1))
    return u.astype(np.float64), v.astype(np.float64)


def ic_chaotic(Ny: int, Nx: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Two opposite-chirality broken wavefronts — upper-left + lower-right
    excited quadrants with their own refractory strips. Analog of the paper's
    `chaotic` / `breakup` category: two counter-rotating spirals interact and
    can fragment. Self-sustaining."""
    rng = np.random.default_rng(seed)
    u = np.full((Ny, Nx), BACKGROUND, dtype=np.float64)
    v = np.full((Ny, Nx), BACKGROUND, dtype=np.float64)
    u[:Ny // 2, :Nx // 2] = U_HIGH
    v[:Ny // 4, :]        = V_HIGH
    u[Ny // 2:, Nx // 2:] = U_HIGH
    v[:, 3 * Nx // 4:]    = np.maximum(v[:, 3 * Nx // 4:], V_HIGH)
    u += 0.01 * rng.standard_normal(u.shape) * U_HIGH
    u = np.clip(u, 0.0, 1.5)
    return u, v


IC_BUILDERS = {
    "spiral":      ic_spiral,
    "centrifugal": ic_centrifugal,
    "planar":      ic_planar,
    "chaotic":     ic_chaotic,
    "uniform":     ic_uniform_noise,   # for --deform (mechanics-compatible)
}


# -------------------------------------------------------------------
# One trajectory: rigid-mesh Yashin Oregonator
# -------------------------------------------------------------------
def make_rigid_state(Ny: int, Nx: int, u0: np.ndarray, v0: np.ndarray,
                     p=DEFAULT, lambda_init: float = LAMBDA_IN) -> sim.State:
    """Uniform square lattice at λ=lambda_init with arbitrary (u0, v0).

    Fortran line 1037–1038: rn(1) = ii·LAMP·dx, rn(2) = jj·LAMP·dx — so the
    physical spacing between adjacent nodes in the equilibrium state is
    `lambda_init · p.dx`. Paper convention (p.dx=1) reduces to the old
    `ls·lambda_init` placement; Fortran convention (p.dx=0.5) shrinks the
    physical domain by a factor of dx.
    """
    spacing = lambda_init * p.dx
    ks, ls = np.meshgrid(np.arange(Ny + 1), np.arange(Nx + 1), indexing="ij")
    nodes = np.stack([ls * spacing, ks * spacing], axis=-1).astype(np.float64)
    return sim.State(nodes=nodes, u=u0.copy(), v=v0.copy())


def simulate_trajectory(u0: np.ndarray, v0: np.ndarray, p=DEFAULT,
                        lambda_init: float = LAMBDA_IN,
                        dt: float = DT_OUTER,
                        dump_every: int = DUMP_EVERY,
                        n_dumps: int = N_DUMPS,
                        react_sub: int = REACT_SUB,
                        rigid_mesh: bool = True,
                        mech_substeps: int = 1
                        ) -> dict:
    """Integrate rigid- or deformable-mesh reaction-diffusion from a uniform
    square-lattice IC with given (u0, v0). Returns a dict with float32
    snapshots for u, v, phi, nodes (shape (n_dumps+1, ...)).
    """
    Ny, Nx = u0.shape
    state = make_rigid_state(Ny, Nx, u0, v0, p=p, lambda_init=lambda_init)
    snaps = sim.run(state, p, dt=dt, n_steps=n_dumps * dump_every,
                    snapshot_every=dump_every,
                    reaction_substeps=react_sub,
                    mech_substeps=mech_substeps,
                    pin_left_wall=False,
                    block_left_u_flux=False,
                    rigid_mesh=rigid_mesh)
    return {k: snaps[k].astype(np.float32 if k != "nodes" else np.float64)
            for k in ("u", "v", "phi", "nodes")}


def simulate_from_state(state: sim.State, p=DEFAULT,
                        dt: float = DT_OUTER,
                        dump_every: int = DUMP_EVERY,
                        n_dumps: int = N_DUMPS,
                        react_sub: int = REACT_SUB,
                        rigid_mesh: bool = True,
                        mech_substeps: int = 1
                        ) -> dict:
    """Integrate from a loaded warm-start State. Returns dict with u, v, phi,
    nodes snapshots (n_dumps+1, ...).
    """
    snaps = sim.run(state, p, dt=dt, n_steps=n_dumps * dump_every,
                    snapshot_every=dump_every,
                    reaction_substeps=react_sub,
                    mech_substeps=mech_substeps,
                    pin_left_wall=False,
                    block_left_u_flux=False,
                    rigid_mesh=rigid_mesh)
    return {k: snaps[k].astype(np.float32 if k != "nodes" else np.float64)
            for k in ("u", "v", "phi", "nodes")}


# -------------------------------------------------------------------
# Sliding-window repackage + dataset_info writer (CardiacEP-PINOS layout)
# -------------------------------------------------------------------
def make_sliding_windows(trajs_uv: torch.Tensor, t_in: int, t_out: int
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(N_traj, C, T, H, W) → x:(N, C, t_in, H, W), y:(N, C, t_out, H, W).

    Works for any channel count C (2 for rigid rigid, 4 for deformable with
    (u, v, δx, δy)).
    """
    N_traj, C, T, H, W = trajs_uv.shape
    n_win = T - t_in - t_out + 1
    assert n_win > 0, f"T={T} too short for t_in+t_out={t_in+t_out}"
    xs, ys = [], []
    for n in range(N_traj):
        for s in range(n_win):
            xs.append(trajs_uv[n, :, s:s + t_in])
            ys.append(trajs_uv[n, :, s + t_in:s + t_in + t_out])
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


# -------------------------------------------------------------------
# Deformation-channel helpers (option B: u, v, δx, δy on element grid)
# -------------------------------------------------------------------
def element_center_displacement(nodes: np.ndarray, lambda_init: float,
                                dx: float = 1.0) -> np.ndarray:
    """Element-center displacement from undeformed reference lattice.

    nodes : (T, Ny+1, Nx+1, 2)        float64, [..., 0]=x, [..., 1]=y
    returns (T, Ny, Nx, 2) float32 — δ = deformed_center - reference_center

    Element (k, l) reference center in the undeformed λ=lambda_init, Δ=dx
    square lattice is at ((l+0.5)·λ·dx, (k+0.5)·λ·dx). Deformed center is
    the mean of the four corner nodes.
    """
    _, Ny_p1, Nx_p1, _ = nodes.shape
    Ny, Nx = Ny_p1 - 1, Nx_p1 - 1
    centers = 0.25 * (
        nodes[:, :-1, :-1] + nodes[:, :-1, 1:]
        + nodes[:, 1:,  1:] + nodes[:, 1:,  :-1]
    )                                         # (T, Ny, Nx, 2)
    spacing = lambda_init * dx
    ks = np.arange(Ny) + 0.5
    ls = np.arange(Nx) + 0.5
    ref_y, ref_x = np.meshgrid(ks * spacing, ls * spacing, indexing="ij")
    ref = np.stack([ref_x, ref_y], axis=-1)   # (Ny, Nx, 2)
    delta = centers - ref[None]
    return delta.astype(np.float32)


def traj_to_channels(traj: dict, lambda_init: float, deform: bool,
                     dx_grid: float = 1.0) -> np.ndarray:
    """Pack a simulator-snapshot dict into a (C, T, Ny, Nx) float32 tensor.

    deform=False → C=2 (u, v)
    deform=True  → C=4 (u, v, δx, δy)
    """
    u = traj["u"].astype(np.float32)          # (T, Ny, Nx)
    v = traj["v"].astype(np.float32)
    if not deform:
        return np.stack([u, v], axis=0)       # (2, T, Ny, Nx)
    d = element_center_displacement(traj["nodes"], lambda_init,
                                    dx=dx_grid)  # (T, Ny, Nx, 2)
    dxc = d[..., 0]                           # (T, Ny, Nx)
    dyc = d[..., 1]
    return np.stack([u, v, dxc, dyc], axis=0)  # (4, T, Ny, Nx)


def write_dataset_info(path: Path, *, sim_type: str, cm: float,
                       res: int, grid_res_cm: float, dt_ms: int,
                       n_timesteps: int, u_lo: float, u_hi: float,
                       v_lo: float, v_hi: float, t_in: int, t_out: int,
                       n_train_pairs: int, n_test_pairs: int,
                       domain_label: str) -> None:
    """Write dataset_info_<res>_<cm>.txt in the format PINO_Train_Oreg expects.

    Parser greps for (via re.findall(r'\\d+')):
      'Grid_resolution', 'Timestep resolution', 'Training data shapes',
      'Testing data shapes', 'Input-Output pairs',
      'Resting potential (E_rest) = '.
    """
    total_ms = dt_ms * (n_timesteps - 1)
    lines = [
        f"Simulation Info: simtype = {sim_type}, Conductivity Multipler = {cm}, "
        f"resolution abstraction factor = x1, time resolution scaling = t1",
        f"Coordinates loaded from mesh: {domain_label}",
        f"Loaded tensor from (generated), shape = (2, {n_timesteps}, {res}, {res})",
        f"Grid_size: {res}",
        f"Grid_resolution: {grid_res_cm} cm",
        f"Timestep resolution: {dt_ms} ms",
        f"Full simulation time period: {total_ms} ms",
        f"Metadata: {{'timesteps': {n_timesteps}, 'x_dim': {res}, 'y_dim': {res}, "
        f"'t_res_ms': {dt_ms}, 'duration_ms': {total_ms}}}",
        f"Channel 0 pre-normalisation range: {u_lo:.4f}, {u_hi:.4f}",
        f"Resting potential (E_rest) = {u_lo:.4f} mV, amplitude (A) = {u_hi - u_lo:.4f} mV",
        "Channel 0 not normalised, raw values retained",
        f"Channel 1 pre-normalisation range: {v_lo:.4f}, {v_hi:.4f}",
        "Channel 1 not normalised, raw values retained",
        f"Loaded dataset shape: torch.Size([2, {n_timesteps}, {res}, {res}])",
        f"Input-Output pairs formed using moving window of [ Input: {t_in}, "
        f"Output: {t_out} timesteps ]",
        f"Training data shapes: {{torch.Size([{n_train_pairs}, 2, {t_in}, {res}, {res}])}} "
        f"{{torch.Size([{n_train_pairs}, 2, {t_out}, {res}, {res}])}}",
        f"Testing data shapes: {{torch.Size([{n_test_pairs}, 2, {t_in}, {res}, {res}])}} "
        f"{{torch.Size([{n_test_pairs}, 2, {t_out}, {res}, {res}])}}",
    ]
    path.write_text("\n".join(lines) + "\n")


# -------------------------------------------------------------------
# Sanity plotter — 2×8 snapshots of u and v over one trajectory
# -------------------------------------------------------------------
def plot_sanity(u_tr: np.ndarray, v_tr: np.ndarray, out_png: Path,
                ic_type: str, dt_dump: float) -> None:
    T = u_tr.shape[0]
    idx = [0, T // 20, T // 10, T // 6, T // 4, T // 2, 3 * T // 4, T - 1]
    u_lo, u_hi = float(u_tr.min()), float(u_tr.max())
    v_lo, v_hi = float(v_tr.min()), float(v_tr.max())
    fig, axes = plt.subplots(2, len(idx), figsize=(2.8 * len(idx), 6))
    for col, t in enumerate(idx):
        im_u = axes[0, col].imshow(u_tr[t], cmap="RdBu_r", vmin=u_lo, vmax=u_hi)
        im_v = axes[1, col].imshow(v_tr[t], cmap="RdBu_r", vmin=v_lo, vmax=v_hi)
        axes[0, col].set_title(f"t = {t * dt_dump:.1f} T₀", fontproperties=CJK)
        for row in (0, 1):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel("u", fontproperties=CJK)
    axes[1, 0].set_ylabel("v", fontproperties=CJK)
    fig.colorbar(im_u, ax=axes[0, :], shrink=0.7)
    fig.colorbar(im_v, ax=axes[1, :], shrink=0.7)
    fig.suptitle(f"Yashin modified-Oregonator (rigid mesh λ={LAMBDA_IN}, φ≈0.105) "
                 f"— IC: {ic_type}", fontproperties=CJK)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", choices=list(IC_BUILDERS.keys()), default="spiral")
    ap.add_argument("--n-traj", type=int, default=20, dest="n_traj")
    ap.add_argument("--n-test", type=int, default=4, dest="n_test")
    ap.add_argument("--t-in",  type=int, default=5, dest="t_in")
    ap.add_argument("--t-out", type=int, default=5, dest="t_out")
    ap.add_argument("--n-grid", type=int, default=N_GRID, dest="n_grid")
    ap.add_argument("--n-dumps", type=int, default=N_DUMPS, dest="n_dumps")
    ap.add_argument("--dump-every", type=int, default=DUMP_EVERY, dest="dump_every")
    ap.add_argument("--dt-outer", type=float, default=DT_OUTER, dest="dt_outer",
                    help="outer-solver step in T_0. Default %(default)s. "
                         "For --deform, use smaller (e.g. 0.001) to keep "
                         "explicit-Euler mechanics stable.")
    ap.add_argument("--lambda0", type=float, default=None, dest="lambda0",
                    help="override gel mobility coefficient (default 100 "
                         "from Yashin Table II). Lower values slow gel "
                         "response; smallest safe mechanics time step scales "
                         "as 1/Λ_0.")
    ap.add_argument("--mech-substeps", type=int, default=1, dest="mech_substeps",
                    help="number of forward-Euler sub-steps per outer dt for "
                         "the mechanics + flux block. Needed for --deform with "
                         "paper Λ_0=100 (try 20-50) because the paper uses "
                         "VODE/stiff implicit which we don't have. Default "
                         "%(default)s.")
    ap.add_argument("--dx", type=float, default=None, dest="dx",
                    help="undeformed element edge length Δ. Paper convention "
                         "is 1.0; Fortran temp/2D.f90 uses 0.5. Shrinks the "
                         "physical domain (node spacing = λ_init·dx) and "
                         "scales the flux/mobility/area coefficients in "
                         "lockstep with the Fortran reference. Default from "
                         "Parameters (1.0).")
    ap.add_argument("--cm", type=float, default=1.0,
                    help="label only — Yashin D is fixed by L_0 / T_0")
    ap.add_argument("--f", type=float, default=None,
                    help="override stoichiometric factor f (default 0.7 from Table I)")
    ap.add_argument("--epsilon", type=float, default=None,
                    help="override timescale ratio ε (default 0.354 from Table I)")
    ap.add_argument("--q", type=float, default=None,
                    help="override Oregonator q (default 9.52e-5 from Table I)")
    ap.add_argument("--sanity", action="store_true",
                    help="run 1 trajectory, plot, skip .pt writing")
    ap.add_argument("--deform", action="store_true",
                    help="enable gel mechanics (rigid_mesh=False). Captures "
                         "time-varying φ and node positions; saves them "
                         "alongside u/v. Output folder suffix: _deform")
    ap.add_argument("--start-state", type=str, default=None,
                    dest="start_state",
                    help="single warm-start .npz state (analogue of one "
                         "CardiacEP-PINOS .roe file). When set, the script "
                         "runs ONE long trajectory of n_dumps+1 frames and "
                         "splits it chronologically into train (first) and "
                         "test (last) slices — exactly the CardiacEP-PINOS "
                         "one-trajectory-per-(sim_type, conmul) layout. "
                         "--n-traj / --n-test are ignored in this mode.")
    ap.add_argument("--single-traj", action="store_true", dest="single_traj",
                    help="same chronological-split single-trajectory layout "
                         "as --start-state, but build the IC from --ic "
                         "(seed=42) instead of loading a .npz. Ignores "
                         "--n-traj / --n-test.")
    ap.add_argument("--train-frac", type=float, default=0.8, dest="train_frac",
                    help="fraction of the single-trajectory frames used for "
                         "train sliding windows; the rest is the test split. "
                         "Only used with --start-state.")
    ap.add_argument("--gif", type=str, default=None,
                    help="if set, render the single trajectory (used with "
                         "--start-state) as a GIF at this path. Reuses the "
                         "simulation data so no extra integration is done.")
    ap.add_argument("--gif-fps", type=int, default=20, dest="gif_fps")
    ap.add_argument("--out-root", type=str,
                    default=str(ROOT / "dataset"))
    args = ap.parse_args()

    p = DEFAULT
    overrides = {}
    if args.f is not None:        overrides["f"] = args.f
    if args.epsilon is not None:  overrides["epsilon"] = args.epsilon
    if args.q is not None:        overrides["q"] = args.q
    if args.lambda0 is not None:  overrides["Lambda_0"] = args.lambda0
    if args.dx is not None:       overrides["dx"] = args.dx
    if overrides:
        p = replace(p, **overrides)
        print(f"Overriding params: {overrides}")
    Ny = Nx = args.n_grid
    ic_fn = IC_BUILDERS[args.ic]
    dt_dump = args.dt_outer * args.dump_every
    N_T = args.n_dumps + 1

    # Confirm initial φ and stationary state
    phi_init = p.phi_0 / (p.lambda_perp * LAMBDA_IN ** 2)
    u_s, v_s = rxn.stationary_uniform(phi_init, p)
    print(f"λ_init={LAMBDA_IN}  φ_init={phi_init:.4f}  "
          f"stationary (u*,v*)=({u_s:.4f},{v_s:.4f})  "
          f"grid {Ny}×{Nx}  dt_dump={dt_dump} T₀")

    if args.sanity:
        t0 = time.time()
        u0, v0 = ic_fn(Ny, Nx, seed=42)
        print(f"[sanity] IC u∈[{u0.min():.3f},{u0.max():.3f}]  "
              f"v∈[{v0.min():.3f},{v0.max():.3f}]  "
              f"rigid_mesh={not args.deform}")
        traj = simulate_trajectory(u0, v0, p=p, lambda_init=LAMBDA_IN,
                                   dt=args.dt_outer,
                                   dump_every=args.dump_every,
                                   n_dumps=args.n_dumps,
                                   react_sub=REACT_SUB,
                                   rigid_mesh=not args.deform, mech_substeps=args.mech_substeps)
        u_tr = traj["u"]; v_tr = traj["v"]
        phi_tr = traj["phi"]; nodes_tr = traj["nodes"]
        print(f"[sanity] traj u∈[{u_tr.min():.3f},{u_tr.max():.3f}]  "
              f"v∈[{v_tr.min():.3f},{v_tr.max():.3f}]  "
              f"φ∈[{phi_tr.min():.3f},{phi_tr.max():.3f}]  "
              f"({time.time()-t0:.1f}s)")
        if args.deform:
            # Domain-extent evolution: spread of node x across frames
            x_min = nodes_tr[..., 0].min(axis=(1, 2))
            x_max = nodes_tr[..., 0].max(axis=(1, 2))
            y_min = nodes_tr[..., 1].min(axis=(1, 2))
            y_max = nodes_tr[..., 1].max(axis=(1, 2))
            print(f"[sanity] domain x-extent: [{x_min.min():.3f},{x_max.max():.3f}]  "
                  f"y-extent: [{y_min.min():.3f},{y_max.max():.3f}]")
        out_dir = ROOT / "plots"
        out_dir.mkdir(exist_ok=True)
        tag = "deform" if args.deform else "rigid"
        out_png = out_dir / f"sanity_{tag}_{args.ic}.png"
        plot_sanity(u_tr, v_tr, out_png, args.ic, dt_dump)
        print(f"[sanity] saved {out_png.relative_to(ROOT)}")
        return

    ic_folder = (args.ic + "_deform") if args.deform else args.ic
    folder = (Path(args.out_root) / ic_folder
              / f"Train_{N_T}_frames_{args.t_in}_inputsteps_{args.t_out}_outputsteps")
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {folder}  (deform={args.deform})")

    t0 = time.time()
    rigid_flag = not args.deform

    # ----- single-trajectory (CardiacEP-PINOS) mode -----
    if args.start_state is not None or args.single_traj:
        if args.single_traj and args.start_state is None:
            u0, v0 = ic_fn(Ny, Nx, seed=42)
            st = make_rigid_state(Ny, Nx, u0, v0, p=p, lambda_init=LAMBDA_IN)
            print(f"Built fresh IC from ic_{args.ic} (seed=42): "
                  f"u∈[{u0.min():.3f},{u0.max():.3f}]  "
                  f"v∈[{v0.min():.3f},{v0.max():.3f}]")
        else:
            ss_path = Path(args.start_state)
            st = sim.load_state(ss_path)
            assert st.u.shape == (Ny, Nx), \
                f"start state {ss_path.name} grid {st.u.shape} ≠ {(Ny, Nx)}"
            print(f"Loaded start state {ss_path.name}  t0={st.t:.2f}  "
                  f"u∈[{st.u.min():.3f},{st.u.max():.3f}]  "
                  f"v∈[{st.v.min():.3f},{st.v.max():.3f}]")
        traj = simulate_from_state(st, p=p, dt=args.dt_outer,
                                   dump_every=args.dump_every,
                                   n_dumps=args.n_dumps,
                                   react_sub=REACT_SUB,
                                   rigid_mesh=rigid_flag, mech_substeps=args.mech_substeps)
        u_tr = traj["u"]; v_tr = traj["v"]
        phi_tr = traj["phi"]; nodes_tr = traj["nodes"]
        print(f"  single trajectory {N_T} frames  "
              f"u[{u_tr.min():.3f},{u_tr.max():.3f}] "
              f"v[{v_tr.min():.3f},{v_tr.max():.3f}] "
              f"φ[{phi_tr.min():.3f},{phi_tr.max():.3f}]  "
              f"({time.time()-t0:.1f}s)")
        if args.deform:
            delta_full = element_center_displacement(nodes_tr, LAMBDA_IN,
                                                     dx=p.dx)
            print(f"  element-center δ range: "
                  f"x∈[{delta_full[...,0].min():.4f},{delta_full[...,0].max():.4f}]  "
                  f"y∈[{delta_full[...,1].min():.4f},{delta_full[...,1].max():.4f}]")

        # Optional GIF render from the same simulation data
        if args.gif is not None:
            from PIL import Image as PILImage
            from make_gif import render_frame
            gif_path = Path(args.gif)
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            vmax_u = max(0.9, float(u_tr.max()))
            vmax_v = max(0.5, float(v_tr.max()))
            t_frames = st.t + np.arange(N_T) * dt_dump
            n_tr_split = int(round(args.train_frac * N_T))
            pil_frames = []
            tg0 = time.time()
            for i in range(N_T):
                label = f"{args.ic}  [{'train' if i < n_tr_split else 'TEST'}]"
                frame = render_frame(u_tr[i], v_tr[i], float(t_frames[i]),
                                     vmax_u, vmax_v, label)
                pil_frames.append(PILImage.fromarray(frame))
                if (i + 1) % 50 == 0 or i == N_T - 1:
                    print(f"    gif rendered {i+1}/{N_T}  "
                          f"({time.time()-tg0:.1f}s)", flush=True)
            duration_ms = int(round(1000.0 / args.gif_fps))
            pil_frames[0].save(gif_path, save_all=True,
                               append_images=pil_frames[1:],
                               duration=duration_ms, loop=0, optimize=True)
            sz_mb = gif_path.stat().st_size / 1e6
            print(f"  wrote {gif_path}  ({N_T} frames, {args.gif_fps} fps, "
                  f"{sz_mb:.1f} MB)")

        # Chronological split: first train_frac frames → train windows,
        # remainder → test windows. No overlap — test starts at frame n_tr.
        n_tr = int(round(args.train_frac * N_T))
        n_te = N_T - n_tr
        min_frames = args.t_in + args.t_out
        if n_tr < min_frames or n_te < min_frames:
            raise SystemExit(
                f"Split too small: train {n_tr} / test {n_te} frames, each "
                f"must be ≥ t_in+t_out={min_frames}. "
                "Increase --n-dumps or adjust --train-frac.")
        print(f"split: train frames [0:{n_tr}]  test frames [{n_tr}:{N_T}]")

        full_channels = traj_to_channels(traj, LAMBDA_IN, args.deform,
                                          dx_grid=p.dx)  # (C, T, H, W)
        C = full_channels.shape[0]
        trajs_train = full_channels[np.newaxis, :, :n_tr]                # (1, C, n_tr, H, W)
        trajs_test  = full_channels[np.newaxis, :, n_tr:]                # (1, C, n_te, H, W)

    # ----- legacy multi-seed (ignored IC-type / seed-loop) mode -----
    else:
        C = 4 if args.deform else 2
        trajs_train = np.empty((args.n_traj, C, N_T, Ny, Nx), dtype=np.float32)
        for i in range(args.n_traj):
            u0, v0 = ic_fn(Ny, Nx, seed=1000 + i)
            traj_i = simulate_trajectory(u0, v0, p=p, lambda_init=LAMBDA_IN,
                                         dt=args.dt_outer,
                                         dump_every=args.dump_every,
                                         n_dumps=args.n_dumps,
                                         react_sub=REACT_SUB,
                                         rigid_mesh=rigid_flag, mech_substeps=args.mech_substeps)
            trajs_train[i] = traj_to_channels(traj_i, LAMBDA_IN, args.deform,
                                               dx_grid=p.dx)
            print(f"  [train {i:02d}] "
                  f"u[{traj_i['u'].min():.3f},{traj_i['u'].max():.3f}] "
                  f"v[{traj_i['v'].min():.3f},{traj_i['v'].max():.3f}]  "
                  f"elapsed {time.time()-t0:.1f}s", flush=True)

        trajs_test = np.empty((args.n_test, C, N_T, Ny, Nx), dtype=np.float32)
        for i in range(args.n_test):
            u0, v0 = ic_fn(Ny, Nx, seed=9000 + i)
            traj_i = simulate_trajectory(u0, v0, p=p, lambda_init=LAMBDA_IN,
                                         dt=args.dt_outer,
                                         dump_every=args.dump_every,
                                         n_dumps=args.n_dumps,
                                         react_sub=REACT_SUB,
                                         rigid_mesh=rigid_flag, mech_substeps=args.mech_substeps)
            trajs_test[i] = traj_to_channels(traj_i, LAMBDA_IN, args.deform,
                                              dx_grid=p.dx)
            print(f"  [test  {i:02d}] elapsed {time.time()-t0:.1f}s",
                  flush=True)

    # Sliding-window repackage
    x_tr, y_tr = make_sliding_windows(torch.from_numpy(trajs_train),
                                      args.t_in, args.t_out)
    x_te, y_te = make_sliding_windows(torch.from_numpy(trajs_test),
                                      args.t_in, args.t_out)
    print(f"Train pairs: x {tuple(x_tr.shape)}  y {tuple(y_tr.shape)}")
    print(f"Test  pairs: x {tuple(x_te.shape)}  y {tuple(y_te.shape)}")

    cm_str = f"{args.cm}"
    train_pt = folder / f"2D_Oreg_train_{Ny}_{cm_str}.pt"
    test_pt  = folder / f"2D_Oreg_test_{Ny}_{cm_str}.pt"
    torch.save({"x": x_tr, "y": y_tr}, train_pt)
    torch.save({"x": x_te, "y": y_te}, test_pt)
    print(f"Wrote {train_pt}")
    print(f"Wrote {test_pt}")

    # Save extra physical fields (phi, nodes) alongside — only needed for
    # start-state deformable single-trajectory mode, where traj_full is in scope.
    if args.deform and args.start_state is not None:
        extras = {
            "phi": torch.from_numpy(phi_tr.astype(np.float32)),        # [T, H, W]
            "nodes": torch.from_numpy(nodes_tr.astype(np.float32)),    # [T, Ny+1, Nx+1, 2]
            "lambda_init": LAMBDA_IN,
        }
        extras_pt = folder / f"2D_Oreg_phi_nodes_{Ny}_{cm_str}.pt"
        torch.save(extras, extras_pt)
        print(f"Wrote {extras_pt}  (phi {tuple(extras['phi'].shape)}, "
              f"nodes {tuple(extras['nodes'].shape)})")

    # dataset_info — dt stored as integer (times 1000) so re.findall gets it
    dx_cm = 0.004 * LAMBDA_IN         # L_0 = 40 μm = 0.004 cm; × initial swelling
    dt_ms = int(round(dt_dump * 1000))  # T_0 = 1 s → dt_dump = 0.5 → "500"
    u_lo, u_hi = float(trajs_train[:, 0].min()), float(trajs_train[:, 0].max())
    v_lo, v_hi = float(trajs_train[:, 1].min()), float(trajs_train[:, 1].max())
    info_path = folder / f"dataset_info_{Ny}_{cm_str}.txt"
    domain = (f"(λ={LAMBDA_IN} ref lattice, {'mechanics on' if args.deform else 'rigid'}, "
              f"Neumann BC)")
    write_dataset_info(info_path, sim_type=args.ic, cm=args.cm, res=Ny,
                       grid_res_cm=dx_cm, dt_ms=dt_ms, n_timesteps=N_T,
                       u_lo=u_lo, u_hi=u_hi, v_lo=v_lo, v_hi=v_hi,
                       t_in=args.t_in, t_out=args.t_out,
                       n_train_pairs=x_tr.shape[0], n_test_pairs=x_te.shape[0],
                       domain_label=domain)
    # Append a line so downstream tools know the channel layout.
    ch_names = "(u, v, δx, δy)" if args.deform else "(u, v)"
    with info_path.open("a") as f:
        f.write(f"Channel layout: {trajs_train.shape[1]} channels = {ch_names}\n")
        f.write(f"Undeformed element edge Δ (dx): {p.dx}\n")
        f.write(f"Reference node spacing (λ_init·dx): {LAMBDA_IN * p.dx}\n")
    print(f"Wrote {info_path}")

    plot_sanity(trajs_train[0, 0], trajs_train[0, 1],
                folder / f"sanity_{args.ic}.png", args.ic, dt_dump)
    print(f"Wrote {folder}/sanity_{args.ic}.png")


if __name__ == "__main__":
    main()
