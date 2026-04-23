"""Explicit-Euler time integration of the 2D gLSM.

The state comprises:
    nodes   (Ny+1, Nx+1, 2)  — laboratory-frame coords of every node
    u_elem  (Ny, Nx)         — activator mole fraction on each element
    v_elem  (Ny, Nx)         — catalyst mole fraction on each element

φ is a derived quantity (eq 46). u and v are stored as true mole fractions
per the paper's convention (eq 54): the chemistry inside element m evolves by
    d u_elem / dt = (flux RHS, eq 57, 60, 61) + φ(m)^{-1} · F(u_elem/φ, ...)
    d v_elem / dt = ε G(...) / φ(m)

Wait — actually the paper (eq 57, 58) keeps u, v as mole fractions but the
RHS contains F, G directly (not divided by φ). The advection terms are
rewritten specially. We follow eq 57/58 literally.

Wall BC (Case I): the left column of nodes is pinned; u-flux across the left
element edges is zero (no diffusive exchange with solvent outside the wall).
Every other boundary is free.

No regularisation / limiting: pure explicit Euler with an operator-style split
(reaction, then flux, then mechanics). For the stiffness of the Oregonator we
sub-step the reaction every outer Δt.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .params import Parameters
from . import reaction as rxn
from . import mechanics as mech
from . import diffusion as dif

# Reference undeformed element edge in paper/Yashin units (λ_init-scaled).
# temp/2D.f90 uses dx=0.5, but that is their free scaling; paper Eq 57
# convention is Δ=1 (the undeformed element edge equals the length unit).
# Our generate_datasets.LAMBDA_IN=1.1 is the initial in-plane swelling,
# which enters the jsp/jsu prefactor as λ_init² through the `r_m` distances.
LAMBDA_IN = 1.1


@dataclass
class State:
    """Container for the simulation state."""
    nodes: np.ndarray       # (Ny+1, Nx+1, 2)
    u: np.ndarray           # (Ny, Nx) mole fraction of activator
    v: np.ndarray           # (Ny, Nx) mole fraction of catalyst
    t: float = 0.0          # dimensionless time (units of T_0)
    # Previous-step node velocity. Fortran carries this as `veno` and uses
    # rn += 0.5·dt·(ven + veno) — 2nd-order trapezoidal. If None on entry
    # (first step), it is treated as zero (equivalent to forward Euler for
    # the very first step, then trapezoidal from step 2 onward — same as
    # Fortran's `veno = 0` initialization).
    vel_prev: np.ndarray | None = None

    def copy(self) -> "State":
        vp = None if self.vel_prev is None else self.vel_prev.copy()
        return State(self.nodes.copy(), self.u.copy(), self.v.copy(),
                     float(self.t), vp)


# ---------------------------------------------------------------------------
# Initial state builders
# ---------------------------------------------------------------------------
def make_case_I(Ny: int, Nx: int, p: Parameters,
                lambda_init: float | None = None,
                u_init_scale: float = 1e-3) -> State:
    """Build Case I initial state: small initial gel size + small u.

    λ_init defaults to p.lambda_perp=1.1. u_init = u_init_scale · φ_0 · λ_init^{-3}
    (paper footnote-type note: "initial concentrations of BZ reagents ...
    u = 10⁻³·φ_0·λ⁻³").  v_init = 0.
    """
    if lambda_init is None:
        lambda_init = p.lambda_perp
    ks, ls = np.meshgrid(np.arange(Ny + 1), np.arange(Nx + 1), indexing="ij")
    nodes = np.stack([ls * lambda_init, ks * lambda_init], axis=-1).astype(np.float64)
    phi_init = p.phi_0 / (p.lambda_perp * lambda_init ** 2)
    u = np.full((Ny, Nx), u_init_scale * p.phi_0 * lambda_init ** -3, dtype=np.float64)
    v = np.zeros((Ny, Nx), dtype=np.float64)
    return State(nodes=nodes, u=u, v=v)


def make_case_II(Ny: int, Nx: int, p: Parameters,
                 lambda_init: float = 2.02) -> State:
    """Case II: large initial gel size + uniform stationary BZ concentrations."""
    ks, ls = np.meshgrid(np.arange(Ny + 1), np.arange(Nx + 1), indexing="ij")
    nodes = np.stack([ls * lambda_init, ks * lambda_init], axis=-1).astype(np.float64)
    phi_init = p.phi_0 / (p.lambda_perp * lambda_init ** 2)
    us, vs = rxn.stationary_uniform(phi_init, p)
    u = np.full((Ny, Nx), us, dtype=np.float64)
    v = np.full((Ny, Nx), vs, dtype=np.float64)
    return State(nodes=nodes, u=u, v=v)


# ---------------------------------------------------------------------------
# One outer step (Δt): operator-split reaction → flux → mechanics
# ---------------------------------------------------------------------------
def step(state: State, dt: float, p: Parameters, *,
         reaction_substeps: int = 20,
         mech_substeps: int = 1,
         pin_left_wall: bool = True,
         block_left_u_flux: bool = True,
         rigid_mesh: bool = False,
         d_scale: float = 1.0,
         periodic: str = "") -> State:
    """Advance the simulation by dt (measured in T_0).

    Integration scheme (operator split):
      1. Advance reaction inside each element for `reaction_substeps`
         sub-steps (stiff Oregonator needs small dt).
      2. Advance the diffusion + mechanics coupled system for `mech_substeps`
         forward-Euler sub-steps. The paper (Yashin-Balazs 2007, p.124707-12)
         uses VODE — an implicit stiff solver — for this block; here we fall
         back to explicit Euler with sub-stepping. With Λ_0=100 the CFL-like
         stability bound is roughly dt_mech < 1/Λ_0 ≈ 0.01 T_0, so for outer
         dt≈5e-3 T_0 choose mech_substeps ≥ 1 and increase if the run blows up.

    If `rigid_mesh=True`, node positions are frozen (all velocities set to
    zero); in that case `mech_substeps=1` is always fine because no force
    is integrated.
    """
    Ny, Nx = state.u.shape

    # Strict port of temp/2D.f90::CHEM (lines 573-711) + CHEM_BOUNDARY
    # (739-761). Order of operations per Fortran gel_dynamic() main loop:
    #   1. compute pressure P from current (φ, v)
    #   2. compute node force F_n, then velocity ven = M_n · F_n
    #   3. advance nodes: rn += dtx · ven  (dtx=dt here, no multirate split)
    #   4. recompute φ_new from new nodes  → jsv = 1 − φ_new/φ_old
    #   5. interpolate u, φ, v to nodes (un, wn, vn)
    #   6. compute jsp (Fortran line 617-636), jsu (line 643-662)
    #   7. RK4 reaction on (u, v)          (line 697-706)
    #   8. final update:
    #      u = u + RK4_u·dt − u·jsv − dt·(λ⊥/(λ_init²·Δ²·φ₀))·(jsp+jsu)·φ_new
    #      v = v + RK4_v·dt − v·jsv
    _, phi_old = mech.element_J_and_phi(state.nodes, p)
    u0, v0 = state.u.copy(), state.v.copy()

    # ------ 2-3. mechanics: move nodes (paper Eq 49) ------------------
    # Fortran line 462: rn += 0.5·dt·(ven + veno)  — 2nd-order trapezoidal.
    # We mirror that exactly: compute the current-step velocity on the PRE-
    # move geometry, then advance nodes with the mean of current and
    # previous-step velocity.
    if rigid_mesh:
        nodes_new = state.nodes
        vel = np.zeros_like(state.nodes)
        phi_new = phi_old
    else:
        F_node = mech.nodal_force(state.nodes, phi_old, v0, p)
        M_node = mech.nodal_mobility(phi_old, p)
        vel = M_node[..., None] * F_node
        if pin_left_wall:
            vel[:, 0, :] = 0.0
        vel_prev = (np.zeros_like(vel)
                    if state.vel_prev is None else state.vel_prev)
        # mech_substeps > 1 is kept for optional extra CFL margin; divides
        # the trapezoidal motion into N equal pieces (simple refinement).
        dt_m = dt / mech_substeps
        nodes_new = state.nodes.copy()
        for _ in range(mech_substeps):
            nodes_new = nodes_new + 0.5 * dt_m * (vel + vel_prev)
        _, phi_new = mech.element_J_and_phi(nodes_new, p)

    # ------ 4. jsv (Fortran line 687) --------------------------------
    jsv = 1.0 - phi_new / np.clip(phi_old, 1e-6, None)

    # ------ 5-6. flux jsp+jsu (Fortran lines 617-662) ----------------
    if periodic and rigid_mesh:
        # Fast path kept for the frozen rigid-stable dataset layout.
        lam = float(state.nodes[1, 0, 1] - state.nodes[0, 0, 1])
        u_flux = dif.u_flux_rhs_periodic_rigid(u0, phi_old, lam, p, axes=periodic)
        if d_scale != 1.0:
            u_flux = u_flux * d_scale
        jsp_sum = None  # sentinel: use u_flux directly in update below
    else:
        # Literal Fortran port: node-pair J^(p) plus element-centred J^(u).
        jsp, jsu = dif.jsp_jsu_fortran(u0, phi_new, nodes_new, vel)
        # Fortran line 705 coefficient: -(LAMV/(LAMP²·dx²·FA0))·(jsp+jsu)·w
        # In paper conventions LAMV=λ_perp, LAMP=λ_init, FA0=φ₀, Δ=p.dx.
        coef = -p.lambda_perp / (
            LAMBDA_IN * LAMBDA_IN * p.dx * p.dx * p.phi_0
        )
        u_flux = coef * (jsp + jsu) * phi_new
        if d_scale != 1.0:
            u_flux = u_flux * d_scale
        jsp_sum = (jsp, jsu)

    # ------ 7. RK4 reaction on (u, v) (Fortran line 697-706) ---------
    # All four K/L stages are evaluated with the PRE-step φ (=phi_old), the
    # same convention Fortran uses: reaction dependence on φ is frozen
    # during the RK4 step; φ only changes via the mechanics step.
    def F(uu, vv): return rxn.F_np(uu, vv, phi_old, p)
    def G(uu, vv): return rxn.G_np(uu, vv, phi_old, p) * p.epsilon
    K1 = F(u0, v0);                       L1 = G(u0, v0)
    K2 = F(u0 + dt*K1/2, v0 + dt*L1/2);   L2 = G(u0 + dt*K1/2, v0 + dt*L1/2)
    K3 = F(u0 + dt*K2/2, v0 + dt*L2/2);   L3 = G(u0 + dt*K2/2, v0 + dt*L2/2)
    K4 = F(u0 + dt*K3,   v0 + dt*L3);     L4 = G(u0 + dt*K3,   v0 + dt*L3)
    reaction_u = (K1 + 2*K2 + 2*K3 + K4) * (dt / 6.0)
    reaction_v = (L1 + 2*L2 + 2*L3 + L4) * (dt / 6.0)

    # ------ 8. final update (Fortran line 705-706) -------------------
    u_new = np.clip(u0 + reaction_u - u0 * jsv + dt * u_flux, 0.0, 1.5)
    v_new = np.clip(v0 + reaction_v - v0 * jsv,                0.0, 1.5)

    # `reaction_substeps` is no longer used in the Fortran-strict path —
    # kept in the signature for backwards compatibility. RK4 single step
    # matches Fortran exactly.
    _ = reaction_substeps

    # Save current velocity for the next step's trapezoidal update
    # (Fortran lines 465-466: veno(...) = ven(...)).
    return State(nodes=nodes_new, u=u_new, v=v_new, t=state.t + dt,
                 vel_prev=(vel.copy() if not rigid_mesh else None))


# ---------------------------------------------------------------------------
# Pacemaker / boundary-pulse callback
# ---------------------------------------------------------------------------
# Analogue of the CardiacEP-PINOS `pacemaker` and `boundary_pulse` mechanisms
# in `2D_Waves_AP.py / run_simulation`. Every `period` T_0 the activator u and
# catalyst v inside `mask` are reset to (u_set, v_set), but only if the region
# is currently "quiescent" (proxy: v_mean inside the mask below `gate_v`).
# The guard prevents re-firing on top of an active wave and matches the
# "if sim.v[cx, cy] < 0.10" check in the reference script.
def _maybe_fire(u: np.ndarray, v: np.ndarray, pacemaker: dict, t: float) -> bool:
    """Re-excite `pacemaker['mask']` if the next scheduled firing has been
    crossed AND the region is quiescent. Mutates u/v in place, returns True
    if firing happened so the caller can advance the next firing time."""
    mask = pacemaker["mask"]
    if pacemaker.get("gate_v") is not None:
        if v[mask].mean() > pacemaker["gate_v"]:
            return False
    u[mask] = pacemaker.get("u_set", 0.9)
    v[mask] = pacemaker.get("v_set", 0.0)
    return True


# ---------------------------------------------------------------------------
# State I/O (analogue of openCARP `.roe` start states)
# ---------------------------------------------------------------------------
def save_state(state: State, path) -> None:
    """Persist a full State (nodes + u + v + t) to an .npz file so it can be
    used as a warm-start for later runs (analogue of CardiacEP-PINOS `.roe`
    start-state files in `datasets_start_states/`)."""
    np.savez(str(path), nodes=state.nodes, u=state.u, v=state.v,
             t=np.array([state.t], dtype=np.float64))


def load_state(path) -> State:
    """Load a State written by `save_state`."""
    d = np.load(str(path))
    return State(nodes=d["nodes"], u=d["u"], v=d["v"], t=float(d["t"][0]))


# ---------------------------------------------------------------------------
# Trajectory driver
# ---------------------------------------------------------------------------
def run(state: State, p: Parameters, *, dt: float, n_steps: int,
        snapshot_every: int = 100,
        pacemaker: dict | None = None,
        clamp: dict | None = None,
        **step_kwargs) -> dict[str, np.ndarray]:
    """Run the simulation for n_steps and return snapshots at stride
    snapshot_every.

    `pacemaker` (optional, pulsed):
        {'mask', 'period', 'u_set', 'v_set', 'gate_v'} — re-excite masked
        region at fixed intervals when locally quiescent.

    `clamp` (optional, continuous Dirichlet until release):
        {'mask', 'u_value', 'v_value', 'release_at'} — hold u (and optionally
        v) at fixed values in the masked region every step, up until
        `release_at` (in T_0 units). After the release, the clamp is dropped
        and dynamics proceed freely. Useful for pre-pacing a stable planar
        wave train before recording the dataset period (analogue of
        CardiacEP-PINOS `prepacing_beats`).
    """
    Ny, Nx = state.u.shape
    n_snaps = n_steps // snapshot_every + 1
    snaps = {
        "t":     np.empty(n_snaps, dtype=np.float64),
        "nodes": np.empty((n_snaps, Ny + 1, Nx + 1, 2), dtype=np.float64),
        "u":     np.empty((n_snaps, Ny, Nx), dtype=np.float64),
        "v":     np.empty((n_snaps, Ny, Nx), dtype=np.float64),
        "phi":   np.empty((n_snaps, Ny, Nx), dtype=np.float64),
    }
    snaps["t"][0] = state.t
    snaps["nodes"][0] = state.nodes
    snaps["u"][0] = state.u
    snaps["v"][0] = state.v
    _, snaps["phi"][0] = mech.element_J_and_phi(state.nodes, p)

    next_fire = state.t + pacemaker["period"] if pacemaker else None

    si = 1
    cur = state
    for i in range(1, n_steps + 1):
        cur = step(cur, dt, p, **step_kwargs)
        # Pacemaker: pulsed re-excitation (only when region is quiescent).
        if pacemaker is not None and cur.t >= next_fire - 1e-9:
            _maybe_fire(cur.u, cur.v, pacemaker, cur.t)
            next_fire += pacemaker["period"]
        # Clamp: continuous Dirichlet until release. Applied every step so the
        # region never drifts away from (u_value, v_value), which reliably
        # emits periodic planar wave fronts during the pre-pace window.
        if clamp is not None and cur.t < clamp["release_at"] - 1e-9:
            cur.u[clamp["mask"]] = clamp.get("u_value", 0.9)
            if clamp.get("v_value") is not None:
                cur.v[clamp["mask"]] = clamp["v_value"]
        if i % snapshot_every == 0:
            snaps["t"][si] = cur.t
            snaps["nodes"][si] = cur.nodes
            snaps["u"][si] = cur.u
            snaps["v"][si] = cur.v
            _, snaps["phi"][si] = mech.element_J_and_phi(cur.nodes, p)
            si += 1
    return {k: v[:si] for k, v in snaps.items()}
