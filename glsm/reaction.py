"""Modified 2-variable Oregonator in a polymer gel — eqs (6), (7) of Yashin-Balazs 2007.

F(u,v,φ) = (1-φ)² u - u² - (1-φ) f v · (u - q(1-φ)²) / (u + q(1-φ)²)
G(u,v,φ) = (1-φ)² u - (1-φ) v

These reduce to the familiar Tyson-Fife Oregonator when φ → 0.

The polymer volume fraction φ enters in two places:
  • as a diluent (every concentration rescaled by (1-φ) or (1-φ)²),
  • through the parameter χ* in the free energy, **not** here directly.

Both u and v are **mole fractions** defined per unit volume of the whole system.
Inside the gLSM they are stored as `u(m) = u_element / φ(m)` etc. (see
eq 54 onward) because fluxes are easier to express per polymer mass. This file
is agnostic about that choice: it just takes whatever numeric u, v, φ you pass.
"""
from __future__ import annotations

import numpy as np
import torch

from .params import Parameters


# ---------------------------------------------------------------------------
# NumPy implementations (for sanity checks and single-point tests)
# ---------------------------------------------------------------------------
def F_np(u, v, phi, p: Parameters):
    """Activator (HBrO2) source, eq (6). Supports array broadcasting."""
    one_minus_phi = 1.0 - phi
    sq = one_minus_phi ** 2
    q_eff = p.q * sq
    # Guard the 1/(u+q_eff) term with a tiny epsilon — the network / integrator
    # can transiently produce u slightly negative near wavefronts, and
    # dividing by ~0 would spuriously spike the residual. Physically u ≥ 0.
    safe = np.clip(u + q_eff, 1e-12, None)
    return sq * u - u * u - one_minus_phi * p.f * v * (u - q_eff) / safe


def G_np(u, v, phi, p: Parameters):
    """Catalyst (Mox) source, eq (7)."""
    one_minus_phi = 1.0 - phi
    return (one_minus_phi ** 2) * u - one_minus_phi * v


# ---------------------------------------------------------------------------
# PyTorch implementations (vectorised, autograd-compatible, GPU-ready)
# ---------------------------------------------------------------------------
def F_torch(u: torch.Tensor, v: torch.Tensor, phi: torch.Tensor,
            p: Parameters) -> torch.Tensor:
    one_minus_phi = 1.0 - phi
    sq = one_minus_phi * one_minus_phi
    q_eff = p.q * sq
    safe = (u + q_eff).clamp_min(1e-12)
    return sq * u - u * u - one_minus_phi * p.f * v * (u - q_eff) / safe


def G_torch(u: torch.Tensor, v: torch.Tensor, phi: torch.Tensor,
            p: Parameters) -> torch.Tensor:
    one_minus_phi = 1.0 - phi
    return (one_minus_phi * one_minus_phi) * u - one_minus_phi * v


# ---------------------------------------------------------------------------
# Stationary homogeneous solution (eq 27)
# ---------------------------------------------------------------------------
def stationary_uniform(phi: float, p: Parameters) -> tuple[float, float]:
    """Return (u_s, v_s) such that F=G=0 at the given uniform φ.

    From G=0:  v = (1-φ)·u.  Substitute into F=0 → solve quadratic in u.
    Used as Case II initial condition.
    """
    one_minus_phi = 1.0 - phi
    sq = one_minus_phi ** 2
    q_eff = p.q * sq
    # F(u, (1-φ)u, φ) = sq·u - u² - (1-φ)·f·(1-φ)·u·(u-q_eff)/(u+q_eff)
    #                 = sq·u - u² - sq·f·u·(u-q_eff)/(u+q_eff)
    # Setting that to zero and dividing by u (assuming u>0):
    #   sq - u = sq·f·(u-q_eff)/(u+q_eff)
    # Cross-multiplying:
    #   (sq - u)(u + q_eff) = sq·f·(u - q_eff)
    # Expand:
    #   sq·u + sq·q_eff - u² - q_eff·u = sq·f·u - sq·f·q_eff
    # Collect u² term: -u² + u(sq - q_eff - sq·f) + sq·q_eff·(1 + f) = 0
    # Multiply by -1:   u² + u(sq·f + q_eff - sq) - sq·q_eff·(1+f) = 0
    b = sq * p.f + q_eff - sq
    c = -sq * q_eff * (1.0 + p.f)
    disc = b * b - 4.0 * c
    assert disc > 0, f"No real stationary solution at φ={phi}"
    u_s = 0.5 * (-b + disc ** 0.5)
    v_s = one_minus_phi * u_s
    return float(u_s), float(v_s)
