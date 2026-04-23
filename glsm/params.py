"""Physical parameters for the 2D chemoresponsive gLSM.

Values taken directly from Yashin & Balazs 2007, Tables I and II, and from the
textual parameter choices in Section V "MODEL PARAMETERS".

All quantities are given in the dimensionless form used throughout the paper:
  - time in units of  T_0 = (k3 · H · A)^{-1} ≈ 1 s     (see p. 124707-11)
  - length in units of L_0 = sqrt(Du · T_0) ≈ 40 μm
  - reactant concentrations in Tyson–Fife mole-fraction form u = X/X_0, v = Z/Z_0
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Parameters:
    """Immutable parameter bundle so experiments can't accidentally mutate a global."""

    # --- Reaction kinetics (Oregonator, Table I + paper text p.124707-11) ---
    q: float = 9.52e-5       # Oregonator stiffness, dimensionless; fixed by rate constants
    epsilon: float = 0.354   # time-scale separation v vs u; from k5·B / (k3·H·A)
    f: float = 0.7           # stoichiometric factor — paper's "free parameter", chosen value

    # --- Flory–Huggins interaction (Table II, neutral NIPAAm gel, Ref. 31) ---
    chi_0: float = 0.338     # χ at T=20°C
    chi_1: float = 0.518     # linear φ-dependence of χ: χ(φ)=χ0+χ1·φ

    # --- Polymer–solvent coupling to BZ redox (adjustable) ---
    chi_star: float = 0.105  # strength of v-mediated hydration; 0 → nonresponsive gel

    # --- Gel preparation (Table II, grafted Ru(bpy)₃²⁺ gel, Ref. 21) ---
    phi_0: float = 0.139     # polymer volume fraction in undeformed/preparation state
    c0_v0: float = 1.3e-3    # dimensionless cross-link density c₀·v₀

    # --- Transport and mobility ---
    # D_u absorbed into the length scale L_0; nothing to set here.
    Lambda_0: float = 100.0  # dimensionless kinetic coefficient (p.124707-11)

    # --- Geometry / constraint ---
    lambda_perp: float = 1.1 # degree of swelling perpendicular to the XY plane
    # Δ: undeformed element edge length. Paper convention is Δ=1 (pure
    # non-dimensional form). Fortran temp/2D.f90 uses dx=0.5 — a different
    # physical length unit so dx², dx⁻² factors appear in M_n, the flux coef,
    # and the reference area of element_J_and_phi. Set dx=0.5 to reproduce
    # Fortran scaling exactly; dx=1.0 keeps the paper / Δ=1 convention.
    dx: float = 1.0


# Default bundle used by every run unless overridden.
DEFAULT = Parameters()


def as_dict(p: Parameters) -> dict:
    """Flat dict for logging / dataset_info files."""
    return {
        "q": p.q, "epsilon": p.epsilon, "f": p.f,
        "chi_0": p.chi_0, "chi_1": p.chi_1, "chi_star": p.chi_star,
        "phi_0": p.phi_0, "c0_v0": p.c0_v0, "Lambda_0": p.Lambda_0,
        "lambda_perp": p.lambda_perp, "dx": p.dx,
    }
