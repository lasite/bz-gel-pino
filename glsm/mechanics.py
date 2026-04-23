"""Elastic + osmotic mechanics of the 2D gLSM.

References are to equation numbers in Yashin & Balazs 2007.

State layout (NumPy convention, all float64):

    nodes   : (Ny+1, Nx+1, 2)  — laboratory-frame coordinates of node (k, l)
    phi     : (Ny,   Nx)       — polymer volume fraction per element, φ(m)
    u, v    : (Ny,   Nx)       — reactant mole fractions per element
                                (defined as u = u_total / φ, eq (54) onward)

Elements are labelled m = (k, l) with k ∈ [0, Ny), l ∈ [0, Nx). Element m has
corner nodes  (k,l), (k,l+1), (k+1,l+1), (k+1,l) — local labels 1,2,3,4
(see Fig. 2 of the paper).
"""
from __future__ import annotations

import numpy as np

from .params import Parameters


# ---------------------------------------------------------------------------
# Volume fraction from deformation (between eqs 46 and 47)
# ---------------------------------------------------------------------------
def element_J_and_phi(nodes: np.ndarray, p: Parameters) -> tuple[np.ndarray, np.ndarray]:
    """Compute the elementwise volumetric-change factor J and the volume fraction φ.

    J(m) = λ⊥ · |d1 × d2| / (2 · Δ²)
    where d1 = x3 - x1 is the (bottom-left → top-right) diagonal, d2 = x4 - x2
    is the other diagonal (eq 46), and Δ = p.dx is the undeformed element
    edge length. The paper uses Δ=1 so the denominator drops out; Fortran
    temp/2D.f90 uses Δ=0.5, and the ratio |cross|/2 is divided by dx² to
    match `area0 = LAMP²·dx²` (Fortran line 528) which keeps φ_init = φ_0
    at the equilibrium undeformed lattice.

    φ(m) = φ_0 / J(m).
    """
    # Corner nodes: shape (Ny, Nx, 2) each
    x1 = nodes[:-1, :-1]
    x2 = nodes[:-1, 1:]
    x3 = nodes[1:, 1:]
    x4 = nodes[1:, :-1]

    d1 = x3 - x1      # main diagonal
    d2 = x4 - x2      # anti-diagonal
    cross_z = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    dx2 = p.dx * p.dx
    J = 0.5 * p.lambda_perp * np.abs(cross_z) / dx2
    phi = p.phi_0 / J
    return J, phi


# ---------------------------------------------------------------------------
# Osmotic pressure (eq 25) and element pressure P(φ, v) (eq 24)
# ---------------------------------------------------------------------------
def osmotic_pressure(phi: np.ndarray, v: np.ndarray, p: Parameters) -> np.ndarray:
    """π_osm(φ, v) = -[φ + ln(1-φ) + χ(φ)·φ²] + χ*·v·φ  (eq 25)."""
    chi_phi = p.chi_0 + p.chi_1 * phi           # χ(φ) = χ0 + χ1·φ
    # Guard ln(1-φ) against φ → 1 (collapsed gel)
    one_minus_phi = np.clip(1.0 - phi, 1e-9, None)
    return -(phi + np.log(one_minus_phi) + chi_phi * phi * phi) + p.chi_star * v * phi


def pressure_P(phi: np.ndarray, v: np.ndarray, p: Parameters) -> np.ndarray:
    """Internal element pressure (eq 24).

    P = π_osm + c0·v0 · φ / (2·φ_0)
    The second term is the entropic contribution from cross-link density.
    """
    return osmotic_pressure(phi, v, p) + p.c0_v0 * phi / (2.0 * p.phi_0)


# ---------------------------------------------------------------------------
# Nodal force (eq 48)
# ---------------------------------------------------------------------------
# F_(k,l) = c0·v0 / 3 · Σ_(δ,δ'=0,±1) (x_(k+δ,l+δ') - x_(k,l))        [spring]
#         + λ⊥/2 · {
#             ez × (x_(k,l+1) - x_(k+1,l)) · P(k, l)
#           + ez × (x_(k-1,l)   - x_(k,l+1)) · P(k-1, l)
#           + ez × (x_(k,l-1)   - x_(k-1,l)) · P(k-1, l-1)
#           + ez × (x_(k+1,l)   - x_(k,l-1)) · P(k, l-1)
#           }                                                          [pressure]
# (the first factor H0 in eq 48 equates out against the corresponding factor
# H0 in the mobility, eq 53; we drop both as is standard in gLSM)
def nodal_force(nodes: np.ndarray, phi: np.ndarray, v: np.ndarray,
                p: Parameters) -> np.ndarray:
    """Return F_(k,l) for every node — shape (Ny+1, Nx+1, 2).

    NB: this computes the internal force. Fixed-wall BCs are applied by the
    caller by zeroing the force on pinned nodes **after** this returns.
    """
    Ny_nodes, Nx_nodes, _ = nodes.shape
    Ny = Ny_nodes - 1
    Nx = Nx_nodes - 1

    # ---- Spring contribution: sum over the 8 neighbours (edges + diagonals) ----
    # Equation 48 first line: coefficient c0·v0/3 times Σ (x_neighbour - x_centre).
    # Natural domain-edge convention: at the boundary, "missing neighbours" are
    # simply absent from the sum (they correspond to element edges shared with
    # vacuum, and do not contribute a spring). We achieve this by shifting with
    # zero padding and zero-weighting the shifted-into-boundary entries.
    F = np.zeros_like(nodes)
    for dk, dl in [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]:
        # Build shifted coordinates: x_(k+dk, l+dl), NaN-safe via in-domain mask
        k_lo = max(0, -dk);  k_hi = Ny_nodes - max(0, dk)
        l_lo = max(0, -dl);  l_hi = Nx_nodes - max(0, dl)
        src_k0 = max(0, dk); src_k1 = Ny_nodes - max(0, -dk)
        src_l0 = max(0, dl); src_l1 = Nx_nodes - max(0, -dl)
        delta = nodes[src_k0:src_k1, src_l0:src_l1] - nodes[k_lo:k_hi, l_lo:l_hi]
        F[k_lo:k_hi, l_lo:l_hi] += delta
    F *= p.c0_v0 / 3.0

    # ---- Pressure contribution (second line of eq 48) ----
    # Each node (k,l) collects one term from each of up-to-four surrounding
    # elements m = (k,l), (k-1,l), (k-1,l-1), (k,l-1). We pad P with 0 on the
    # out-of-domain side so that missing elements contribute nothing; this is
    # physically consistent — there is no element to supply pressure there.
    P = pressure_P(phi, v, p)
    P_pad = np.pad(P, ((1, 1), (1, 1)), mode="constant", constant_values=0.0)
    # After padding, P_pad[k+1, l+1] == P[k, l], so element m=(k,l) maps to
    # P_pad slice [k+1, l+1]. The four surrounding elements at NODE (k,l) are:
    #   upper-right: m = (k,   l  ) → P_pad[k+1, l+1]
    #   upper-left : m = (k,   l-1) → P_pad[k+1, l  ]
    #   lower-right: m = (k-1, l  ) → P_pad[k,   l+1]
    #   lower-left : m = (k-1, l-1) → P_pad[k,   l  ]
    # Shape (Ny+1, Nx+1) for each.
    Pur = P_pad[1:, 1:]
    Pul = P_pad[1:, :-1]
    Plr = P_pad[:-1, 1:]
    Pll = P_pad[:-1, :-1]

    # For the cross products we need coordinates of neighbouring nodes. At the
    # domain boundary those don't exist — but in those cases the corresponding
    # element's pressure is zero (pad) so the term drops out; we just pad x with
    # zeros (any value really) for shape compatibility.
    xp = np.pad(nodes, ((1, 1), (1, 1), (0, 0)), mode="edge")
    # xp[k+1, l+1] == nodes[k, l]; shifts as before.
    def X(dk, dl):
        # xp[k+1+dk, l+1+dl] over valid node range (k,l) ∈ [0, Ny] × [0, Nx]
        return xp[1 + dk:1 + dk + Ny_nodes, 1 + dl:1 + dl + Nx_nodes]

    # Edge vectors γ_s for each element — as used in eq 48, the four pressure
    # terms at node (k,l) contain:
    #   (x_(k,l+1) - x_(k+1,l))   × e3 · P(k,   l)   [element at upper-right of the node
    #                                                 when reading eq 48 literally]
    #   (x_(k-1,l) - x_(k,l+1))   × e3 · P(k-1, l)
    #   (x_(k,l-1) - x_(k-1,l))   × e3 · P(k-1, l-1)
    #   (x_(k+1,l) - x_(k,l-1))   × e3 · P(k,   l-1)
    #
    # The paper writes "e3 × (x_a - x_b)" in eq (48). Literal evaluation
    # e3 × v = (-v_y, v_x) gives a force pointing INTO the element at a
    # lower-left corner under positive osmotic pressure — unphysical. The
    # correct outward-pointing force comes from (v × e3) = (v_y, -v_x).
    # The sign discrepancy is a convention mismatch between "e3" treated as
    # an out-of-page unit vector vs into-the-page. Both conventions appear
    # in the gel-LSM literature; we adopt the one that produces physical
    # (outward) force for positive pressure.
    def cross_z(a):
        out = np.empty_like(a)
        out[..., 0] = a[..., 1]
        out[..., 1] = -a[..., 0]
        return out

    term1 = cross_z(X(0, 1)  - X(1, 0))  * Pur[..., None]
    term2 = cross_z(X(-1, 0) - X(0, 1))  * Plr[..., None]
    term3 = cross_z(X(0, -1) - X(-1, 0)) * Pll[..., None]
    term4 = cross_z(X(1, 0)  - X(0, -1)) * Pul[..., None]
    F += 0.5 * p.lambda_perp * (term1 + term2 + term3 + term4)
    return F


# ---------------------------------------------------------------------------
# Nodal mobility (eq 53)
# ---------------------------------------------------------------------------
# M_m = 4 · Λ_0 · (1 - φ̄_m)^{-1} · (φ̄_m/φ_0)^{-1/2} · Δ^-2
# where φ̄_m is the mean of φ over the up-to-four adjoining elements.
# (H_0-factor cancels with F_m; Δ=1 here.)
def nodal_mobility(phi: np.ndarray, p: Parameters) -> np.ndarray:
    """Return M_(k,l) at every node — shape (Ny+1, Nx+1)."""
    Ny, Nx = phi.shape
    # Sum of adjacent elements and their count (1–4) per node
    phi_pad = np.pad(phi, ((1, 1), (1, 1)), mode="constant", constant_values=0.0)
    mask_pad = np.pad(np.ones_like(phi), ((1, 1), (1, 1)),
                      mode="constant", constant_values=0.0)

    # Element m=(k,l) contributes to NODES (k,l), (k,l+1), (k+1,l), (k+1,l+1)
    # i.e. for each node (K, L), the 4 contributing elements are
    #   (K-1,L-1), (K-1,L), (K,L-1), (K,L)
    # (with out-of-domain elements absent).
    s00 = phi_pad[:-1, :-1];  m00 = mask_pad[:-1, :-1]
    s01 = phi_pad[:-1, 1:];   m01 = mask_pad[:-1, 1:]
    s10 = phi_pad[1:, :-1];   m10 = mask_pad[1:, :-1]
    s11 = phi_pad[1:, 1:];    m11 = mask_pad[1:, 1:]
    phi_sum = s00 + s01 + s10 + s11
    cnt = (m00 + m01 + m10 + m11).clip(min=1.0)
    phi_bar = phi_sum / cnt       # shape (Ny+1, Nx+1)

    # Clip inside (0, 1) for numerical safety at free edges where cnt<4 and
    # average could otherwise overshoot.
    phi_bar = phi_bar.clip(1e-3, 1.0 - 1e-3)
    # Paper Eq. 53: M_m = 4·H_0⁻¹·Δ⁻²·Λ_0·(1 − φ̄)·(φ̄/φ_0)⁻¹ᐟ². The H_0
    # prefactor is absorbed (it cancels against the force). Δ = p.dx is
    # explicit here so the mobility matches Fortran (line 458:
    # mn = 4/dx² · AZ0 · (1-w_mean) · (w_mean/FA0)^(-1/2)).
    # Note: (1−φ̄) appears with EXPONENT +1 in the mobility (so M → 0 as
    # φ̄ → 1, gel near collapse moves slowly).
    return (4.0 / (p.dx * p.dx) * p.Lambda_0
            * (1.0 - phi_bar) * (phi_bar / p.phi_0) ** (-0.5))
