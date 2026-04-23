"""Interdiffusion flux of the BZ reactant u across element edges.

Paper eqs (59)–(61).

Each element m has 4 edges, labelled s = 1..4 anticlockwise (see Fig. 2(a)):
    s=1: bottom edge, between nodes 1 and 2   (local), i.e. between element m and m_below
    s=2: right edge,  between nodes 2 and 3                                  m_right
    s=3: top edge,    between nodes 3 and 4                                  m_above
    s=4: left edge,   between nodes 4 and 1                                  m_left

The edge vector γ_s(m) = x_{n2} - x_{n1} runs counterclockwise around the element.

Eq (59):   J_s(m) = J_s^(p)(m) + J_s^(u)(m)
Eq (60):   J_s^(p)(m) = -½ [ u̅_s(m) / (1 - φ̅_s(m)) · v_s(m) + u̅_{s+1}(m) / (1 - φ̅_{s+1}(m)) · v_{s+1}(m) ]
                      (bar = average of the two end nodes of edge s)
Eq (61):   J_s^(u)(m) = -[1 - (φ(m)+φ(m'))/2] · [ u(m')/(1-φ(m')) - u(m)/(1-φ(m)) ] · (r(m')-r(m)) / |r(m')-r(m)|²
                      where m' is the neighbour across edge s.

In our 2D integration loop it is more natural to compute the u-SOURCE (net
flux INTO element m) rather than each edge flux. That is done by

    (du/dt)_flux(m) = - λ⊥ φ_0^{-1} Δ^{-2} Σ_{s=1}^{4}  e_3 · [ J_s(m) × γ_s(m) ]
                                                                   (eq 57)

We assemble this directly as the difference of interdiffusion fluxes across
pairs of elements (pushforward form), and the polymer-advection contribution
from the four corner node velocities.

Convention: we work with `u_elem = u(m)`, the element-wise mole fraction (see
eq 54). Paper's `u̅_s` on an edge is the average of the *nodal* u-values — but
in our lumped-element representation u lives on elements, not nodes. We
approximate the edge u̅_s(m) by a mixture of the values on the two elements
that share the edge:
    u̅_edge ≈ ½ (u(m) + u(m'))
This is the usual upwind-free edge-centered value used in finite-volume methods.
"""
from __future__ import annotations

import numpy as np

from .params import Parameters


# ---------------------------------------------------------------------------
# Strict Fortran port — matches temp/2D.f90::CHEM + CHEM_BOUNDARY
# ---------------------------------------------------------------------------
def _interp_elem_to_nodes(u_elem: np.ndarray) -> np.ndarray:
    """Interpolate an element-centred field to nodes using 4-element averaging
    with Neumann (edge) reflection at the domain boundary.

    u_elem  : (Ny, Nx)
    returns : (Ny+1, Nx+1)

    Mirrors the de-facto rule used in temp/2D.f90 CHEM_BOUNDARY (lines 724-761):
    u_boundary elements are set by Neumann reflection first, then every node
    is the average of its four surrounding element cells. Corner and edge
    nodes thus degenerate to the single-element or two-element average.
    """
    pad = np.pad(u_elem, ((1, 1), (1, 1)), mode="edge")  # (Ny+2, Nx+2)
    return 0.25 * (pad[:-1, :-1] + pad[:-1, 1:] + pad[1:, :-1] + pad[1:, 1:])


def jsp_jsu_fortran(u_elem: np.ndarray, phi_elem: np.ndarray,
                    nodes: np.ndarray, node_velocity: np.ndarray,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Polymer-advection (J^(p)) and self-diffusion (J^(u)) contributions,
    element-by-element, exactly as in temp/2D.f90::CHEM (lines 602-662).

    Parameters
    ----------
    u_elem, phi_elem : (Ny, Nx)
    nodes            : (Ny+1, Nx+1, 2)      [..., 0]=x, [..., 1]=y
    node_velocity    : (Ny+1, Nx+1, 2)

    Returns
    -------
    jsp, jsu : (Ny, Nx) each
        Summed over the 4 edges surrounding each element; jsp already has the
        -½ factor from Eq 60 baked in. Caller combines them with the
        prefactor -λ⊥/(λ_init²·Δ²·φ₀) · w and adds to the explicit-Euler u
        update (see simulator.step).
    """
    Ny, Nx = u_elem.shape
    # -- interpolate u, φ to nodes --------------------------------------
    u_n   = _interp_elem_to_nodes(u_elem)             # (Ny+1, Nx+1)
    phi_n = _interp_elem_to_nodes(phi_elem)
    one_minus_phi_n = np.clip(1.0 - phi_n, 1e-6, None)
    u_over_1mp_n = u_n / one_minus_phi_n              # u / (1-φ) at nodes

    # node velocities, components
    vx_n = node_velocity[..., 0]
    vy_n = node_velocity[..., 1]
    # product: (u/(1-φ)) · v_node  split by component
    qx = u_over_1mp_n * vx_n                          # (Ny+1, Nx+1)
    qy = u_over_1mp_n * vy_n

    # Element (k, l)'s four corner nodes are nodes[k, l], [k, l+1], [k+1, l+1],
    # [k+1, l]. Naming mirrors Fortran's n1=(xi, yi), n2=(xi+1, yi),
    # n3=(xi+1, yi+1), n4=(xi, yi+1) — corners counterclockwise starting from
    # bottom-left. Edges: s1=n1→n2 (bottom), s2=n2→n3 (right),
    # s3=n3→n4 (top), s4=n4→n1 (left).
    n1 = slice(None, -1),  slice(None, -1)   # (k, l)
    n2 = slice(None, -1),  slice(1, None)    # (k, l+1)
    n3 = slice(1, None),   slice(1, None)    # (k+1, l+1)
    n4 = slice(1, None),   slice(None, -1)   # (k+1, l)

    # Fortran ax_s / ay_s (lines 617-624): sum of the two node values of edge s
    ax1 = qx[n1] + qx[n2];  ay1 = qy[n1] + qy[n2]
    ax2 = qx[n2] + qx[n3];  ay2 = qy[n2] + qy[n3]
    ax3 = qx[n3] + qx[n4];  ay3 = qy[n3] + qy[n4]
    ax4 = qx[n4] + qx[n1];  ay4 = qy[n4] + qy[n1]

    # Fortran bx_s / by_s (lines 626-633): edge vector γ_s = x_{n_{s+1}} - x_{n_s}
    rx = nodes[..., 0]
    ry = nodes[..., 1]
    bx1 = rx[n2] - rx[n1];  by1 = ry[n2] - ry[n1]
    bx2 = rx[n3] - rx[n2];  by2 = ry[n3] - ry[n2]
    bx3 = rx[n4] - rx[n3];  by3 = ry[n4] - ry[n3]
    bx4 = rx[n1] - rx[n4];  by4 = ry[n1] - ry[n4]

    # Fortran jsp (lines 635-636): sum of e3·(a × b) over 4 edges, then -½.
    jsp = (ax1 * by1 - ay1 * bx1
         + ax2 * by2 - ay2 * bx2
         + ax3 * by3 - ay3 * bx3
         + ax4 * by4 - ay4 * bx4)
    jsp = -0.5 * jsp                                   # (Ny, Nx)

    # -- self-diffusion jsu (Fortran lines 643-662) ---------------------
    # Element-centre positions rm[k, l] = mean of 4 corner nodes.
    rm_x = 0.25 * (rx[n1] + rx[n2] + rx[n3] + rx[n4])
    rm_y = 0.25 * (ry[n1] + ry[n2] + ry[n3] + ry[n4])

    # Neighbours: (k, l-1) = south, (k, l+1) = north, (k+1, l) = east,
    #            (k-1, l) = west (in (k, l) = (row, col) convention, but
    # Fortran's loops use (xi, yi) where xi→col, yi→row, so what Fortran
    # calls "south neighbour at yi-1" = our (k-1, l). We mirror Fortran
    # literally by labelling Fortran's "yi" → our row "k":
    #   yi → k; xi → l.
    # So Fortran "(xi, yi-1)" = our (k-1, l)  — NORTH in Python's (row, col)
    # but SOUTH in Fortran's user-facing coordinates. We just follow the
    # Fortran naming to avoid sign confusion.
    one_minus_phi = np.clip(1.0 - phi_elem, 1e-6, None)
    u_over_elem = u_elem / one_minus_phi

    # Edge-and-neighbour index pairs (Fortran's 4 edges):
    # edge 1 → neighbour at (xi, yi-1)      = (k-1, l)
    # edge 2 → neighbour at (xi+1, yi)      = (k,   l+1)
    # edge 3 → neighbour at (xi, yi+1)      = (k+1, l)
    # edge 4 → neighbour at (xi-1, yi)      = (k,   l-1)
    def shifted(a, dk, dl):
        """Return array shifted by (dk, dl), with edge-padded boundary
        (so flux vanishes across the domain edge via the identity
        a[-1]-a[-1]=0, similar to Fortran's Neumann BC on um)."""
        if dk != 0:
            a = np.roll(np.pad(a, ((1, 1), (0, 0)), mode="edge"),
                        -dk, axis=0)[1:-1]
        if dl != 0:
            a = np.roll(np.pad(a, ((0, 0), (1, 1)), mode="edge"),
                        -dl, axis=1)[:, 1:-1]
        return a

    def jsu_term(dk: int, dl: int, bx_edge: np.ndarray, by_edge: np.ndarray):
        """(1 − ½(φ+φ'))·(u'/(1-φ') - u/(1-φ)) · (cx·by − cy·bx) / rr
        mirroring Fortran's jsu1..jsu4.
        """
        uop_nbr = shifted(u_over_elem, dk, dl)
        phi_nbr = shifted(phi_elem,   dk, dl)
        rm_x_nbr = shifted(rm_x, dk, dl)
        rm_y_nbr = shifted(rm_y, dk, dl)
        cx = rm_x_nbr - rm_x
        cy = rm_y_nbr - rm_y
        rr = cx * cx + cy * cy + 1e-18
        one_minus_meanphi = 1.0 - 0.5 * (phi_elem + phi_nbr)
        du = uop_nbr - u_over_elem
        return one_minus_meanphi * du * (cx * by_edge - cy * bx_edge) / rr

    jsu1 = jsu_term(-1, 0, bx1, by1)   # (xi, yi-1) ↔ Fortran edge 1
    jsu2 = jsu_term( 0, 1, bx2, by2)   # (xi+1, yi) ↔ edge 2
    jsu3 = jsu_term( 1, 0, bx3, by3)   # (xi, yi+1) ↔ edge 3
    jsu4 = jsu_term( 0,-1, bx4, by4)   # (xi-1, yi) ↔ edge 4
    jsu = -(jsu1 + jsu2 + jsu3 + jsu4)

    return jsp, jsu


# ---------------------------------------------------------------------------
# Fast path for a rigid, uniform, periodic mesh
# ---------------------------------------------------------------------------
# When the mesh is rigid (no advection, J^(p)=0) AND uniform (constant element
# spacing Δ' = λ_init in both axes) AND periodic, the full eq(57) contraction
# reduces to a 5-point periodic Laplacian of u/(1-φ) weighted by (1-φ_bar).
# For uniform φ (our dataset-generation setting) this simplifies further to a
# plain Laplacian of u with coefficient λ⊥/φ_0 · (1-φ) / Δ'².
#
# This helper returns that clean, faithful-to-eq(57) discretisation with
# periodic wraparound on whichever axes are requested — correct boundary
# physics for training data (no Neumann reflection, no pacemaker artefacts).
def u_flux_rhs_periodic_rigid(u_elem: np.ndarray, phi: np.ndarray,
                              lambda_init: float, p: Parameters,
                              axes: str = "xy") -> np.ndarray:
    """5-point Laplacian of `u/(1-φ)` weighted by `(1-φ_bar)` with periodic
    wrap on the axes named in `axes` (any subset of "x", "y"). For
    rigid uniform mesh only.

    axes="xy": both axes periodic (torus)
    axes="x":  x-axis periodic (y is Neumann, matching u_flux_rhs default)
    axes="y":  y-axis periodic (x is Neumann)
    axes="":   no periodicity — equivalent to Neumann everywhere
    """
    one_minus_phi = np.clip(1.0 - phi, 1e-6, None)
    q = u_elem / one_minus_phi

    # Neighbour views via np.roll (periodic); then zero the wrap difference on
    # any axis we want to keep Neumann.
    q_up    = np.roll(q, -1, axis=0)   # k+1
    q_down  = np.roll(q,  1, axis=0)   # k-1
    q_right = np.roll(q, -1, axis=1)   # l+1
    q_left  = np.roll(q,  1, axis=1)   # l-1

    phi_up    = np.roll(phi, -1, axis=0)
    phi_down  = np.roll(phi,  1, axis=0)
    phi_right = np.roll(phi, -1, axis=1)
    phi_left  = np.roll(phi,  1, axis=1)

    # 1 - (φ + φ')/2 — the (1-φ_bar) factor from eq (61)
    w_up    = 1.0 - 0.5 * (phi + phi_up)
    w_down  = 1.0 - 0.5 * (phi + phi_down)
    w_right = 1.0 - 0.5 * (phi + phi_right)
    w_left  = 1.0 - 0.5 * (phi + phi_left)

    # Edge gradients (q_neighbour - q_self) / Δ' on each of the 4 edges
    inv_dd = 1.0 / (lambda_init * lambda_init)
    flux_up    = w_up    * (q_up    - q)
    flux_down  = w_down  * (q_down  - q)
    flux_right = w_right * (q_right - q)
    flux_left  = w_left  * (q_left  - q)

    # Zero out wrap fluxes on axes we keep Neumann (no-flux at boundary)
    if "y" not in axes:
        flux_up[-1]    = 0.0    # top edge: no flux off the top
        flux_down[0]   = 0.0    # bottom edge
    if "x" not in axes:
        flux_right[:, -1] = 0.0
        flux_left[:,  0]  = 0.0

    # Sum of 4-neighbour differences (discrete divergence of the Fickian flux)
    lap = (flux_up + flux_down + flux_right + flux_left) * inv_dd

    # eq (57) prefactor: λ⊥ / (φ_0 · Δ²). Our Δ is the reference element edge
    # (=1 in paper units), so the remaining scale is λ⊥/φ_0 here.
    return (p.lambda_perp / p.phi_0) * lap


# ---------------------------------------------------------------------------
# Node velocity (for polymer-advection part of the flux)
# ---------------------------------------------------------------------------
def edge_velocity(v_nodes: np.ndarray, axis: int) -> np.ndarray:
    """Return the velocity averaged over the two end-nodes of each element edge.

    axis=0: horizontal edges (top/bottom), edge at node pairs (k,l)-(k,l+1)
    axis=1: vertical  edges (left/right), edge at node pairs (k,l)-(k+1,l)
    """
    if axis == 0:   # horizontal edge: average along x (l-direction)
        return 0.5 * (v_nodes[:, :-1] + v_nodes[:, 1:])       # (Ny+1, Nx, 2)
    else:           # vertical edge: average along y (k-direction)
        return 0.5 * (v_nodes[:-1, :] + v_nodes[1:, :])       # (Ny, Nx+1, 2)


# ---------------------------------------------------------------------------
# Main: RHS of the u-equation (eq 57) in full
# ---------------------------------------------------------------------------
def u_flux_rhs(u_elem: np.ndarray, phi: np.ndarray,
               nodes: np.ndarray, node_velocity: np.ndarray,
               p: Parameters, *, block_left_edge: bool = True) -> np.ndarray:
    """Return the non-reaction part of du/dt for every element — shape (Ny, Nx).

    Assembles both the polymer-advection flux (eq 60) and the self-diffusion
    flux (eq 61) and contracts with γ_s × J_s (eq 57).

    Parameters
    ----------
    u_elem        : (Ny, Nx) element mole fraction of activator
    phi           : (Ny, Nx) element polymer volume fraction
    nodes         : (Ny+1, Nx+1, 2) node coordinates
    node_velocity : (Ny+1, Nx+1, 2) dx/dt at every node
    block_left_edge : if True, no u-flux across the domain left edge (Case I wall)
    """
    Ny, Nx = u_elem.shape

    # --- element centre coordinates for self-diffusion flux (eq 61) ---
    x_m = 0.25 * (nodes[:-1, :-1] + nodes[:-1, 1:] + nodes[1:, 1:] + nodes[1:, :-1])

    # Safe 1/(1-φ) with small floor to avoid division by zero at collapsed gel
    one_minus_phi = np.clip(1.0 - phi, 1e-6, None)
    u_over = u_elem / one_minus_phi                 # u(m) / (1-φ(m)) on elements

    # --- assemble flux contribution from each of the four neighbour directions ---
    # For each edge, compute Σ_s e_3·[J_s × γ_s] as the net effect. Because
    # (e_3 × v_edge) × γ_edge + e_3·[J^(u) × γ_edge] can be written as an edge
    # sum, we process the four edges independently and sum the result into the
    # element.
    contrib = np.zeros((Ny, Nx), dtype=u_elem.dtype)

    def cross_z(a, b):
        """Scalar z-component of a × b for 2-D vectors (last axis = components)."""
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    # ---- south edge (s=1): between element (k,l) and neighbour (k-1,l) ----
    # Edge goes from node (k,l) to node (k,l+1) — direction +x in undeformed state.
    n_s = nodes[:-1, 1:] - nodes[:-1, :-1]          # (Ny, Nx, 2) edge vector
    v_edge = edge_velocity(node_velocity, axis=0)   # (Ny+1, Nx, 2) horizontal edges
    v_south = v_edge[:-1]                           # element's south edge: node row k

    # Advection J_s^(p) ≈ -u̅_edge/(1-φ̅_edge) · v̄_edge, where u̅_edge and φ̅_edge
    # are averaged from the two elements sharing the edge. For the south edge,
    # element (k-1,l) is the neighbour below.
    u_mean_s = np.empty_like(u_elem)
    phi_mean_s = np.empty_like(phi)
    u_mean_s[0] = u_elem[0]                         # domain boundary: single element
    u_mean_s[1:] = 0.5 * (u_elem[1:] + u_elem[:-1])
    phi_mean_s[0] = phi[0]
    phi_mean_s[1:] = 0.5 * (phi[1:] + phi[:-1])
    J_p_south = - (u_mean_s / np.clip(1.0 - phi_mean_s, 1e-6, None))[..., None] * v_south

    # Self-diffusion J_s^(u) — eq 61 — across south edge.
    # m' = (k-1, l); for k=0 there is no neighbour below.
    du = np.zeros_like(u_elem)
    du[1:] = u_over[:-1] - u_over[1:]
    dr = np.zeros((Ny, Nx, 2), dtype=u_elem.dtype)
    dr[1:] = x_m[:-1] - x_m[1:]                     # r(m') - r(m) (south neighbour)
    dr_sq = (dr ** 2).sum(-1) + 1e-12
    # (1 - (φ(m)+φ(m'))/2)
    oneminusmeanphi = np.zeros_like(phi)
    oneminusmeanphi[1:] = 1.0 - 0.5 * (phi[:-1] + phi[1:])
    factor_south = (oneminusmeanphi * du / dr_sq)[..., None] * dr  # J^(u)  (Ny,Nx,2)

    # This flux is across the south edge. Element (k=0,*) has no south neighbour,
    # and J^(u)_south=0 there.
    J_south = J_p_south - factor_south

    # contribution to element (k,l) from s=1 is e_3 · [J_s × γ_s]
    contrib += cross_z(J_south, n_s)

    # ---- north edge (s=3): between (k,l) and (k+1,l) ----
    v_north = v_edge[1:]                            # (Ny, Nx, 2)
    n_n = nodes[1:, :-1] - nodes[1:, 1:]            # opposite direction from south
    u_mean_n = np.empty_like(u_elem);   phi_mean_n = np.empty_like(phi)
    u_mean_n[-1] = u_elem[-1];          phi_mean_n[-1] = phi[-1]
    u_mean_n[:-1] = 0.5 * (u_elem[1:] + u_elem[:-1])
    phi_mean_n[:-1] = 0.5 * (phi[1:] + phi[:-1])
    J_p_north = - (u_mean_n / np.clip(1.0 - phi_mean_n, 1e-6, None))[..., None] * v_north
    du_n = np.zeros_like(u_elem);   dr_n = np.zeros_like(dr)
    du_n[:-1] = u_over[1:] - u_over[:-1]
    dr_n[:-1] = x_m[1:] - x_m[:-1]
    dr_sq_n = (dr_n ** 2).sum(-1) + 1e-12
    oneminusmeanphi_n = np.zeros_like(phi)
    oneminusmeanphi_n[:-1] = 1.0 - 0.5 * (phi[1:] + phi[:-1])
    factor_north = (oneminusmeanphi_n * du_n / dr_sq_n)[..., None] * dr_n
    J_north = J_p_north - factor_north
    contrib += cross_z(J_north, n_n)

    # ---- west edge (s=4): between (k,l) and (k,l-1) ----
    v_edge_w = edge_velocity(node_velocity, axis=1)  # (Ny, Nx+1, 2)
    v_west = v_edge_w[:, :-1]                        # (Ny, Nx, 2)
    n_w = nodes[:-1, :-1] - nodes[1:, :-1]           # left edge: downward
    u_mean_w = np.empty_like(u_elem);   phi_mean_w = np.empty_like(phi)
    u_mean_w[:, 0] = u_elem[:, 0];       phi_mean_w[:, 0] = phi[:, 0]
    u_mean_w[:, 1:] = 0.5 * (u_elem[:, 1:] + u_elem[:, :-1])
    phi_mean_w[:, 1:] = 0.5 * (phi[:, 1:] + phi[:, :-1])
    J_p_west = - (u_mean_w / np.clip(1.0 - phi_mean_w, 1e-6, None))[..., None] * v_west
    du_w = np.zeros_like(u_elem);   dr_w = np.zeros_like(dr)
    du_w[:, 1:] = u_over[:, :-1] - u_over[:, 1:]
    dr_w[:, 1:] = x_m[:, :-1] - x_m[:, 1:]
    dr_sq_w = (dr_w ** 2).sum(-1) + 1e-12
    oneminusmeanphi_w = np.zeros_like(phi)
    oneminusmeanphi_w[:, 1:] = 1.0 - 0.5 * (phi[:, :-1] + phi[:, 1:])
    factor_west = (oneminusmeanphi_w * du_w / dr_sq_w)[..., None] * dr_w
    J_west = J_p_west - factor_west
    # Case I wall: block u-flux across the left edge of the domain.
    if block_left_edge:
        J_west[:, 0] = 0.0
    contrib += cross_z(J_west, n_w)

    # ---- east edge (s=2): between (k,l) and (k,l+1) ----
    v_east = v_edge_w[:, 1:]
    n_e = nodes[1:, 1:] - nodes[:-1, 1:]             # right edge: upward
    u_mean_e = np.empty_like(u_elem);   phi_mean_e = np.empty_like(phi)
    u_mean_e[:, -1] = u_elem[:, -1];     phi_mean_e[:, -1] = phi[:, -1]
    u_mean_e[:, :-1] = 0.5 * (u_elem[:, 1:] + u_elem[:, :-1])
    phi_mean_e[:, :-1] = 0.5 * (phi[:, 1:] + phi[:, :-1])
    J_p_east = - (u_mean_e / np.clip(1.0 - phi_mean_e, 1e-6, None))[..., None] * v_east
    du_e = np.zeros_like(u_elem);   dr_e = np.zeros_like(dr)
    du_e[:, :-1] = u_over[:, 1:] - u_over[:, :-1]
    dr_e[:, :-1] = x_m[:, 1:] - x_m[:, :-1]
    dr_sq_e = (dr_e ** 2).sum(-1) + 1e-12
    oneminusmeanphi_e = np.zeros_like(phi)
    oneminusmeanphi_e[:, :-1] = 1.0 - 0.5 * (phi[:, 1:] + phi[:, :-1])
    factor_east = (oneminusmeanphi_e * du_e / dr_sq_e)[..., None] * dr_e
    J_east = J_p_east - factor_east
    contrib += cross_z(J_east, n_e)

    # Eq 57 prefactor: - λ⊥ φ_0^{-1} Δ^{-2}, and Δ=1.
    return -p.lambda_perp / p.phi_0 * contrib
