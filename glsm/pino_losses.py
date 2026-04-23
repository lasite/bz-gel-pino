"""Physics-informed losses for the rigid-mesh 2D Yashin-Balazs 2007 gLSM.

In the rigid-mesh, uniform-φ regime (matches the `generate_datasets.py`
settings the dataset was built with), the coupled polymer-Oregonator system
reduces to pure reaction-diffusion in dimensionless paper units (T_0, L_0):

    ∂u/∂t = F(u, v, φ_init) + D_eff · ∇²u
    ∂v/∂t = ε · G(u, v, φ_init)               (no v-diffusion)

    D_eff     = (λ⊥ / φ_0) · d_scale
    φ_init    = φ_0 / (λ⊥ · λ_init²)
    Δ' = dx  = λ_init                (dimensionless spatial grid step)
    Δt_frame = (dt_ms / 1000)        (dimensionless frame-to-frame time step)

F, G are the Tyson-Fife polymer-modified Oregonator right-hand sides from
`glsm.reaction` (eqs 6, 7 of the paper).

Interface mirrors cardiac_pino's `APLoss` so it drops into PINO_Train.py's
`WeightedSumLoss([l2, resloss, ic, bcn], ...)` combination unchanged.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Re-use cardiac_pino's finite-difference utility for Neumann / periodic
# Laplacian on [..., H, W] tensors.
_AP_UTILS_ROOT = Path("/media/b418/Wangyj/PINN/cardiac_pino/CardiacEP-PINOS")
if str(_AP_UTILS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AP_UTILS_ROOT))

import torch
import torch.nn.functional as F

from AP_neuralop_utils.losses.differentiation import FiniteDiff  # noqa: E402

from .params import DEFAULT, Parameters
from . import reaction as rxn


class OregLoss:
    """Finite-difference PDE-residual loss for the rigid-mesh Yashin gLSM.

    Same __call__ signature as cardiac_pino APLoss so it plugs into
    `WeightedSumLoss`.
    """

    def __init__(self,
                 *,
                 p: Parameters = DEFAULT,
                 f: float = 0.9,
                 epsilon: float = 0.2,
                 q: float | None = None,
                 d_scale: float = 1.0,
                 lambda_init: float = 1.1,
                 dt_dimless: float = 0.34,
                 periodic: bool = False,
                 loss_fn=F.mse_loss,
                 u_loss_weighting: float = 1.0,
                 v_loss_weighting: float = 1.0,
                 reduction: str = "mean"):
        # Override reaction params on top of DEFAULT — these are the actual
        # numbers the stable-spiral dataset was generated with:
        overrides: dict = {"f": f, "epsilon": epsilon}
        if q is not None:
            overrides["q"] = q
        from dataclasses import replace
        self.p = replace(p, **overrides)

        self.phi_init = self.p.phi_0 / (self.p.lambda_perp * lambda_init ** 2)
        self.D_eff = (self.p.lambda_perp / self.p.phi_0) * d_scale
        self.lambda_init = lambda_init
        self.dt_dimless = float(dt_dimless)
        self.periodic = periodic
        self.loss_fn = loss_fn
        self.u_w = u_loss_weighting
        self.v_w = v_loss_weighting
        self.reduction = reduction

        # Build one FiniteDiff; dx=dy=lambda_init in dimensionless paper units.
        self._fd2d = FiniteDiff(
            dim=2,
            h=(lambda_init, lambda_init),
            periodic_in_x=periodic,
            periodic_in_y=periodic,
        )

    # ------------------------------------------------------------------
    def _laplacian_per_frame(self, u_bt: torch.Tensor) -> torch.Tensor:
        """u_bt: [B, T, H, W] -> [B, T, H, W], 2D Laplacian per (b, t) slice."""
        B, T, H, W = u_bt.shape
        return self._fd2d.laplacian(u_bt.reshape(B * T, H, W)).view(B, T, H, W)

    def _phi_tensor(self, template: torch.Tensor) -> torch.Tensor:
        return torch.full_like(template, self.phi_init)

    # ------------------------------------------------------------------
    def residual_central(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Central-finite-difference residual loss over interior time.

        y_pred : [B, 2, T, H, W]  (channel 0 = u, channel 1 = v)
        Matches cardiac APLoss.residual_finite_difference_flexible's shape
        — uses central diff dt at t+1 …t-1, evaluates the RHS at t.
        Requires T >= 3.
        """
        B, C, T, H, W = y_pred.shape
        assert C == 2, f"expected 2 channels (u, v), got {C}"
        if T < 3:
            raise ValueError(f"Need T>=3 for central diff; got T={T}")

        u = y_pred[:, 0]           # [B, T, H, W]
        v = y_pred[:, 1]

        # Time derivatives via central diff at interior time steps [1 … T-2]
        u_t = (u[:, 2:] - u[:, :-2]) / (2.0 * self.dt_dimless)   # [B, T-2, H, W]
        v_t = (v[:, 2:] - v[:, :-2]) / (2.0 * self.dt_dimless)

        # RHS evaluated at interior time points [1 … T-2].
        # The simulator clamps u, v to [0, 1.5] every sub-step — outside this
        # range the Oregonator ratio (u-q)/(u+q) can spike by ~1/q near
        # u ≈ -q, blowing the residual to ~1e17 on cold-start FNO outputs.
        # We clamp here so the residual stays on a physically meaningful
        # manifold (gradient on out-of-range outputs comes from the L2 loss,
        # not this term).
        u_raw = u[:, 1:-1]
        v_raw = v[:, 1:-1]
        u_c = u_raw.clamp(0.0, 1.5)
        v_c = v_raw.clamp(0.0, 1.5)
        phi = self._phi_tensor(u_c)
        F_c = rxn.F_torch(u_c, v_c, phi, self.p)                 # [B, T-2, H, W]
        G_c = rxn.G_torch(u_c, v_c, phi, self.p)
        # Laplacian uses the unclamped u so spatial gradient flows through.
        lap_u = self._laplacian_per_frame(u_raw)                 # [B, T-2, H, W]

        rhs_u = F_c + self.D_eff * lap_u
        rhs_v = self.p.epsilon * G_c

        # When not periodic, FiniteDiff uses one-sided stencils at the
        # boundary — those are less accurate, so exclude the outermost ring
        # from the loss, matching cardiac's practice of dropping boundaries.
        if not self.periodic:
            sl = (slice(None), slice(None), slice(1, -1), slice(1, -1))
            u_t = u_t[sl]; v_t = v_t[sl]
            rhs_u = rhs_u[sl]; rhs_v = rhs_v[sl]

        loss_u = self.loss_fn(u_t, rhs_u, reduction=self.reduction)
        loss_v = self.loss_fn(v_t, rhs_v, reduction=self.reduction)
        return self.u_w * loss_u + self.v_w * loss_v

    # ------------------------------------------------------------------
    def __call__(self, y_pred=None, **kwargs):
        # Accepts (y_pred, y, x, ...) for compatibility with WeightedSumLoss;
        # only y_pred is used here (residual is purely a PDE check on the
        # predicted trajectory).
        if y_pred is None:
            raise TypeError("OregLoss requires y_pred kwarg")
        return self.residual_central(y_pred)

    def __str__(self) -> str:
        return (f"OregLoss(f={self.p.f:.3f}, eps={self.p.epsilon:.3f}, "
                f"q={self.p.q:.3e}, phi_init={self.phi_init:.4f}, "
                f"D_eff={self.D_eff:.3f}, Δ'={self.lambda_init}, "
                f"Δt={self.dt_dimless}, periodic={self.periodic})")
