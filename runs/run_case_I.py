"""Main run — chemoresponsive gel (χ* = 0.105), 20×40 Case I.

Expected behaviour (Fig 6(B) of the paper):
  • Travelling waves of oxidation (v) propagate from the fixed wall
    towards the free edge.
  • Coupled waves of local swelling (low φ) follow the oxidation fronts.
  • Gel width oscillates rhythmically.

Duration: 100 T_0 is typically enough to see several wave passages at
these parameters (one period ≈ 100 T_0 according to Fig 6 caption).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from glsm.params import DEFAULT
from glsm import simulator as sim

CJK = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
                     size=10)


def main():
    p = DEFAULT                         # chi_star = 0.105 (responsive)
    state = sim.make_case_I(Ny=20, Nx=40, p=p,
                            lambda_init=p.lambda_perp,
                            u_init_scale=1e-3)
    print(f"Initial: φ={p.phi_0/(p.lambda_perp**3):.4f}  "
          f"u={state.u[0,0]:.2e}  v={state.v[0,0]:.1f}  "
          f"grid {state.u.shape}  χ* = {p.chi_star}")

    # 200 T_0 to see multiple wave passages
    t0 = time.time()
    snaps = sim.run(state, p, dt=0.005, n_steps=40000, snapshot_every=200,
                    reaction_substeps=10,
                    pin_left_wall=True, block_left_u_flux=True)
    print(f"Ran {len(snaps['t'])} snapshots in {time.time()-t0:.1f}s")
    print(f"Final u range: [{snaps['u'].min():.3e}, {snaps['u'].max():.3e}]")
    print(f"Final v range: [{snaps['v'].min():.3e}, {snaps['v'].max():.3e}]")
    print(f"Final φ range: [{snaps['phi'].min():.3e}, {snaps['phi'].max():.3e}]")

    out_dir = HERE.parent / "plots"
    out_dir.mkdir(exist_ok=True)

    # Centre horizontal cross section of v and φ over time (Fig 6 style)
    ny, nx = snaps["v"].shape[1:]
    mid = ny // 2
    v_cross = snaps["v"][:, mid, :]
    phi_cross = snaps["phi"][:, mid, :]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(phi_cross, aspect="auto", cmap="gray",
                          extent=[0, nx, snaps["t"][-1], 0])
    axes[0].set_title("φ 中线切片  x–t", fontproperties=CJK)
    axes[0].set_xlabel("element index (x)")
    axes[0].set_ylabel("time (T_0)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    im1 = axes[1].imshow(v_cross, aspect="auto", cmap="gray",
                          extent=[0, nx, snaps["t"][-1], 0])
    axes[1].set_title("v 中线切片  x–t", fontproperties=CJK)
    axes[1].set_xlabel("element index (x)")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    fig.suptitle(f"Case I χ*={p.chi_star}  (20×40 responsive)", fontproperties=CJK)
    fig.tight_layout()
    fig.savefig(out_dir / "case_I_chi_star_0p105_cross.png", dpi=130)
    plt.close(fig)

    # Multi-frame 2D density plots (Fig 7 style) — one period of oscillation
    N = len(snaps["t"])
    times_to_show = np.linspace(N // 3, N - 1, 6, dtype=int)
    fig, axes = plt.subplots(2, len(times_to_show), figsize=(2.6 * len(times_to_show), 5))
    vmin_v, vmax_v = snaps["v"][times_to_show].min(), snaps["v"][times_to_show].max()
    vmin_p, vmax_p = snaps["phi"][times_to_show].min(), snaps["phi"][times_to_show].max()
    for col, ti in enumerate(times_to_show):
        ax = axes[0, col]
        im = ax.imshow(snaps["v"][ti], cmap="gray", vmin=vmin_v, vmax=vmax_v)
        ax.set_title(f"t={snaps['t'][ti]:.1f}", fontproperties=CJK)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("v", fontproperties=CJK)
        ax = axes[1, col]
        im = ax.imshow(snaps["phi"][ti], cmap="gray", vmin=vmin_p, vmax=vmax_p)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("φ", fontproperties=CJK)
    fig.suptitle(f"Case I χ*={p.chi_star}  2D density", fontproperties=CJK)
    fig.tight_layout()
    fig.savefig(out_dir / "case_I_chi_star_0p105_2d.png", dpi=130)
    plt.close(fig)

    print(f"Saved {out_dir}/case_I_chi_star_0p105_*.png")


if __name__ == "__main__":
    main()
