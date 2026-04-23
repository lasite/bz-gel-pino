"""Sanity run — nonresponsive gel (χ*=0), 20×40 Case I boundary conditions.

Expected behaviour (Fig 5(I) of the paper):
  • v oscillates uniformly in space (horizontal stripes in x–t plot)
  • gel deforms slowly towards equilibrium swelling (λ_eq ≈ 1.3) except at
    the wall where left column is pinned
  • no chemomechanical coupling because χ* = 0 ⇒ π_osm does not depend on v
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
    p = replace(DEFAULT, chi_star=0.0)     # nonresponsive gel
    state = sim.make_case_I(Ny=20, Nx=40, p=p,
                            lambda_init=p.lambda_perp,
                            u_init_scale=1e-3)
    print(f"Initial: φ={p.phi_0/(p.lambda_perp**3):.4f}  "
          f"u={state.u[0,0]:.2e}  v={state.v[0,0]:.1f}  "
          f"grid {state.u.shape}  "
          f"outer span [{state.nodes[0,0,0]:.2f}, {state.nodes[-1,-1,0]:.2f}] × "
          f"[{state.nodes[0,0,1]:.2f}, {state.nodes[-1,-1,1]:.2f}]")

    # dt=0.005 T_0 with 10 reaction sub-steps = 5e-4 inner step, stable for
    # both mechanics (mesh deformation) and the stiff Oregonator.
    t0 = time.time()
    snaps = sim.run(state, p, dt=0.005, n_steps=20000, snapshot_every=200,
                    reaction_substeps=10,
                    pin_left_wall=True, block_left_u_flux=True)
    print(f"Ran {len(snaps['t'])} snapshots in {time.time()-t0:.1f}s")
    print(f"Final u range: [{snaps['u'].min():.3e}, {snaps['u'].max():.3e}]")
    print(f"Final v range: [{snaps['v'].min():.3e}, {snaps['v'].max():.3e}]")
    print(f"Final φ range: [{snaps['phi'].min():.3e}, {snaps['phi'].max():.3e}]")

    out_dir = HERE.parent / "plots"
    out_dir.mkdir(exist_ok=True)

    # Centre horizontal cross section of v over time (Fig 5 style)
    ny, nx = snaps["v"].shape[1:]
    mid = ny // 2
    v_cross = snaps["v"][:, mid, :]       # (n_t, Nx)
    phi_cross = snaps["phi"][:, mid, :]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(phi_cross, aspect="auto", cmap="gray",
                          extent=[0, nx, snaps["t"][-1], 0])
    axes[0].set_title("φ 中线切片  x–t", fontproperties=CJK)
    axes[0].set_xlabel("element index (x)")
    axes[0].set_ylabel("time (T₀)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    im1 = axes[1].imshow(v_cross, aspect="auto", cmap="gray",
                          extent=[0, nx, snaps["t"][-1], 0])
    axes[1].set_title("v 中线切片  x–t", fontproperties=CJK)
    axes[1].set_xlabel("element index (x)")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    fig.suptitle("Case I  χ*=0  (nonresponsive, 20×40)", fontproperties=CJK)
    fig.tight_layout()
    fig.savefig(out_dir / "case_I_chi_star_0_cross.png", dpi=130)
    plt.close(fig)

    # Final shape
    fig, ax = plt.subplots(figsize=(6, 4))
    node_final = snaps["nodes"][-1]
    for row in node_final:
        ax.plot(row[:, 0], row[:, 1], "-", lw=0.3, color="#555")
    for col in range(node_final.shape[1]):
        ax.plot(node_final[:, col, 0], node_final[:, col, 1], "-", lw=0.3, color="#555")
    ax.set_aspect("equal")
    ax.set_title(f"最终网格  t={snaps['t'][-1]:.1f} T₀", fontproperties=CJK)
    fig.tight_layout()
    fig.savefig(out_dir / "case_I_chi_star_0_mesh.png", dpi=130)
    plt.close(fig)

    print(f"Saved {out_dir}/case_I_chi_star_0_*.png")


if __name__ == "__main__":
    main()
