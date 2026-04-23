"""Visualize a deformable BZ-gel dataset as a two-panel GIF.

Left panel  : v (catalyst) on the deformed element mesh — colour encodes
              chemistry, the painted region's shape encodes the mechanics.
Right panel : |δ| (displacement magnitude) on the same deformed mesh —
              makes it explicit where the gel is stretching / compressing.

Both panels share a dashed grey outline of the undeformed reference
rectangle so the viewer has a fixed visual anchor for "how much has the
gel moved". The outside-of-gel background is dark so the gel's footprint
reads as a solid shape without needing an explicit boundary line.

Usage:
    python viz/plot_deform_gel.py -d dataset/spiral_deform/Train_201_frames_5_inputsteps_5_outputsteps
"""
from __future__ import annotations
import argparse
from pathlib import Path

import re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

LAMBDA_IN = 1.1


def detect_dx_from_info(data_dir: Path, res: int, cm: float) -> float:
    """Read p.dx from the dataset_info_<res>_<cm>.txt line written by
    generate_datasets.py. Falls back to 1.0 (paper convention) if missing."""
    info = data_dir / f"dataset_info_{res}_{cm}.txt"
    if not info.exists():
        return 1.0
    m = re.search(r"Undeformed element edge Δ \(dx\):\s*([0-9eE\.+\-]+)",
                  info.read_text())
    return float(m.group(1)) if m else 1.0


def reconstruct_trajectory(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """[N, C, T_in, H, W], [N, C, T_out, H, W] -> [C, N+T_in+T_out-1, H, W]."""
    N, C, T_in, H, W = x.shape
    T_out = y.shape[2]
    head = x[:, :, 0].permute(1, 0, 2, 3)
    tail_x = x[-1, :, 1:]
    tail_y = y[-1]
    return torch.cat([head, tail_x, tail_y], dim=1)


def _elem_to_node(arr_elem: np.ndarray) -> np.ndarray:
    """Average element-centred quantity onto the (Ny+1, Nx+1) node grid.

    arr_elem shape (..., Ny, Nx). Edge-padded so that boundary nodes take the
    nearest-element value. Returns shape (..., Ny+1, Nx+1).
    """
    padded = np.pad(arr_elem, tuple([(0, 0)] * (arr_elem.ndim - 2) + [(1, 1), (1, 1)]),
                    mode="edge")
    return 0.25 * (padded[..., :-1, :-1] + padded[..., :-1, 1:]
                 + padded[...,  1:, :-1] + padded[...,  1:, 1:])


def deltas_to_node_positions(delta_elem: np.ndarray,
                              spacing: float) -> np.ndarray:
    """Convert element-centered δ to deformed node positions.

    delta_elem : (2, T, Ny, Nx)
    spacing    : reference node spacing (λ_init · dx)
    returns    : (T, Ny+1, Nx+1, 2)   — (x, y) per node per frame

    Each node is the average of the up to 4 surrounding elements' δ plus the
    reference node position (l·spacing, k·spacing).
    """
    _, T, Ny, Nx = delta_elem.shape
    avg = _elem_to_node(delta_elem)   # (2, T, Ny+1, Nx+1)
    ks = np.arange(Ny + 1); ls = np.arange(Nx + 1)
    ref_y, ref_x = np.meshgrid(ks * spacing, ls * spacing, indexing="ij")
    nodes = np.empty((T, Ny + 1, Nx + 1, 2), dtype=np.float32)
    nodes[..., 0] = ref_x[None] + avg[0]
    nodes[..., 1] = ref_y[None] + avg[1]
    return nodes


def reference_outline(Ny: int, Nx: int, spacing: float
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Corner-to-corner closed polyline of the undeformed gel rectangle."""
    x = np.array([0, Nx, Nx, 0, 0], dtype=float) * spacing
    y = np.array([0, 0, Ny, Ny, 0], dtype=float) * spacing
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data-dir", required=True, type=Path)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--res", type=int, default=101)
    ap.add_argument("--cm", type=float, default=1.0)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--channel", default="v", choices=["u", "v"],
                    help="which chemistry field to colour on the left panel")
    ap.add_argument("--dx", type=float, default=None,
                    help="undeformed element edge Δ. Auto-read from "
                         "dataset_info if not given; falls back to 1.0.")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    pt_path = data_dir / f"2D_Oreg_{args.split}_{args.res}_{args.cm}.pt"
    d = torch.load(pt_path.as_posix(), map_location="cpu", weights_only=False)
    x, y = d["x"], d["y"]
    assert x.shape[1] == 4, f"expected 4 channels (u,v,δx,δy), got {x.shape[1]}"
    traj = reconstruct_trajectory(x, y).numpy()
    C, T, Ny, Nx = traj.shape
    print(f"Loaded {pt_path.name}, reconstructed {traj.shape}")

    dx_elem = args.dx if args.dx is not None \
              else detect_dx_from_info(data_dir, args.res, args.cm)
    spacing = LAMBDA_IN * dx_elem
    print(f"reference node spacing = λ_init · Δ = {LAMBDA_IN} · {dx_elem} "
          f"= {spacing}")

    ch_idx = {"u": 0, "v": 1}[args.channel]
    fld = traj[ch_idx]
    delta = traj[2:4]
    nodes = deltas_to_node_positions(delta, spacing)

    # |δ| at element centres
    dmag = np.sqrt(delta[0] ** 2 + delta[1] ** 2)

    # Interpolate element-centred fields onto the node grid so that Gouraud
    # shading can be used (pcolormesh needs X, Y, C all same shape).
    fld_node = _elem_to_node(fld)         # (T, Ny+1, Nx+1)
    dmag_node = _elem_to_node(dmag)

    v_vmin, v_vmax = 0.0, float(fld_node.max())
    d_vmin, d_vmax = 0.0, float(dmag_node.max())

    pad = 0.02 * max(Nx, Ny) * spacing
    x_lo = float(nodes[..., 0].min()) - pad
    x_hi = float(nodes[..., 0].max()) + pad
    y_lo = float(nodes[..., 1].min()) - pad
    y_hi = float(nodes[..., 1].max()) + pad
    ref_area = (Nx * spacing) * (Ny * spacing)
    ref_x, ref_y = reference_outline(Ny, Nx, spacing)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 6.6))
    fig.patch.set_facecolor("white")
    for ax in (ax_l, ax_r):
        ax.set_facecolor("#101015")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.05,
                        wspace=0.15)

    qm_l = ax_l.pcolormesh(nodes[0, ..., 0], nodes[0, ..., 1], fld_node[0],
                           shading="gouraud", cmap="viridis",
                           vmin=v_vmin, vmax=v_vmax)
    qm_r = ax_r.pcolormesh(nodes[0, ..., 0], nodes[0, ..., 1], dmag_node[0],
                           shading="gouraud", cmap="magma",
                           vmin=d_vmin, vmax=d_vmax)
    fig.colorbar(qm_l, ax=ax_l, fraction=0.046, pad=0.03,
                 label=f"{args.channel}  "
                       f"({'activator' if args.channel == 'u' else 'catalyst'})")
    fig.colorbar(qm_r, ax=ax_r, fraction=0.046, pad=0.03,
                 label="|δ|  (displacement magnitude)")
    title = fig.suptitle("", fontsize=12)

    frames = list(range(0, T, args.stride))
    print(f"Rendering {len(frames)} frames (stride={args.stride})…")

    def draw(ax, X, Y, Z, cmap, vmin, vmax, label_left: bool):
        ax.clear()
        ax.set_facecolor("#101015")
        ax.pcolormesh(X, Y, Z, shading="gouraud", cmap=cmap,
                      vmin=vmin, vmax=vmax)
        ax.plot(ref_x, ref_y, linestyle="--", linewidth=0.8,
                color="#bbbbbb", alpha=0.7)
        ax.set_aspect("equal")
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("v on deformed gel" if label_left
                     else "|δ| on deformed gel",
                     fontsize=11, color="#222")

    def update(f: int):
        draw(ax_l, nodes[f, ..., 0], nodes[f, ..., 1], fld_node[f],
             "viridis", v_vmin, v_vmax, True)
        draw(ax_r, nodes[f, ..., 0], nodes[f, ..., 1], dmag_node[f],
             "magma", d_vmin, d_vmax, False)
        area = ((nodes[f, ..., 0].max() - nodes[f, ..., 0].min())
              * (nodes[f, ..., 1].max() - nodes[f, ..., 1].min()))
        title.set_text(
            f"deformable BZ-gel spiral wave   "
            f"frame {f + 1}/{T}   "
            f"bbox {100 * area / ref_area:+.1f}% vs ref   "
            f"|δ|_max={dmag[f].max():.2f}"
        )
        return []

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    out = args.out if args.out is not None \
          else data_dir / f"deform_{args.split}_{args.channel}.gif"
    ani.save(out.as_posix(), writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
