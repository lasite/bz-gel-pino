"""Build the cardiac-style Full_Set evaluation dataset from an existing split.

Rebuilds the 200-frame raw trajectory from the already-saved train/test
sliding-window tensors (no re-simulation), then writes the three cardiac-
convention files to a sibling `Full_Set_...` folder:

    <ic>/
      Train_<N_T>_frames_<t_in>_inputsteps_<t_out>_outputsteps/
          2D_Oreg_train_<res>_<cm>.pt
          2D_Oreg_test_<res>_<cm>.pt
          dataset_info_<res>_<cm>.txt
      Full_Set_<N_T>_frames_<t_in>_inputsteps_<t_out>_outputsteps/    <-- new
          2D_Oreg_eval_<res>_<cm>.pt        # 191 full-trajectory windows
          2D_Oreg_eval_full_<res>_<cm>.pt   # raw [2, N_T, H, W]
          dataset_info_<res>_<cm>.txt

Mirrors cardiac_pino/.../data_constructor_openCARP.py lines 673-698.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from runs.generate_datasets import make_sliding_windows, write_dataset_info  # type: ignore


def _parse_dataset_info(p: Path) -> dict:
    """Parse a few fields we need from the existing train/test dataset_info."""
    out: dict = {}
    for line in p.read_text().splitlines():
        if line.startswith("Grid_resolution"):
            nums = re.findall(r"\d+\.?\d*", line)
            out["dx_cm"] = float(nums[0])
        elif line.startswith("Timestep resolution"):
            out["dt_ms"] = int(re.findall(r"\d+", line)[0])
        elif line.startswith("Input-Output pairs"):
            nums = re.findall(r"\d+", line)
            out["t_in"] = int(nums[0])
            out["t_out"] = int(nums[1])
        elif line.startswith("Simulation Info"):
            m = re.search(r"simtype\s*=\s*(\w+)", line)
            if m:
                out["sim_type"] = m.group(1)
            m = re.search(r"Conductivity Multipler\s*=\s*([\d.]+)", line)
            if m:
                out["cm"] = float(m.group(1))
        elif line.startswith("Loaded dataset shape"):
            nums = re.findall(r"\d+", line)
            out["n_timesteps"] = int(nums[1])
            out["res"] = int(nums[2])
    return out


def _reconstruct_segment(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Rebuild the contiguous time segment used to build x, y.

    x, y have shape [N, C, T_in or T_out, H, W] from an overlapping stride-1
    sliding window (generate_datasets.make_sliding_windows). For that layout:

        x[i, :, 0]          = frame i          for i in 0..N-1
        x[N-1, :, 1..T_in-1] = frames N..N+T_in-2
        y[N-1, :, 0..T_out-1] = frames N+T_in-1..N+T_in+T_out-2

    Total frames = N + T_in + T_out - 1 = (N-1) + T_in + T_out.
    """
    N, C, T_in, H, W = x.shape
    _, _, T_out, _, _ = y.shape
    # [N, C, H, W] -> [C, N, H, W]
    head = x[:, :, 0].transpose(0, 1)
    tail_x = x[-1, :, 1:]            # [C, T_in-1, H, W]
    tail_y = y[-1]                   # [C, T_out, H, W]
    seg = torch.cat([head, tail_x, tail_y], dim=1)
    assert seg.shape == (C, N + T_in + T_out - 1, H, W), seg.shape
    return seg


def _verify(seg: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
    """Check every (input, output) window in x, y matches slices of the segment."""
    N, C, T_in, H, W = x.shape
    T_out = y.shape[2]
    for i in range(N):
        ref_in = seg[:, i:i + T_in]
        ref_out = seg[:, i + T_in:i + T_in + T_out]
        if not torch.equal(ref_in, x[i]):
            raise RuntimeError(f"mismatch input at i={i}")
        if not torch.equal(ref_out, y[i]):
            raise RuntimeError(f"mismatch output at i={i}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-dir", required=True, type=Path,
        help="Path to existing Train_<N_T>_frames_...outputsteps/ folder",
    )
    ap.add_argument("--dataset-name", default="2D_Oreg")
    ap.add_argument("--res", type=int, default=101)
    ap.add_argument("--cm", type=float, default=1.0)
    ap.add_argument(
        "--domain-label",
        default="(uniform λ=1.1 square, Neumann BC)",
        help="Only used to stamp dataset_info; cosmetic.",
    )
    args = ap.parse_args()

    train_dir: Path = args.train_dir.resolve()
    cm_str = f"{args.cm}"
    train_pt = train_dir / f"{args.dataset_name}_train_{args.res}_{cm_str}.pt"
    test_pt = train_dir / f"{args.dataset_name}_test_{args.res}_{cm_str}.pt"
    info_in = train_dir / f"dataset_info_{args.res}_{cm_str}.txt"

    for p in (train_pt, test_pt, info_in):
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    meta = _parse_dataset_info(info_in)
    print(f"Parsed dataset_info: {meta}")

    tr = torch.load(train_pt.as_posix(), map_location="cpu", weights_only=False)
    te = torch.load(test_pt.as_posix(), map_location="cpu", weights_only=False)
    x_tr, y_tr = tr["x"], tr["y"]
    x_te, y_te = te["x"], te["y"]
    print(f"Train windows: x {tuple(x_tr.shape)}  y {tuple(y_tr.shape)}")
    print(f"Test  windows: x {tuple(x_te.shape)}  y {tuple(y_te.shape)}")

    t_in = meta["t_in"]
    t_out = meta["t_out"]
    assert x_tr.shape[2] == t_in and y_tr.shape[2] == t_out, "shape / meta mismatch"

    seg_train = _reconstruct_segment(x_tr, y_tr)
    seg_test = _reconstruct_segment(x_te, y_te)
    _verify(seg_train, x_tr, y_tr)
    _verify(seg_test, x_te, y_te)
    print(f"seg_train {tuple(seg_train.shape)}  seg_test {tuple(seg_test.shape)}")

    full = torch.cat([seg_train, seg_test], dim=1)  # [C, N_T, H, W]
    C, N_T, H, W = full.shape
    if N_T != meta["n_timesteps"]:
        print(
            f"[warn] reconstructed N_T={N_T} ≠ info n_timesteps={meta['n_timesteps']}"
        )
    print(f"Full trajectory: {tuple(full.shape)}  "
          f"u[{full[0].min():.4f},{full[0].max():.4f}]  "
          f"v[{full[1].min():.4f},{full[1].max():.4f}]")

    # sliding window over the whole trajectory — [1, C, N_T, H, W] batch of 1
    full_batched = full.unsqueeze(0)
    x_ev, y_ev = make_sliding_windows(full_batched, t_in, t_out)
    print(f"Eval windows: x {tuple(x_ev.shape)}  y {tuple(y_ev.shape)}")

    expected = N_T - (t_in + t_out) + 1
    assert x_ev.shape[0] == expected, (x_ev.shape, expected)

    # write out
    out_dir = (train_dir.parent
               / f"Full_Set_{N_T}_frames_{t_in}_inputsteps_{t_out}_outputsteps")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out_dir}")

    eval_pt = out_dir / f"{args.dataset_name}_eval_{args.res}_{cm_str}.pt"
    eval_full_pt = out_dir / f"{args.dataset_name}_eval_full_{args.res}_{cm_str}.pt"
    torch.save({"x": x_ev, "y": y_ev}, eval_pt)
    torch.save(full, eval_full_pt)
    print(f"Wrote {eval_pt}")
    print(f"Wrote {eval_full_pt}")

    u_lo, u_hi = float(full[0].min()), float(full[0].max())
    v_lo, v_hi = float(full[1].min()), float(full[1].max())
    info_out = out_dir / f"dataset_info_{args.res}_{cm_str}.txt"
    write_dataset_info(
        info_out,
        sim_type=meta.get("sim_type", "unknown"),
        cm=meta.get("cm", args.cm),
        res=args.res,
        grid_res_cm=meta["dx_cm"],
        dt_ms=meta["dt_ms"],
        n_timesteps=N_T,
        u_lo=u_lo, u_hi=u_hi, v_lo=v_lo, v_hi=v_hi,
        t_in=t_in, t_out=t_out,
        n_train_pairs=int(x_ev.shape[0]),  # reuse Training-shapes line for eval count
        n_test_pairs=int(x_ev.shape[0]),
        domain_label=args.domain_label,
    )
    # Add the two extra cardiac-style lines the Full_Set info file carries.
    with info_out.open("a") as f:
        f.write(
            f"Evaluation data shapes: {{torch.Size([{x_ev.shape[0]}, 2, {t_in}, "
            f"{args.res}, {args.res}])}} {{torch.Size([{y_ev.shape[0]}, 2, "
            f"{t_out}, {args.res}, {args.res}])}}\n"
        )
        f.write(f"Evaluation data shape: {{torch.Size([2, {N_T}, {args.res}, "
                f"{args.res}])}}\n")
    print(f"Wrote {info_out}")


if __name__ == "__main__":
    main()
