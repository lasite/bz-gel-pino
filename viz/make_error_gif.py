"""
Generate GT / Pred / Error GIF for a trained FNO model on the BZ-gel stable dataset.

Usage:
    python viz/make_error_gif.py \
        -r results/Results_stable_Data_Only_mse_5_frames_22_04_2026-19_20 \
        -d dataset/stable/Train_200_frames_5_inputsteps_5_outputsteps

Follows the same windowed-evaluation + overlap-averaging scheme as
cardiac_pino/.../Evaluation_P2P.py (reconstruct_rollout).
"""

import sys
from pathlib import Path

# vendored utils from cardiac_pino
_AP_UTILS_ROOT = Path("/media/b418/Wangyj/PINN/cardiac_pino/CardiacEP-PINOS")
if str(_AP_UTILS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AP_UTILS_ROOT))

import argparse
import os
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from neuralop.models import FNO
from neuralop.layers.embeddings import GridEmbeddingND
from neuralop.data.transforms.data_processors import DefaultDataProcessor


def reconstruct_rollout(y_pred_full, output_steps, stride=1):
    B, C, T, X, Y = y_pred_full.shape
    T_total = (B - 1) * stride + T
    rollout = torch.zeros((C, T_total, X, Y), device=y_pred_full.device)
    count = torch.zeros_like(rollout)
    for i in range(B):
        start = i * stride
        end = start + T
        rollout[:, start:end, :, :] += y_pred_full[i]
        count[:, start:end, :, :] += 1
    rollout /= torch.clamp(count, min=1.0)
    return rollout


def parse_dataset_info(info_path):
    dx = dy = None
    delta_t = None
    time_frames = None
    with open(info_path, "r") as f:
        for line in f:
            if line.startswith("Grid_resolution"):
                nums = re.findall(r"\d+\.?\d*", line)
                dx = dy = float(nums[0])
            elif line.startswith("Timestep resolution"):
                nums = re.findall(r"\d+", line)
                delta_t = int(nums[0])
            elif line.startswith("Input-Output pairs"):
                nums = re.findall(r"\d+", line)
                time_frames = int(nums[0])
    return dx, dy, delta_t, time_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-r", "--results-dir", required=True,
        help="Path to the training results folder with model_state_dict.pt",
    )
    ap.add_argument(
        "-d", "--data-dir",
        default="dataset/stable/Full_Set_200_frames_5_inputsteps_5_outputsteps",
        help="Path to dataset folder. Auto-detects Full_Set (2D_Oreg_eval_...)"
             " or the Train_... split (2D_Oreg_test_...).",
    )
    ap.add_argument("-res", "--resolution", type=int, default=101)
    ap.add_argument("-cm", "--conmul", type=float, default=1.0)
    ap.add_argument("-ms", "--mesh-size", type=float, default=10.0)
    ap.add_argument("-bs", "--batch-size", type=int, default=5)
    ap.add_argument("-fps", "--fps", type=int, default=8)
    ap.add_argument(
        "--state-dict",
        default=None,
        help="Override: explicit path to state_dict .pt. Defaults to "
             "<results-dir>/best_model_state_dict.pt if present, else "
             "<results-dir>/model_state_dict.pt",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- dataset info ----------
    data_dir = Path(args.data_dir).resolve()
    info_file = data_dir / f"dataset_info_{args.resolution}_{args.conmul}.txt"
    dx, dy, delta_t, time_frames = parse_dataset_info(info_file)
    time_boundary = float(delta_t) * (float(time_frames) - 1)
    print(f"dx={dx}, delta_t={delta_t} ms, frames/window={time_frames}, "
          f"time_boundary={time_boundary} ms")

    # ---------- evaluation windows ----------
    # Prefer the cardiac-style Full_Set eval (covers the whole 200-frame
    # trajectory, 191 windows). Fall back to the train/test split's test set
    # (~31 windows, last 20% of the trajectory only).
    eval_path = data_dir / f"2D_Oreg_eval_{args.resolution}_{args.conmul}.pt"
    test_path = data_dir / f"2D_Oreg_test_{args.resolution}_{args.conmul}.pt"
    if eval_path.exists():
        src = eval_path
        src_tag = "eval (full 200-frame trajectory)"
    elif test_path.exists():
        src = test_path
        src_tag = "test (last-20% split only)"
    else:
        raise SystemExit(f"Neither {eval_path.name} nor {test_path.name} in {data_dir}")
    data = torch.load(src.as_posix(), map_location="cpu", weights_only=False)
    x_test = data["x"].type(torch.float32)
    y_test = data["y"].type(torch.float32)
    print(f"Loaded {src_tag} from {src.name}: "
          f"x={tuple(x_test.shape)}, y={tuple(y_test.shape)}")

    # ---------- model ----------
    embedding = GridEmbeddingND(
        in_channels=2, dim=3,
        grid_boundaries=[
            [0.0, time_boundary],
            [0.0, float(args.mesh_size)],
            [0.0, float(args.mesh_size)],
        ],
    )
    model = FNO(
        n_modes=(8, 16, 16),
        in_channels=2,
        out_channels=2,
        hidden_channels=32,
        projection_channel_ratio=2,
        positional_embedding=embedding,
    ).to(device)

    results_dir = Path(args.results_dir).resolve()
    if args.state_dict is not None:
        ckpt_path = Path(args.state_dict)
    else:
        best = results_dir / "best_model_state_dict.pt"
        ckpt_path = best if best.exists() else results_dir / "model_state_dict.pt"
    print(f"Loading weights from: {ckpt_path}")
    sd = torch.load(ckpt_path.as_posix(), map_location=device, weights_only=False)
    model.load_state_dict(sd)
    model.eval()

    # The training used encode_input=False, encode_output=False, so
    # DefaultDataProcessor with None normalisers is effectively a no-op;
    # we can feed x_test directly.
    data_processor = DefaultDataProcessor(in_normalizer=None, out_normalizer=None).to(device)

    # ---------- forward pass over all test windows ----------
    all_y_true, all_y_pred = [], []
    with torch.no_grad():
        for i in range(0, x_test.shape[0], args.batch_size):
            x = x_test[i:i + args.batch_size].to(device)
            y = y_test[i:i + args.batch_size].to(device)
            batch = data_processor.preprocess({"x": x, "y": y}, batched=True)
            y_pred = model(batch["x"])
            all_y_true.append(batch["y"].cpu())
            all_y_pred.append(y_pred.cpu())
    y_true_full = torch.cat(all_y_true, dim=0)   # [B, C, T, H, W]
    y_pred_full = torch.cat(all_y_pred, dim=0)
    print(f"Stacked y_true={tuple(y_true_full.shape)}, y_pred={tuple(y_pred_full.shape)}")

    # ---------- reconstruct continuous rollout via overlap averaging ----------
    out_steps = y_pred_full.shape[2]
    y_true_roll = reconstruct_rollout(y_true_full, out_steps, stride=1)  # [C, T, H, W]
    y_pred_roll = reconstruct_rollout(y_pred_full, out_steps, stride=1)
    T_total = y_true_roll.shape[1]
    print(f"Rollout length T_total={T_total}")

    # convert to numpy
    gt = y_true_roll.numpy()                   # [C, T, H, W]
    pred = y_pred_roll.numpy()
    err = gt - pred

    # ---------- losses for reporting ----------
    mse_total = float(((gt - pred) ** 2).mean())
    mse_per_t = ((gt - pred) ** 2).mean(axis=(0, 2, 3))
    print(f"Overall MSE over rollout = {mse_total:.3e}")
    print(f"MSE(t) min/max = {mse_per_t.min():.3e} / {mse_per_t.max():.3e}")

    # ---------- build GIF: 2 rows (u, v) x 3 cols (GT, Pred, Err) ----------
    ch_names = ["u (gel volume fraction)", "v (oxidized reagent)"]
    C = gt.shape[0]

    # one color scale per channel shared across GT/Pred and across all frames
    vmins = [min(gt[c].min(), pred[c].min()) for c in range(C)]
    vmaxs = [max(gt[c].max(), pred[c].max()) for c in range(C)]
    err_abss = [float(np.abs(err[c]).max()) for c in range(C)]

    fig, axs = plt.subplots(C, 3, figsize=(12, 4 * C))
    if C == 1:
        axs = axs[None, :]
    ims = [[None] * 3 for _ in range(C)]

    for c in range(C):
        ims[c][0] = axs[c, 0].imshow(
            gt[c, 0], cmap="viridis", vmin=vmins[c], vmax=vmaxs[c])
        ims[c][1] = axs[c, 1].imshow(
            pred[c, 0], cmap="viridis", vmin=vmins[c], vmax=vmaxs[c])
        ims[c][2] = axs[c, 2].imshow(
            err[c, 0], cmap="bwr", vmin=-err_abss[c], vmax=err_abss[c])

        axs[c, 0].set_ylabel(ch_names[c], fontsize=11)
        fig.colorbar(ims[c][0], ax=axs[c, 0], fraction=0.046, pad=0.04)
        fig.colorbar(ims[c][1], ax=axs[c, 1], fraction=0.046, pad=0.04)
        fig.colorbar(ims[c][2], ax=axs[c, 2], fraction=0.046, pad=0.04)
        for ax in axs[c]:
            ax.set_xticks([])
            ax.set_yticks([])

    axs[0, 0].set_title("Ground Truth")
    axs[0, 1].set_title("Prediction")
    axs[0, 2].set_title("Error (GT − Pred)")

    suptitle = fig.suptitle("", fontsize=13)

    def update(frame):
        for c in range(C):
            ims[c][0].set_array(gt[c, frame])
            ims[c][1].set_array(pred[c, frame])
            ims[c][2].set_array(err[c, frame])
        suptitle.set_text(
            f"BZ-gel stable — FNO rollout (test windows)\n"
            f"frame {frame + 1}/{T_total}   "
            f"MSE(t)={mse_per_t[frame]:.3e}"
        )
        return [im for row in ims for im in row] + [suptitle]

    ani = animation.FuncAnimation(
        fig, update, frames=range(T_total), interval=1000 // args.fps, blit=False
    )

    out_gif = results_dir / "Evolution_GT_Pred_Error.gif"
    ani.save(out_gif.as_posix(), writer=PillowWriter(fps=args.fps))
    plt.close()
    print(f"Saved GIF -> {out_gif}")


if __name__ == "__main__":
    main()
