"""Plot training / testing loss curves for the stable BZ-gel runs.

Outputs two PNGs into plots/:
  1. losses_comparison.png — data-only vs PINO side-by-side on train loss +
     test MSE + PDE residual + 11× data:phys effective contribution.
  2. losses_pino_breakdown.png — PINO's per-metric decomposition
     (l2, ap_phys, ic, bcn) across all 150 epochs.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/media/b418/Wangyj/PINN/yashin2007_2d_glsm")
OUT = ROOT / "plots"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    "data-only":
        ROOT / "results"
             / "Results_stable_Data_Only_mse_5_frames_22_04_2026-19_52"
             / "training_log.json",
    "PINO":
        ROOT / "results"
             / "Results_stable_PINO_1.0_50_mse_5_frames22_04_2026-20_23"
             / "training_log.json",
}

RES_KEY = "(101, 1.0)"


def load(p: Path) -> list[dict]:
    with open(p) as f:
        return json.load(f)


def epoch(e: dict) -> int:
    return int(e["epoch"])


def series(logs, key):
    xs, ys = [], []
    for e in logs:
        if key in e and e[key] is not None:
            xs.append(epoch(e))
            ys.append(e[key])
    return xs, ys


def main():
    data = {tag: load(p) for tag, p in RUNS.items()}

    # --------- Plot 1: head-to-head comparison ---------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    colors = {"data-only": "#1f77b4", "PINO": "#d62728"}

    # (a) training loss (avg_loss per epoch — the training objective actually
    # being minimized; for PINO this is the weighted combined loss in main phase)
    ax = axes[0]
    for tag, logs in data.items():
        xs, ys = series(logs, "avg_loss")
        ax.plot(xs, ys, color=colors[tag], label=tag, lw=1.4)
    ax.axvline(100, color="gray", ls="--", lw=1, alpha=0.6)
    ax.text(100.5, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 1 else 0.9,
            " main phase →", fontsize=9, color="gray")
    ax.set_ylabel("training avg_loss / epoch")
    ax.set_yscale("log")
    ax.set_title("(a) Training loss")
    ax.grid(alpha=0.3)
    ax.legend()

    # (b) test MSE
    ax = axes[1]
    for tag, logs in data.items():
        xs, ys = series(logs, f"{RES_KEY}_mse")
        ax.plot(xs, ys, "o-", color=colors[tag], label=tag, ms=5, lw=1.4)
    ax.axvline(100, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_ylabel("test MSE")
    ax.set_yscale("log")
    ax.set_title(f"(b) Test MSE at {RES_KEY} — save_best metric")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    # (c) test PDE residual (ap_phys = OregLoss for PINO, APLoss for data-only
    # (which never saw it during training — just evaluation))
    ax = axes[2]
    for tag, logs in data.items():
        xs, ys = series(logs, f"{RES_KEY}_ap_phys")
        ax.plot(xs, ys, "s-", color=colors[tag], label=tag, ms=4, lw=1.4)
    ax.axvline(100, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_ylabel("PDE residual (OregLoss)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_title("(c) Test-set Oreg PDE residual")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    # (d) test L2 relative error
    ax = axes[3]
    for tag, logs in data.items():
        xs, ys = series(logs, f"{RES_KEY}_l2")
        ax.plot(xs, ys, "^-", color=colors[tag], label=tag, ms=4, lw=1.4)
    ax.axvline(100, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_ylabel("test Lp(d=2) loss")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_title("(d) Test Lp loss (d=2, p=2)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    fig.suptitle(
        "stable BZ-gel — data-only vs PINO training on same FNO(8,16,16)/32"
        "\n150 epochs: 100 init (L2) + 50 main"
        " (PINO: L2 + 0.01·Oreg + 0.1·IC + 0.1·BCN)",
        fontsize=12,
    )
    fig.tight_layout()
    out1 = OUT / "losses_comparison.png"
    fig.savefig(out1, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out1}")

    # --------- Plot 2: PINO per-metric breakdown ---------
    pino = data["PINO"]
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = [
        ("l2",       "L₂ (Lp d=2)",    "#1f77b4", "-"),
        ("mse",      "MSE",             "#d62728", "-"),
        ("ap_phys",  "Oreg residual",   "#2ca02c", "-"),
        ("ic",       "IC loss",         "#9467bd", "--"),
        ("bcn",      "BC Neumann",      "#ff7f0e", "--"),
        ("boundary", "boundary",        "#8c564b", ":"),
    ]
    for short, label, color, ls in metrics:
        xs, ys = series(pino, f"{RES_KEY}_{short}")
        ax.plot(xs, ys, color=color, ls=ls, lw=1.6, label=label, marker="o", ms=4)
    ax.axvline(100, color="gray", ls="--", lw=1, alpha=0.6)
    ax.text(101, 0.5, "main phase\n(PINO loss active)", fontsize=9, color="gray")
    ax.set_xlabel("epoch")
    ax.set_ylabel("value (log scale)")
    ax.set_yscale("log")
    ax.set_title(
        "PINO — all test-set metrics across 150 epochs"
        " (dashed = BC / IC, dotted = domain-boundary-only)",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    out2 = OUT / "losses_pino_breakdown.png"
    fig.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out2}")

    # --------- Tabular summary to stdout ---------
    print("\n=== best checkpoints (by test MSE) ===")
    for tag, logs in data.items():
        evs = [e for e in logs if f"{RES_KEY}_mse" in e]
        best = min(evs, key=lambda e: e[f"{RES_KEY}_mse"])
        print(f"{tag:10s}  ep{best['epoch']:>3}  "
              f"test_mse={best[f'{RES_KEY}_mse']:.3e}  "
              f"Oreg={best[f'{RES_KEY}_ap_phys']:.3e}  "
              f"L2={best[f'{RES_KEY}_l2']:.3f}")


if __name__ == "__main__":
    main()
