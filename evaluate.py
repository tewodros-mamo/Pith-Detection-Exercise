"""Evaluate a trained model on the held-out test set.

Reports MAE (mm) overall, per z-position, and per-board; saves prediction CSV and
visual-check overlay figures (best / typical / worst boards, plus pith-outside cases).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from train import PithNet, load_split, evaluate, CENTER, SCALE

ROOT = Path(__file__).parent
Z = np.array([0.0, 10.357143, 20.714286, 31.071429, 41.428571, 51.785714, 62.142857,
              72.5, 82.857143, 93.214286, 103.571429, 113.928571, 124.285714,
              134.642857, 145.0])


def overlay(ax, name, true_x, pred_x, title):
    img = Image.open(ROOT / "test" / "images" / name)
    ax.imshow(img, extent=[0, 145, 0, 145])
    ax.plot(true_x, Z, "-o", color="red", ms=3, lw=1.5, label="true")
    ax.plot(pred_x, Z, "--s", color="deepskyblue", ms=3, lw=1.5, label="predicted")
    lo = min(-5, true_x.min() - 5, pred_x.min() - 5)
    hi = max(150, true_x.max() + 5, pred_x.max() + 5)
    ax.set_xlim(lo, hi)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()

    torch.set_num_threads(8)
    ckpt = torch.load(ROOT / "outputs" / f"model_{args.run_name}.pt", weights_only=True)
    model = PithNet(ckpt["head"])
    model.load_state_dict(ckpt["model"])

    imgs, labels = load_split("test")
    names = pd.read_csv(ROOT / "cache_test_names.csv")["image"].tolist()
    mae, preds = evaluate(model, imgs, labels)
    preds = preds.numpy()
    truth = labels.numpy()
    err = np.abs(preds - truth)

    print(f"TEST MAE: {mae:.3f} mm")
    print(f"median abs err: {np.median(err):.3f} mm   p90: {np.percentile(err,90):.3f}   p99: {np.percentile(err,99):.3f}   max: {err.max():.2f}")
    print("per-position MAE (x_00..x_14):", np.round(err.mean(0), 2))
    board_mae = err.mean(1)
    print(f"per-board MAE: median {np.median(board_mae):.2f}, worst {board_mae.max():.2f}")
    inside = (truth.min(1) >= 0) & (truth.max(1) <= 145)
    print(f"MAE pith fully inside ({inside.sum()} boards): {err[inside].mean():.3f} mm")
    print(f"MAE pith partly outside ({(~inside).sum()} boards): {err[~inside].mean():.3f} mm")

    out = pd.DataFrame(preds, columns=[f"x_{i:02d}_mm" for i in range(15)])
    out.insert(0, "image", names)
    out.to_csv(ROOT / "outputs" / f"predictions_{args.run_name}.csv", index=False)

    # ---- figures ----
    order = np.argsort(board_mae)
    med = order[len(order) // 2]
    picks = [(order[0], "best"), (med, "median"), (order[-1], "worst"), (order[-2], "2nd worst")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, (i, tag) in zip(axes, picks):
        overlay(ax, names[i], truth[i], preds[i], f"{tag}: {names[i]}  MAE {board_mae[i]:.2f} mm")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ROOT / "outputs" / f"visual_check_extremes_{args.run_name}.png", dpi=110)

    outs = np.where(~inside)[0]
    sel = outs[np.argsort(board_mae[outs])][:4] if len(outs) >= 4 else outs
    fig, axes = plt.subplots(1, len(sel), figsize=(4 * len(sel), 4.4))
    for ax, i in zip(np.atleast_1d(axes), sel):
        overlay(ax, names[i], truth[i], preds[i], f"pith outside: {names[i]}  MAE {board_mae[i]:.2f} mm")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ROOT / "outputs" / f"visual_check_outside_{args.run_name}.png", dpi=110)

    rng = np.random.default_rng(7)
    sel = rng.choice(len(names), 4, replace=False)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, i in zip(axes, sel):
        overlay(ax, names[i], truth[i], preds[i], f"random: {names[i]}  MAE {board_mae[i]:.2f} mm")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ROOT / "outputs" / f"visual_check_random_{args.run_name}.png", dpi=110)

    # error analysis figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(board_mae, bins=40, color="#4878cf", edgecolor="white")
    axes[0].set_xlabel("per-board MAE (mm)"); axes[0].set_ylabel("boards"); axes[0].set_title("Per-board MAE distribution")
    axes[1].plot(Z, err.mean(0), "-o", color="#4878cf")
    axes[1].set_xlabel("z position (mm)"); axes[1].set_ylabel("MAE (mm)"); axes[1].set_title("MAE per z position"); axes[1].set_ylim(bottom=0)
    axes[2].scatter(truth.mean(1), board_mae, s=6, alpha=0.4, color="#4878cf")
    axes[2].axvline(0, color="gray", ls=":"); axes[2].axvline(145, color="gray", ls=":")
    axes[2].set_xlabel("true mean pith x (mm)"); axes[2].set_ylabel("per-board MAE (mm)"); axes[2].set_title("Error vs pith position")
    plt.tight_layout()
    plt.savefig(ROOT / "outputs" / f"error_analysis_{args.run_name}.png", dpi=110)
    print("figures saved to outputs/")


if __name__ == "__main__":
    main()
