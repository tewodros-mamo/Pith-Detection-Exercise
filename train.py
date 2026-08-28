"""Train a compact CNN to regress 15 pith x-positions (mm) from one board-face image.

Design choices
--------------
- Input 224x224 (downscaled from 512): ring curvature/spacing cues survive; 5x faster on CPU.
- Compact ~1M-param CNN trained from scratch: synthetic single-domain data, 4000 samples,
  CPU-only machine. ImageNet pretraining buys little here and costs 4x per epoch.
- Two heads (ablation):
    * 'band'  : pool feature map over WIDTH only -> 15 vertical bands -> 1x1 conv head.
                Matches the output structure: target i depends on rings near z-band i.
    * 'global': standard global average pool -> FC(15). The generic baseline.
- Targets normalized t = (x_mm - 72.5) / 145 (zero-centered); SmoothL1 loss.
  No clamping anywhere: predictions may leave [0, 145] mm as required.
- Augmentation: horizontal flip (t -> -t) and vertical flip (reverse the 15 targets),
  both exact symmetries of the physical problem.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent
CENTER, SCALE = 72.5, 145.0


def block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, 2, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
    )


class PithNet(nn.Module):
    def __init__(self, head: str = "band"):
        super().__init__()
        self.head_type = head
        self.features = nn.Sequential(
            block(3, 24), block(24, 48), block(48, 96), block(96, 128), block(128, 192),
        )  # 224 -> 7x7x192
        if head == "band":
            self.pool = nn.AdaptiveAvgPool2d((15, 1))  # keep z-resolution, pool width away
            self.head = nn.Sequential(nn.Conv1d(192, 64, 1), nn.ReLU(inplace=True), nn.Conv1d(64, 1, 1))
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(192, 15)

    def forward(self, x):
        f = self.features(x)
        if self.head_type == "band":
            f = self.pool(f).squeeze(-1)          # B,192,15  (band 0 = image top = z=145)
            out = self.head(f).squeeze(1)         # B,15
            return out.flip(-1)                   # reorder bottom-to-top to match labels
        return self.head(self.pool(f).flatten(1))


def load_split(split):
    imgs = np.load(ROOT / f"cache_{split}_images.npy")
    labels = np.load(ROOT / f"cache_{split}_labels.npy")
    return torch.from_numpy(imgs), torch.from_numpy(labels)


def normalize(batch_uint8):
    x = batch_uint8.permute(0, 3, 1, 2).float().div_(255.0)
    return x.sub_(0.5).div_(0.25)


def augment(xb, tb):
    """xb: B,H,W,3 uint8; tb: B,15 normalized targets."""
    bh = torch.rand(len(xb)) < 0.5
    bv = torch.rand(len(xb)) < 0.5
    xb = xb.clone()
    tb = tb.clone()
    xb[bh] = xb[bh].flip(2)   # mirror width  -> x = 145 - x  -> t = -t
    tb[bh] = -tb[bh]
    xb[bv] = xb[bv].flip(1)   # mirror height -> z reversed  -> reverse target order
    tb[bv] = tb[bv].flip(-1)
    return xb, tb


def evaluate(model, imgs, labels, batch=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            x = normalize(imgs[i:i + batch])
            preds.append(model(x) * SCALE + CENTER)
    preds = torch.cat(preds)
    mae = (preds - labels).abs().mean().item()
    return mae, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["band", "global"], default="band")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    run = args.run_name or f"{args.head}_e{args.epochs}"

    torch.set_num_threads(8)
    torch.manual_seed(0)

    imgs, labels = load_split("train")
    t_norm = (labels - CENTER) / SCALE
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(imgs), generator=g)
    val_idx, tr_idx = perm[:400], perm[400:]
    tr_imgs, tr_t = imgs[tr_idx], t_norm[tr_idx]
    val_imgs, val_labels = imgs[val_idx], labels[val_idx]

    model = PithNet(args.head)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(tr_idx) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    print(f"run={run} head={args.head} params={n_par/1e6:.2f}M train={len(tr_idx)} val={len(val_idx)}", flush=True)
    log_path = ROOT / "outputs" / f"log_{run}.csv"
    best = float("inf")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_mae_mm", "seconds"])
        for ep in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            order = torch.randperm(len(tr_idx))
            tot, nb = 0.0, 0
            for i in range(0, len(order) - args.batch + 1, args.batch):
                idx = order[i:i + args.batch]
                xb, tb = augment(tr_imgs[idx], tr_t[idx])
                loss = F.smooth_l1_loss(model(normalize(xb)), tb, beta=0.03)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item(); nb += 1
            mae, _ = evaluate(model, val_imgs, val_labels)
            dt = time.time() - t0
            w.writerow([ep, f"{tot/nb:.5f}", f"{mae:.3f}", f"{dt:.0f}"])
            f.flush()
            flag = ""
            if mae < best:
                best = mae
                torch.save({"model": model.state_dict(), "head": args.head}, ROOT / "outputs" / f"model_{run}.pt")
                flag = " *best*"
            print(f"ep {ep:2d}/{args.epochs} loss {tot/nb:.4f} val MAE {mae:.2f} mm ({dt:.0f}s){flag}", flush=True)
    print(f"done. best val MAE {best:.2f} mm", flush=True)


if __name__ == "__main__":
    main()
