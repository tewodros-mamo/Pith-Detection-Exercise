"""Decode all board-face PNGs once, resize to 224x224, cache as uint8 .npy arrays.

Avoids re-decoding 512x512 PNGs every epoch, which would bottleneck CPU training.
"""
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import time

SIZE = 224
ROOT = Path(__file__).parent


def build(split: str):
    labels = pd.read_csv(ROOT / split / "labels.csv")
    n = len(labels)
    imgs = np.empty((n, SIZE, SIZE, 3), dtype=np.uint8)
    t0 = time.time()
    for i, name in enumerate(labels["image"]):
        img = Image.open(ROOT / split / "images" / name).convert("RGB")
        imgs[i] = np.asarray(img.resize((SIZE, SIZE), Image.LANCZOS))
        if (i + 1) % 500 == 0:
            print(f"{split}: {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    np.save(ROOT / f"cache_{split}_images.npy", imgs)
    np.save(ROOT / f"cache_{split}_labels.npy", labels.iloc[:, 1:].values.astype(np.float32))
    labels["image"].to_csv(ROOT / f"cache_{split}_names.csv", index=False)
    print(f"{split} done: {imgs.shape}", flush=True)


if __name__ == "__main__":
    build("train")
    build("test")
