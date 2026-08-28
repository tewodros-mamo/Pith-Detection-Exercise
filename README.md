# Pith Position Prediction — Solution

Predicts the pith x-position (mm) at 15 z-locations from one board-face image.
**Test MAE: 3.82 mm** (1000 held-out boards; naive train-mean baseline: 25.6 mm).

## Pipeline (run in order)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pandas matplotlib pillow python-pptx

python preprocess.py                                  # cache all images as 224x224 uint8 arrays (~2 min)
python train.py --head band --epochs 35 --run-name band_final   # ~65 min on a 4-core laptop CPU
python evaluate.py --run-name band_final              # test MAE + figures + predictions CSV
python make_ppt.py                                    # build outputs/pith_position_presentation.pptx
```

Head ablation (used for the presentation):

```bash
python train.py --head band --epochs 10 --run-name band_quick
python train.py --head global --epochs 10 --run-name global_quick
```

## Approach in one paragraph

A compact ~1M-parameter CNN (5 stride-2 stages) trained from scratch on 224x224
downscaled images. The head pools the final feature map over the *width only*,
keeping 15 vertical bands, and maps each band to one x-value with a shared 1x1
convolution — aligning the parameters with the geometry of the task (target i
depends mostly on rings near z-band i). Targets are normalised, loss is SmoothL1,
outputs are never clamped (pith may lie outside the board). Augmentation uses the
two exact physical symmetries: horizontal flip (x -> 145 - x) and vertical flip
(reverse the 15 targets). Model selection on a 3600/400 train/val split; the test
set was evaluated exactly once.

## Key results

| Statistic | Value |
|---|---|
| Test MAE | **3.82 mm** |
| Median abs. error | 1.78 mm |
| Pith fully inside face (948 boards) | 2.90 mm |
| Pith partly outside (52 boards) | 20.6 mm |
| Band head vs global-pool head (10-epoch val MAE) | 4.96 vs 5.58 mm |

Main limitation: boards whose pith lies far outside the face (rings nearly
straight) — the model systematically underestimates large distances.

## Files

- `preprocess.py` / `train.py` / `evaluate.py` / `make_ppt.py` — the pipeline
- `outputs/model_band_final.pt` — trained weights
- `outputs/predictions_band_final.csv` — test-set predictions
- `outputs/pith_position_presentation.pptx` — the presentation
- `outputs/*.png` — all figures (visual checks, error analysis, ablation, architecture)
