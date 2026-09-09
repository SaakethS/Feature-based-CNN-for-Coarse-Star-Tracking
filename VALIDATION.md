# Validation record — 2026-09-09 (23,319-frame retrain)

Environment: Linux host (WSL2 Ubuntu), Python 3.12.3, NumPy 2.3.5, SciPy 1.16.3.
Single-thread BLAS was used for training/benchmarks. These are host measurements.
Every number below was re-measured after two successive doublings of the training
set (5,819 -> 11,643 -> 23,319 usable frames); earlier figures are quoted
alongside where they are directly comparable.

## Build and numerical tests

- C99 build with -O2, warnings enabled, and FMA contraction disabled: successful.
- 10 regression tests: PASS (catalog failures, invalid values, policy thresholds,
  candidate cap, minimum detections, overflow, fractional PSF, attitude fit,
  model/bin integrity and equal-flux selection).
- 200 C/Python feature/network cases: PASS. Maximum histogram difference
  3.725e-09; maximum probability difference 1.288e-06 (tolerance 2e-05).
- 30 rendered images through full C inference vs C detection + Python inference:
  PASS, all 30 usable, matching shortlists. Maximum histogram difference
  3.725e-09; probability difference 1.667e-06.
- Single rendered centroid case: median positional error 0.149 px, 95th percentile
  0.346 px. This limited test does not characterize an actual camera.
- Full C pipeline on that single frame: 3.14 ms averaged over 20 warm repetitions.
  Observed 2.17-3.14 ms across invocations on a loaded host; the network size and
  operation count did not change, so this spread is host scheduling noise, not a
  model effect.
  Static pipeline state: 1227.3 KiB; input image and model storage are additional.

## Training

24,000 rendered attempts, 23,319 usable frames and 681 too-few-star rejections
(previously 12,000/11,643 and before that 6,000/5,819). Mean 24.18 visible cell
labels per frame. Catalog: 8,785 Hipparcos stars, Vmag < 6.5. All 529 labels
represented. Rendered dataset seed 1; trainer seed 0.

The epoch budget was chosen by measurement rather than assumed: a 60-epoch survey
run put the validation minimum at epoch 21, and the active model was then
retrained with `--epochs 21`. Because `train.py` restores best-validation-epoch
weights and the RNG stream is seeded deterministically, the 21-epoch run produced
**bit-identical** W1/b1/W2/b2/W3/b3/in_mean/in_scale to those the 60-epoch run
selected; this was verified by SHA-256 over each array. The short run exists so
the recorded provenance says 21 rather than 60.

Per-epoch losses: `gen/training_history.json` (active model), plus the retained
`gen/training_history_23319_ep60.json` survey run and the earlier
`gen/training_history_11643.json` and `gen/training_history_5819.json`.
`python/plot_loss.py` draws them into `gen/loss_curve.svg`.

| Training set | Train / val frames | Epochs run | Best epoch | Best validation BCE |
|---|---:|---:|---:|---:|
| 5,819 frames | 4,947 / 872 | 120 | 17 | 0.07103 |
| 11,643 frames | 9,897 / 1,746 | 120 | 18 | 0.06049 |
| 23,319 frames | 19,822 / 3,497 | 60 | 21 | 0.04745 |

The validation minimum is shallow: on the 23,319-frame run every epoch from 14 to
27 sits within 0.0005 BCE of the minimum, so epoch 21 is a noisy argmin inside a
plateau rather than a sharp optimum. Epoch 18 would have scored 0.04759, a 0.3%
difference. Do not read precision into the exact epoch number.

Overfitting slows with data but does not disappear. By the last recorded epoch,
validation BCE had risen from its minimum by 116% on 5,819 frames (to 0.15316 at
epoch 120), 55% on 11,643 frames (0.09375 at epoch 120) and 7.4% on 23,319
frames (0.05095 at epoch 60). Early epoch selection remains necessary at every
size tested.

**Validation loss is not comparable across dataset sizes.** Each run holds out
the tail 15% of its own dataset, and those are random attitudes over the same
fixed sky, so a larger dataset also places validation attitudes closer to
training attitudes. The 0.07103 -> 0.06049 -> 0.04745 series therefore overstates
the real improvement. The seed-999 evaluation below is the comparable measurement.

## Feature experiment

Separate 70/15/15 train/validation/test split, 50 epochs, seed 12, same recorded
C detections, hidden widths and selector. Usable-frame operational hit rates:

| Feature | 23,319 frames | 11,643 frames | 5,819 frames |
|---|---:|---:|---:|
| 25-bin histogram | 84.85% | 81.28% | 76.52% |
| 50-bin histogram | 95.20% | 91.41% | 88.77% |
| 100-bin histogram | 97.86% | 96.79% | 94.96% |
| Sorted nearest-neighbor angles + count | 53.60% | 48.20% | 43.18% |

The ranking is unchanged at every dataset size, so the choice of 100 bins does
not depend on how much data was available when it was made.

This exploratory comparison selected 100 bins; its test split is no longer an
unbiased final test after feature selection. Final evaluation instead uses new
seed-999 images. The supplied model uses 120 epochs and a different 85/15 split,
so its score should not be substituted into the controlled-comparison table.
See gen/feature_comparison.json.

## Independent rendered evaluation

Seed 999, 1,000 attempts, 983 usable, identical evaluation set for all three
models. 17 detector failures count as unsuccessful attempts.

| Metric | 5,819 | 11,643 | 23,319 |
|---|---:|---:|---:|
| Hit, all attempts | 94.2% | 95.9% | 96.8% |
| Hit, usable frames | 95.83% | 97.56% | 98.47% |
| Mean shortlist | 1.898 | 1.604 | 1.272 |
| Top-8 hit, usable | 97.36% | 98.58% | 98.98% |
| Coverage at 8 | 30.88% | 31.85% | 32.31% |
| Low-star (<=7 detections) hit | 78.85% | 84.62% | 90.38% |
| Confident-but-wrong | 31 of 855 | 17 of 895 | 11 of 943 |
| Empty shortlists | 0 | 1 | 0 |

Returns diminish on the headline metrics: the usable-frame hit rate gained 1.73
points on the first doubling and 0.91 on the second; top-8 gained 1.22 then 0.40.
Low-star robustness did not diminish, gaining 5.77 then 5.76 points, consistent
with the added frames being concentrated in the under-sampled hard cases.
Confident-but-wrong frames fell from 3.63% to 1.90% to 1.17% of confident frames.

The model is now both more accurate and considerably more decisive: mean shortlist
fell 33% from 1.898 to 1.272 cells while hit rate rose. That combination reduces
downstream matcher work but also reduces margin, since a smaller shortlist leaves
less room for the top cell to be wrong. Confident-but-wrong frames remain the
dominant residual failure mode. See gen/evaluation.json for full policy
comparisons; this is visible-cell localization, not attitude accuracy.

## Geometric acquisition experiment

Seed 2087, 30 rendered attempts, 27 usable, 2,000 brightest catalog stars in the
reference index. Full and assisted-with-fallback each solved 26/30 attempts within
0.1 degree of truth, unchanged across all three models. Six unique matches and
<=0.05 degree residual are required by the reference solver; at most 5,000
hypotheses are examined.

Mean online full search: 13.96 ms; mean assisted-with-fallback: 19.10 ms
(5,819-frame model: 11.25 and 25.98 ms; 11,643-frame: 12.68 and 16.86 ms).
**Still no speedup demonstrated.** The assisted penalty fell from 2.31x the full
search to 1.33x and then 1.37x, so the gain from the first doubling did not
continue; between the last two runs it is within host timing noise. Full-search
time does not depend on the model, and its variation across these runs is host
noise only. Solve counts did not change, so the smaller shortlist bought less
fallback but no measurable net win. Shared detection and index construction are excluded;
classifier overhead and the measured fallback cost are included. See
`gen/matcher_benchmark.json` for every frame, angular errors, timing and failures.
This small experiment is diagnostic, not a reliability estimate.
