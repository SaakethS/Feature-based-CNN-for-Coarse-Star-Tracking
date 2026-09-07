# Validation record — 2026-09-07

Environment: Linux host, Python 3.12.13, NumPy 2.3.5, SciPy 1.17.0.
Single-thread BLAS was used for training/benchmarks. These are host measurements.

## Build and numerical tests

- C99 build with -O2, warnings enabled, and FMA contraction disabled: successful.
- 10 regression tests: PASS (catalog failures, invalid values, policy thresholds,
  candidate cap, minimum detections, overflow, fractional PSF, attitude fit,
  model/bin integrity and equal-flux selection).
- 200 C/Python feature/network cases: PASS. Maximum histogram difference
  3.725e-09; maximum probability difference 8.103e-07.
- 30 rendered images through full C inference vs C detection + Python inference:
  PASS, all 30 usable, matching shortlists. Maximum histogram difference
  7.451e-09; probability difference 7.530e-07.
- Single rendered centroid case: median positional error 0.149 px, 95th percentile
  0.346 px. This limited test does not characterize an actual camera.
- Full C pipeline on that single frame: 2.29 ms averaged over 20 warm repetitions.
  Static pipeline state: 1227.3 KiB; input image and model storage are additional.

## Training

6,000 rendered attempts, 5,819 usable frames and 181 too-few-star rejections.
Catalog: 8,785 Hipparcos stars, Vmag < 6.5. All 529 labels represented.
Rendered dataset seed 1; trainer seed 0; 120 epochs; best validation epoch
17 selected by binary cross-entropy.
The 15% validation tail is drawn from independent random attitudes. Nearby pointing
attitudes can still occur across splits: this tests new observations of a fixed sky,
not generalization to an unseen catalog. Images are never real sensor observations.
The included dataset makes the supplied model reproducible using train.py and export_model.py.

## Feature experiment

Separate 70/15/15 train/validation/test split, 50 epochs, seed 12, same recorded
C detections, hidden widths and selector. Usable-frame operational hit rates:

| Feature | Hit rate |
|---|---:|
| 25-bin histogram | 76.52% |
| 50-bin histogram | 88.77% |
| 100-bin histogram | 94.96% |
| Sorted nearest-neighbor angles + count | 43.18% |

This exploratory comparison selected 100 bins; its test split is no longer an
unbiased final test after feature selection. Final evaluation instead uses new
seed-999 images. The supplied model uses 120 epochs and a different 85/15 split,
so its score should not be substituted into the controlled-comparison table.
See gen/feature_comparison.json.

## Independent rendered evaluation

Seed 999, 1,000 attempts, 983 usable. At least one overlapping cell was selected
in 942 attempts (94.2% overall, 95.83% usable). Mean shortlist: 1.898 usable-frame
candidates. 17 detector failures count as unsuccessful attempts. Among 855 frames
above the 0.90 confidence threshold, 31 missed all visible cells. Low-star hit rate
on usable frames with <=7 detections: 78.85%. See gen/evaluation.json for full
policy comparisons; this is visible-cell localization, not attitude accuracy.

## Geometric acquisition experiment

Seed 2087, 30 rendered attempts, 27 usable, 2,000 brightest catalog stars in the
reference index. Full and assisted-with-fallback each solved 26/30 attempts within
0.1 degree of truth. Six unique matches and <=0.05 degree residual are required
by the reference solver; at most 5,000 hypotheses are examined.

Mean online full search: 11.25 ms; mean assisted-with-fallback: 25.98 ms.
**No speedup demonstrated.** Shared detection and index construction are excluded;
classifier overhead and the measured fallback cost are included. See
`gen/matcher_benchmark.json` for every frame, angular errors, timing and failures.
This small experiment is diagnostic, not a reliability estimate.
