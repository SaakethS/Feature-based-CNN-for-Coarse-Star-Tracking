# Feature-based neural sky-cell classification for coarse star tracking

A research acquisition pipeline: 8-bit image → C star detection → calibrated
unit vectors → angular histogram → MLP → shortlist of 529 sky cells.
A ground-only geometric matcher estimates attitude and benchmarks full-search
fallback. The implemented neural network is an MLP, not a CNN. No EKF is included.

## What is included

- C99 detector, radial camera model, 100-bin angular feature, 100 → 96 → 96 → 529
  MLP (70,321 trainable parameters), and bounded shortlist selection.
- Python rendered-image training through the **actual C detector**, plus an
  explicitly approximate analytic simulator for exploratory experiments.
- A source-attributed 8,785-star Hipparcos subset with V magnitude < 6.5.
- 23,319 usable training/validation frames from 24,000 rendered attempts, recorded
  detections and attitudes, trained weights, matching bins and generated C header.
- Per-epoch learning curves (`gen/training_history*.json`) and a dependency-free
  SVG plot comparing the 5,819-, 11,643- and 23,319-frame runs.
- Exact C-policy evaluation, feature comparison, isolated FOV experiments,
  geometric acquisition benchmark, regression tests and recorded results.
- Original 25-bin weights preserved under `gen/legacy/`; these are not active.

## Quick start (Linux or Windows with WSL)

Use a Linux/WSL terminal with Python 3.10+, a C99 compiler, and GNU Make.
The host bridge uses a Linux shared library; native Windows DLL builds are not
provided. Run from the extracted `startracker` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
make -C c all bridge
python3 python/test_regressions.py
python3 python/crosscheck.py
python3 python/test_end_to_end.py
python3 python/evaluate.py --n 1000 --out gen/evaluation.json
```

The supplied model is already trained. For training again on the included dataset:

```bash
OPENBLAS_NUM_THREADS=1 python3 python/train.py --epochs 21
python3 python/export_model.py
make -C c all bridge
```

21 is where this dataset's validation loss bottoms out. `train.py` restores the
best-validation-epoch weights regardless of the budget, so a longer run selects
the same weights; the short run just makes the recorded provenance honest.
Re-survey the curve with a longer run before assuming 21 still holds for a
different dataset size, learning rate or batch size.

`train.py` also writes `gen/training_history.json`, one record per epoch. To draw
the learning curve, on its own or against the retained earlier runs:

```bash
python3 python/plot_loss.py --runs gen/training_history.json
python3 python/plot_loss.py --out gen/loss_curve.svg \
    --runs gen/training_history_5819.json gen/training_history_11643.json \
           gen/training_history_23319_ep60.json
```

To generate a new dataset first (this replaces the active dataset and bins):

```bash
python3 python/make_dataset.py --n 24000 --seed 1 --mode rendered --catalog data/hipparcos_mag65.csv
python3 python/train.py --epochs 60   # then retrain at the observed best epoch
python3 python/export_model.py
make -C c all bridge
python3 python/evaluate.py --n 1000 --seed 999 --out gen/evaluation.json
```

Regenerate weights after regenerating bins. Hash checks deliberately reject a
mixed model/bin combination. Do not change the C bin count without matching
Python changes, retraining and export. Header guards detect ABI mismatch.
The active model's camera constants are exported together with its weights.
Fresh dataset generation uses independent bin-fitting scenes; the bundled dataset
was rebinned using only its first 70% of recorded detections after feature selection.
Both keep evaluation frames out of bin fitting. Metadata records that distinction.

## Measured results

See `VALIDATION.md` and the JSON files under `gen/` for exact definitions and limits.
On 1,000 new rendered Hipparcos scenes, seed 999:

| Metric | 5,819 frames | 11,643 frames | 23,319 frames |
|---|---:|---:|---:|
| Attempts / usable frames | 1,000 / 983 | 1,000 / 983 | 1,000 / 983 |
| Visible-cell hit rate, counting detector failures | 94.2% | 95.9% | **96.8%** |
| Visible-cell hit rate on usable frames | 95.83% | 97.56% | **98.47%** |
| Mean shortlist on usable frames | 1.90 | 1.60 | **1.27 cells** |
| Top-eight hit rate on usable frames | 97.36% | 98.58% | **98.98%** |
| Low-star (<=7 detections) hit rate on usable frames | 78.85% | 84.62% | **90.38%** |
| Confident-but-wrong frames | 31 of 855 | 17 of 895 | **11 of 943** |
| Random top-eight baseline | 31.35% | 31.35% | 31.35% |

Every column uses the same evaluation: seed 999, 1,000 fresh rendered scenes, the
same 17 detector failures, and the C selector under the default policy. The only
change is the training set size.

Returns are diminishing on the metrics that matter most. Each doubling cut the
usable-frame miss rate by roughly half the previous doubling's gain: +1.73
points, then +0.91. Top-eight hit rate moved +1.22, then +0.40. Low-star
robustness is the exception, still gaining about 5.8 points per doubling, which
is what to expect when the frames being added are the under-sampled hard cases.

Validation cross-entropy fell 0.07103 -> 0.06049 -> 0.04745, but that series
overstates the gain: each run's holdout is the tail 15% of its own dataset, so a
larger dataset also places validation attitudes closer to training attitudes.
Compare models on the seed-999 table above, not on validation loss. See
`gen/loss_curve.svg` for all three curves.

A visible-cell hit means at least one selected cell overlaps the rendered field.
It does **not** mean a correct star identification or attitude solution. FOV labels
use an approximate 12×12 sampling grid and can miss very small boundary overlaps.

Default policy: accept scores >= 0.15, keep at most eight, and reduce to one when
the highest score >= 0.90. Evaluation calls the C selector itself. The output score
is not a calibrated probability of a correct attitude solution. The saved policy
sweep shows the recall/candidate-cost tradeoff, including disabling single-cell
collapse with a confidence threshold > 1.0. Thresholds remain research settings;
no mission reliability target was supplied.

## Experiments

```bash
OPENBLAS_NUM_THREADS=1 python3 python/benchmark_features.py --epochs 50
OPENBLAS_NUM_THREADS=1 python3 python/benchmark_matcher.py --n 30
OPENBLAS_NUM_THREADS=1 python3 python/sweep_fov.py --n 6000 --eval-n 500
```

The feature benchmark uses one 70/15/15 split of the 23,319-frame dataset for
25/50/100-bin histograms and a nearest-neighbor feature. These alternatives are research experiments; the active
C feature stays at 100 bins. The FOV sweep creates isolated `experiments/fx_*`
artifacts and keeps 529 cells fixed to match the C ABI. It does not overwrite the
active model. Arbitrary cell-count sweeps need a separately configured C build.

The ground matcher indexes pair angles from an explicit bright-star catalog subset
(default 2,000), tests rotation hypotheses, verifies at least six unique matches,
and refines attitude with SVD. It expands selected cells conservatively to include
potential image stars. Therefore two shortlisted cells do **not** imply searching
only 2/529 of the catalog. A rejected/unsolved shortlist falls back to full search
in the benchmark. Offline index construction and shared detection time are excluded
from online matcher comparison; classifier overhead is included. A 5,000-hypothesis
budget bounds each solve. This is a reference benchmark, not a flight matcher.

## Limitations and next work

- Results are simulated, not measured on real camera images or flight hardware.
- Hipparcos coordinates are at epoch J1991.25. Proper motion, aberration, and
  production catalog quality filtering remain to be implemented. See `data/SOURCE.md`.
- The renderer uses a sampled Gaussian PSF, read noise, hot pixels, variable
  brightness/background and detection thresholds. It does not model photon shot
  noise, spatial background gradients, motion smear, or all detector artifacts.
- The analytic simulator's threshold approximation is not a substitute for the
  C detector. Use `--mode rendered` for primary validation.
- Low-star scenes and overconfident wrong classifications remain failure modes.
- The 30-frame matcher experiment still did not demonstrate a speedup: assisted
  acquisition with fallback remained slower than the full reference search, though
  the gap narrowed from 2.3x to about 1.4x as the shortlist got smaller and more
  often correct. Optimize and benchmark
  a realistic onboard catalog index before making runtime-benefit claims.
- Fixed buffers bound allocation, not full flight qualification. Detector overflow
  now rejects a frame rather than trusting a truncated star list.
- The ctypes bridge has static workspaces and is not thread-safe. Use separate
  processes for parallel data generation.

## Folder layout

`c/`: production C and host tests/bridge. `python/`: training and experiments.
`data/`: attributed real catalog. `gen/`: reproducible dataset, model, learning
curves and reports.
The architecture PDF is the original design reference and predates these changes.
`CHANGELOG.md` lists the implementation changes. The ZIP contains no Git metadata:
copy its contents into your existing repository while retaining your own `.git`.
