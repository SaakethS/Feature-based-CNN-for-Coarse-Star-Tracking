# Reproducing included results

Run from the repository root, Linux/WSL, after installing requirements.
Set OPENBLAS_NUM_THREADS=1 for comparable CPU timings and deterministic BLAS ordering.

The saved dataset already contains the active 100-bin features and metadata.
It holds 23,319 usable frames from 24,000 rendered attempts at seed 1.

Epoch 21 is where this dataset's validation loss bottoms out, established by a
60-epoch survey run kept in gen/training_history_23319_ep60.json. train.py
restores best-validation-epoch weights whatever the budget, so a longer run
selects the same weights bit for bit; 21 is used so the recorded provenance
matches what was actually run.

```bash
OPENBLAS_NUM_THREADS=1 python3 python/train.py --epochs 21 --seed 0
python3 python/export_model.py
make -C c all bridge
OPENBLAS_NUM_THREADS=1 python3 python/evaluate.py --n 1000 --seed 999 --out gen/evaluation.json
OPENBLAS_NUM_THREADS=1 python3 python/benchmark_features.py --epochs 50
OPENBLAS_NUM_THREADS=1 python3 python/benchmark_matcher.py --n 30 --seed 2087
python3 python/plot_loss.py --out gen/loss_curve.svg \
    --runs gen/training_history_5819.json gen/training_history_11643.json \
           gen/training_history_23319_ep60.json
```

To rebuild the dataset itself, which replaces gen/dataset.npz and gen/bins.npy
and therefore requires retraining and re-exporting before the C build is valid:

```bash
python3 python/make_dataset.py --n 24000 --seed 1 --mode rendered \
                               --catalog data/hipparcos_mag65.csv
```

Generation is single-process: the ctypes bridge has static workspaces and is not
thread-safe. 24,000 attempts took about 11 minutes of CPU.

gen/training_history_5819.json and gen/training_history_11643.json are the two
earlier dataset sizes trained under identical settings. Reproducing them needs
those datasets, which later regenerations replaced; the files are kept so the
learning-curve comparison stays checkable without regenerating 18,000 frames.
After changing dataset size, re-survey the epoch budget with a longer run rather
than reusing 21.

The saved feature comparison was run before changing the active feature to 100
bins, but it constructs all compared features independently from stored vectors.
For the exact bundled bin definition, quantile edges were fitted on pair angles
from the first 70% of dataset vectors using st_features.fit_quantile_bins;
histograms were then reconstructed for every stored vector set.
Fresh make_dataset.py uses an independent fitting stream and is a new experiment,
not a bitwise reproduction of the bundled bin-fitting stream.
Legacy artifacts are unverified original weights; do not mix with active headers.
