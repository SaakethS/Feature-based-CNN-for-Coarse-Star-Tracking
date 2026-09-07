# Reproducing included results

Run from the repository root, Linux/WSL, after installing requirements.
Set OPENBLAS_NUM_THREADS=1 for comparable CPU timings and deterministic BLAS ordering.

The saved dataset already contains the active 100-bin features and metadata.

```bash
OPENBLAS_NUM_THREADS=1 python3 python/train.py --epochs 120 --seed 0
python3 python/export_model.py
make -C c all bridge
OPENBLAS_NUM_THREADS=1 python3 python/evaluate.py --n 1000 --seed 999 --out gen/evaluation.json
OPENBLAS_NUM_THREADS=1 python3 python/benchmark_features.py --epochs 50
OPENBLAS_NUM_THREADS=1 python3 python/benchmark_matcher.py --n 30 --seed 2087
```

The saved feature comparison was run before changing the active feature to 100
bins, but it constructs all compared features independently from stored vectors.
For the exact bundled bin definition, quantile edges were fitted on pair angles
from the first 70% of dataset vectors using st_features.fit_quantile_bins;
histograms were then reconstructed for every stored vector set.
Fresh make_dataset.py uses an independent fitting stream and is a new experiment,
not a bitwise reproduction of the bundled bin-fitting stream.
Legacy artifacts are unverified original weights; do not mix with active headers.
