# Star Tracker ML Cell Classification — C flight path + Python ground tools

A working skeleton of the architecture in *Star Tracker ML Cell-Classification
Architecture*: an onboard classifier that turns a star image into a shortlist of
sky cells, shrinking the lost-in-space search space before the pattern matcher
and EKF run.

**Everything that runs in orbit is C. Everything that runs on the ground is
Python.** The two are kept honest by a cross-check that feeds identical inputs
to both and compares the numbers.

---

## Quick start

```bash
cd python
python3 make_dataset.py --n 20000     # synthetic frames -> gen/dataset.npz
python3 train.py --epochs 60          # -> gen/model.npz
python3 export_model.py               # -> gen/model_params.h

cd ../c && make                       # build the flight path
cd ../python && python3 crosscheck.py # prove C == Python
```

---

## Layout

```
c/
  include/st_config.h     compile-time limits (image size, star cap, bins, cells)
  include/st_types.h      shared plain-data types and return codes
  src/st_centroid.c       image -> sub-pixel star centroids
  src/st_camera.c         pixel -> unit direction vector (with undistortion)
  src/st_feature.c        vectors -> pairwise angular distance histogram
  src/st_mlp.c            the trained network's forward pass
  src/st_select.c         probabilities -> candidate cell shortlist
  src/st_pipeline.c       the one call flight software makes
  tests/crosscheck_main.c exposes the feature + network stages to Python
  tests/pipeline_main.c   runs a PGM through the whole path and times it

python/
  st_cells.py             the 529-cell sky partition (the definition of a label)
  st_catalog.py           Hipparcos loader, with a synthetic fallback
  st_features.py          reference twin of st_camera.c + st_feature.c
  st_sim.py               synthetic frames and the domain-randomization model
  make_dataset.py         builds the training set and fits the bin edges
  train.py                NumPy multi-label trainer (BCE, Adam)
  export_model.py         writes gen/model_params.h
  crosscheck.py           THE test — C vs Python, feature and network

gen/                      generated: bins, dataset, weights, model_params.h
```

## Design decisions worth knowing

**No TensorFlow Lite.** The network is 25 → 96 → 96 → 529 — three matrix-vector
products. `st_mlp.c` is about eighty lines and has no dependencies beyond
`libm`. That removes a cross-compiled runtime from the flight build and keeps
the arithmetic auditable, which is the explainability argument the architecture
document already commits to. TFLite earns its keep for large convolutional
models; this is not one.

**No `malloc`, anywhere.** Every buffer is statically sized from `st_config.h`,
so worst-case memory is known at compile time. `st_pipeline_t` is about 1.2 MB,
dominated by the one-byte-per-pixel detection mask.

**No `acos` in the feature.** The angle between unit vectors **a** and **b** is
acos(**a**·**b**), but a histogram only needs the bin, and cosine is monotonic
on [0, π]. Bin edges are stored in cosine space, descending, and compared
against raw dot products — 276 transcendental calls saved per frame.

**Star cap of 24.** Pair count grows as N(N−1)/2. Capping at the 24 brightest
gives 276 pairs and, more importantly, makes the feature well-defined: training
and flight must agree on *which* stars go into the histogram, not just how they
are binned.

**Standardization lives inside the model.** The input mean and scale are
exported alongside the weights and applied in `st_mlp_forward`, so the C and
Python paths cannot disagree about whether the transform was applied.

**Compile-time guards.** `model_params.h` contains `#error` directives that fail
the build if `ST_NUM_BINS`, `ST_NUM_CELLS`, `ST_MAX_STARS`, or the hidden layer
widths ever drift from what the model was trained with.

**`-ffp-contract=off`.** By default GCC may fuse multiply-add pairs, which is
more accurate but gives different results from NumPy. Disabling it is what lets
the cross-check use a 1e-6 tolerance instead of a loose one that would hide real
bugs.

## What the cross-check proves

```
[features] 200 cases
  max |C - Python| histogram : 7.5e-09   (tol 1e-06)
  max |C - Python| network   : 8.0e-07   (tol 2e-05)
[centroids]
  C used 24 centroids, 24 matched a rendered star within 1.5 px
  centroid error: median 0.385 px, 95th pct 0.648 px
```

The feature test runs with non-zero distortion coefficients on purpose: with
k1 = k2 = 0 a broken undistortion would pass unnoticed.

Timing on a desktop x86 core is about 2.9 ms/frame for the whole path,
centroiding included. Expect roughly 3–6× that on a Pi 5 core; `make pi`
cross-compiles with `-mcpu=cortex-a76`.

---

## What is still a placeholder

These are marked in the code and must be replaced before any of the numbers
mean anything:

1. **The catalog.** `st_catalog.py` falls back to a uniform random sky. The real
   sky is clumpy — the galactic plane above all — and a uniform fake sky makes
   cells look more alike than they are. Point it at Hipparcos.
2. **The 529-cell convention.** `st_cells.py` implements an equal-area zonal
   partition that produces exactly 529 cells. If the GLAS document specifies a
   particular tessellation, replace this file with that one. Confirm with
   Dr. Lee before generating a full training set — every label depends on it.
3. **Camera calibration.** `st_pipeline_init` ships a placeholder focal length.
   A wrong focal length silently rescales every angle, and the histogram will
   not match anything the network was trained on.
4. **Thresholds in `st_select_default_cfg`.** Set these from a precision/recall
   sweep on held-out attitudes, not by feel.

## Two things to check early

**Field of view versus cell size.** 529 equal-area cells are about 9° across. At
the default focal length the frame is a 48° diagonal, so about 24 cells fall in
view at once and the "shortlist" cannot be shorter than that. Real CubeSat star
trackers are usually 10–20°. Either narrow the field of view or coarsen the
partition so a frame covers a small number of cells — otherwise the classifier
is being asked a question whose answer is always "about two dozen".

**Section 10 of the architecture document.** Before building anything further,
confirm the feature is actually discriminative: different cells must produce
distinguishable histograms. `make_dataset.py` gives you the data to check it.
