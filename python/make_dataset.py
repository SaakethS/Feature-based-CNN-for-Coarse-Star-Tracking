"""make_dataset.py — build the training set.

Produces:
  gen/bins.npy      cosine-space bin edges (descending), float32
  gen/dataset.npz   X (n, NUM_BINS) histograms, Y (n, NUM_CELLS) multi-hot

Run:
  python3 make_dataset.py --n 40000 --seed 1

Two-pass design: the first pass collects raw pairwise angles so the quantile
bin edges can be fitted to the distribution the sensor actually produces; the
second pass builds histograms using those edges. Fitting the bins on the same
data you then histogram is fine here — the bins are a property of the optics
and the sky, not of any individual label.
"""

import argparse
import os
import numpy as np

import st_features as F
from st_cells import vectors_to_cells, NUM_CELLS
from st_catalog import Catalog
from st_sim import NoiseModel, simulate_frame

IMG_W, IMG_H = 1280, 960
GEN = os.path.join(os.path.dirname(__file__), "..", "gen")


def frame_cells(cam, R_ci, width, height, grid=12):
    """Every cell overlapping the field of view for this attitude.

    Rather than testing all 529 cells for intersection with the frame, sample a
    grid of pixel positions across the image, map each to the sky, and take the
    set of cells hit. With a cell about 9 degrees across and a field of view of
    roughly the same scale, a 12x12 grid cannot skip a cell that covers any
    meaningful part of the frame.
    """
    us = np.linspace(0, width - 1, grid)
    vs = np.linspace(0, height - 1, grid)
    uu, vv = np.meshgrid(us, vs)
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
    v_cam = cam.pixels_to_vectors(uv).astype(np.float64)
    v_in = v_cam @ R_ci.T                      # camera -> inertial
    return np.unique(vectors_to_cells(v_in))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="training frames")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--catalog", default=None, help="Hipparcos CSV, optional")
    ap.add_argument("--fx", type=float, default=1800.0)
    args = ap.parse_args()

    os.makedirs(GEN, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cam = F.Camera(fx=args.fx, fy=args.fx,
                   cx=IMG_W / 2.0, cy=IMG_H / 2.0, k1=0.0, k2=0.0)
    print(f"camera: fx={args.fx:.0f} px, diagonal FOV "
          f"{cam.fov_deg(IMG_W, IMG_H):.2f} deg")

    cat = Catalog.load(args.catalog)
    print(f"catalog: {len(cat)} stars"
          f"{' (SYNTHETIC PLACEHOLDER)' if args.catalog is None else ''}")
    noise = NoiseModel()

    # ---- pass 1: fit the bin edges -------------------------------------
    n_fit = min(args.n, 2000)
    angles = []
    for _ in range(n_fit):
        fr = simulate_frame(cat, cam, IMG_W, IMG_H, rng, noise)
        if len(fr["uv"]) < 2:
            continue
        uv, _ = F.select_brightest(fr["uv"], fr["flux"])
        vecs = cam.pixels_to_vectors(uv)
        angles.append(F.pairwise_angles(vecs))
    angles = np.concatenate(angles)
    cos_edges, ang_edges = F.fit_quantile_bins(angles)
    np.save(os.path.join(GEN, "bins.npy"), cos_edges)
    print(f"bins fitted on {len(angles)} pairs, angular range "
          f"{np.degrees(ang_edges[0]):.2f}..{np.degrees(ang_edges[-1]):.2f} deg")

    # ---- pass 2: histograms and labels ---------------------------------
    X = np.zeros((args.n, F.NUM_BINS), dtype=np.float32)
    Y = np.zeros((args.n, NUM_CELLS), dtype=np.float32)
    kept = 0
    for _ in range(args.n):
        fr = simulate_frame(cat, cam, IMG_W, IMG_H, rng, noise)
        if len(fr["uv"]) < 4:
            continue                      # too few stars to say anything
        uv, _ = F.select_brightest(fr["uv"], fr["flux"])
        vecs = cam.pixels_to_vectors(uv)
        h = F.histogram(vecs, cos_edges)
        if h is None:
            continue
        cells = frame_cells(cam, fr["R"], IMG_W, IMG_H)
        X[kept] = h
        Y[kept, cells] = 1.0
        kept += 1

    X, Y = X[:kept], Y[:kept]
    np.savez_compressed(os.path.join(GEN, "dataset.npz"), X=X, Y=Y)
    print(f"kept {kept}/{args.n} frames")
    print(f"labels per frame: mean {Y.sum(1).mean():.2f}, max {Y.sum(1).max():.0f}")
    print(f"cells never seen: {(Y.sum(0) == 0).sum()}/{NUM_CELLS}")


if __name__ == "__main__":
    main()
