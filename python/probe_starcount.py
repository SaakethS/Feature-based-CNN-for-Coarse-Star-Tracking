"""probe_starcount.py — is the narrow-field failure about geometry or stars?

Narrowing the field of view does two things at once, and the sweep cannot tell
them apart:

  1. it makes the question harder in the way we WANT (fewer cells in view), and
  2. it puts fewer stars in the frame, which makes the feature weaker, because
     a pairwise-angle histogram built from n stars has n(n-1)/2 pairs and its
     angular support shrinks with the field.

At fx = 6000 the frame holds about 24 stars down to magnitude 6.5, so the
histogram has ~276 pairs spread over a ~15 degree range instead of ~48. If the
collapse is cause 2, a more sensitive detector fixes it and the architecture is
fine. If it survives a deeper catalog, the 25-bin pairwise histogram is simply
not discriminative enough on its own and the feature vector must be widened.

This holds the field of view fixed and varies only the limiting magnitude.

Run:
  python3 probe_starcount.py
"""

import os
import numpy as np

import st_features as F
from st_catalog import Catalog
from st_cells import vectors_to_cells, NUM_CELLS
from st_sim import NoiseModel, simulate_frame
from make_dataset import frame_cells
from train import MLP, evaluate as train_eval

IMG_W, IMG_H = 1280, 960
FX = 4000.0            # 22.6 deg diagonal, held fixed
N_TRAIN, N_EVAL, EPOCHS = 20000, 4000, 60


def build(cat, cam, n, rng, noise, cos_edges=None):
    """Frames -> (histograms, multi-hot labels). Fits bins if none given."""
    if cos_edges is None:
        angles = []
        for _ in range(min(n, 2000)):
            fr = simulate_frame(cat, cam, IMG_W, IMG_H, rng, noise)
            if len(fr["uv"]) < 2:
                continue
            uv, _ = F.select_brightest(fr["uv"], fr["flux"])
            angles.append(F.pairwise_angles(cam.pixels_to_vectors(uv)))
        cos_edges, _ = F.fit_quantile_bins(np.concatenate(angles))

    X = np.zeros((n, F.NUM_BINS), np.float32)
    Y = np.zeros((n, NUM_CELLS), np.float32)
    nstars, kept = [], 0
    for _ in range(n):
        fr = simulate_frame(cat, cam, IMG_W, IMG_H, rng, noise)
        if len(fr["uv"]) < 4:
            continue
        uv, _ = F.select_brightest(fr["uv"], fr["flux"])
        h = F.histogram(cam.pixels_to_vectors(uv), cos_edges)
        if h is None:
            continue
        nstars.append(min(len(fr["uv"]), F.MAX_STARS))
        X[kept] = h
        Y[kept, frame_cells(cam, fr["R"], IMG_W, IMG_H)] = 1.0
        kept += 1
    return X[:kept], Y[:kept], cos_edges, float(np.mean(nstars))


def main():
    cam = F.Camera(fx=FX, fy=FX, cx=IMG_W / 2, cy=IMG_H / 2)
    print(f"fx={FX:.0f}, {cam.fov_deg(IMG_W, IMG_H):.1f} deg diagonal, "
          f"{NUM_CELLS} cells -- held fixed\n")
    print(f"{'mag limit':>10s} {'stars/frame':>12s} {'in view':>8s} "
          f"{'top-8':>7s} {'shortlist':>10s}")
    print("-" * 52)

    for mag in (6.5, 7.5, 8.5, 9.5):
        cat = Catalog.load(None, mag_limit=mag)
        noise = NoiseModel()
        noise.mag_limit = (mag - 1.0, mag)      # detector sensitivity follows
        rng = np.random.default_rng(1)

        Xtr, Ytr, edges, ns = build(cat, cam, N_TRAIN, rng, noise)
        Xva, Yva, _, _ = build(cat, cam, N_EVAL,
                               np.random.default_rng(999), noise, edges)

        mean = Xtr.mean(0).astype(np.float32)
        scale = Xtr.std(0).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        Ztr, Zva = (Xtr - mean) / scale, (Xva - mean) / scale

        r = np.random.default_rng(0)
        model = MLP(Xtr.shape[1], Ytr.shape[1], r)
        for _ in range(EPOCHS):
            idx = r.permutation(len(Ztr))
            for s in range(0, len(idx), 256):
                model.step(Ztr[idx[s:s + 256]], Ytr[idx[s:s + 256]], 3e-3)

        rec, sl = train_eval(model, Zva, Yva)
        print(f"{mag:10.1f} {ns:12.1f} {Yva.sum(1).mean():8.1f} "
              f"{rec:7.3f} {sl:7.1f}/{NUM_CELLS}")

    print("\ncatalog size grows as 10^(0.48 m); the star cap is "
          f"{F.MAX_STARS}, so past the point where the frame holds more than "
          f"{F.MAX_STARS} stars\nonly the SELECTION changes, not the count.")


if __name__ == "__main__":
    main()
