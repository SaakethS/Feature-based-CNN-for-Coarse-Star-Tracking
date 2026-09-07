"""crosscheck.py — prove the C and Python paths compute the same thing.

This is the most important test in the repository. Everything else can be
re-derived; a silent divergence between the training-time feature and the
flight-time feature cannot be detected any other way, and it produces a model
that validates beautifully and fails in orbit.

What it checks
--------------
1. Feature agreement: identical star lists through Python's histogram() and
   through c/tests/crosscheck_main.c, compared element by element.
2. Network agreement: the same histograms through NumPy and through
   st_mlp_forward().
3. Centroid agreement: a rendered image through the C blob finder, compared
   against the sub-pixel positions the renderer actually used.

Run (after make in ../c):
  python3 crosscheck.py
"""

import os
import subprocess
import sys
import numpy as np

import st_features as F
from st_catalog import Catalog
from st_sim import NoiseModel, simulate_frame, render_image

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.environ.get("ST_GEN_DIR", os.path.join(HERE, "..", "gen"))
CBIN = os.path.join(HERE, "..", "c", "st_crosscheck")
IMG_W, IMG_H = 1280, 960

# Tolerances. These are tight on purpose: with -ffp-contract=off both sides
# perform the same operations in the same order in float32, so anything larger
# than accumulated rounding is a real disagreement, not noise.
HIST_TOL = 1e-6
PROB_TOL = 2e-5


def run_c(cam, uv, flux):
    """Feed one star list to the C binary and parse its output."""
    lines = [f"{float(cam.fx):.9g} {float(cam.fy):.9g} "
             f"{float(cam.cx):.9g} {float(cam.cy):.9g} "
             f"{float(cam.k1):.9g} {float(cam.k2):.9g}",
             str(len(uv))]
    for (u, v), f in zip(uv, flux):
        lines.append(f"{u:.9g} {v:.9g} {f:.9g}")
    p = subprocess.run([CBIN], input="\n".join(lines) + "\n",
                       capture_output=True, text=True)
    if p.returncode != 0:
        return None, None, p.stderr.strip()
    hist = prob = None
    for line in p.stdout.splitlines():
        parts = line.split()
        if parts[0] == "HIST":
            hist = np.array(parts[1:], dtype=np.float32)
        elif parts[0] == "PROB":
            prob = np.array(parts[1:], dtype=np.float32)
    return hist, prob, None


def numpy_forward(m, hist):
    """NumPy twin of st_mlp_forward(), including the standardization."""
    x = (hist - m["in_mean"]) / np.where(m["in_scale"] == 0, 1, m["in_scale"])
    h1 = np.maximum(x @ m["W1"].T + m["b1"], 0.0)
    h2 = np.maximum(h1 @ m["W2"].T + m["b2"], 0.0)
    z = h2 @ m["W3"].T + m["b3"]
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def write_pgm(path, img):
    with open(path, "wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
        f.write(img.tobytes())


def check_features_and_network(n_cases=200, seed=7):
    cos_edges = np.load(os.path.join(GEN, "bins.npy")).astype(np.float32)
    model = dict(np.load(os.path.join(GEN, "model.npz")))
    cam = F.Camera(fx=1800.0, fy=1800.0, cx=IMG_W / 2, cy=IMG_H / 2,
                   k1=-0.02, k2=0.004)   # nonzero distortion on purpose:
                                         # k1 = k2 = 0 would let a broken
                                         # undistortion pass unnoticed
    cat = Catalog.load()
    rng = np.random.default_rng(seed)
    noise = NoiseModel()

    worst_h = worst_p = 0.0
    n_ok = 0
    for _ in range(n_cases):
        fr = simulate_frame(cat, cam, IMG_W, IMG_H, rng, noise)
        if len(fr["uv"]) < 4:
            continue
        uv, flux = fr["uv"], fr["flux"]

        c_hist, c_prob, err = run_c(cam, uv, flux)
        if err:
            raise RuntimeError(f"C crosscheck failed: {err}")

        uv_s, _ = F.select_brightest(uv, flux)
        vecs = cam.pixels_to_vectors(uv_s)
        py_hist = F.histogram(vecs, cos_edges)
        if py_hist is None:
            continue
        py_prob = numpy_forward(model, py_hist.astype(np.float64))

        worst_h = max(worst_h, float(np.abs(c_hist - py_hist).max()))
        worst_p = max(worst_p, float(np.abs(c_prob - py_prob).max()))
        n_ok += 1

    print(f"[features] {n_ok} cases")
    print(f"  max |C - Python| histogram : {worst_h:.3e}  (tol {HIST_TOL:.0e})")
    print(f"  max |C - Python| network   : {worst_p:.3e}  (tol {PROB_TOL:.0e})")
    ok = worst_h <= HIST_TOL and worst_p <= PROB_TOL and n_ok >= max(1, int(0.8 * n_cases))
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


def check_centroids(seed=3):
    """Render an image, run the C blob finder, and see whether the recovered
    centroids match the positions the renderer placed stars at.

    This is the step that validates training on the analytic simulator. If the
    C centroider is biased by even a third of a pixel, every angle in the
    histogram shifts and the model is being trained on the wrong feature.
    """
    cam = F.Camera(fx=1800.0, fy=1800.0, cx=IMG_W / 2, cy=IMG_H / 2)
    cat = Catalog.load()
    rng = np.random.default_rng(seed)
    noise = NoiseModel()
    noise.hot_pixel_rate = (0.0, 0.0)     # isolate centroiding from false stars
    noise.psf_sigma_px = (1.1, 1.3)

    img, q, uv_true, params = render_image(cat, cam, IMG_W, IMG_H, rng, noise)
    path = os.path.join(GEN, "test_frame.pgm")
    write_pgm(path, img)

    exe = os.path.join(HERE, "..", "c", "st_pipeline_test")
    p = subprocess.run([exe, path], capture_output=True, text=True)
    print("\n[centroids] rendered frame ->", os.path.relpath(path, HERE))
    print("  " + "\n  ".join(p.stdout.strip().splitlines()))
    if p.returncode != 0:
        print("  RESULT: FAIL (" + p.stderr.strip() + ")")
        return False

    # Parse the centroids the C code actually used.
    c_uv = np.array([[float(t) for t in l.split()[1:3]]
                     for l in p.stdout.splitlines() if l.startswith("S ")])
    if len(c_uv) == 0:
        print("  RESULT: FAIL (no centroids emitted)")
        return False

    # Match each C centroid to its nearest rendered star. Only a fraction of
    # the drawn stars clear the detection threshold — that is correct physics,
    # not a bug — so the test is about POSITIONAL ACCURACY of what was
    # detected, plus the absence of unmatched spurious detections.
    d = np.linalg.norm(c_uv[:, None, :] - uv_true[None, :, :], axis=2)
    nearest = d.min(axis=1)
    matched = nearest < 1.5
    err = nearest[matched]

    print(f"  renderer drew {len(uv_true)} stars "
          f"(most below the {params['detect_sigma']:.1f}-sigma threshold)")
    print(f"  C used {len(c_uv)} centroids, {matched.sum()} matched "
          f"a rendered star within 1.5 px")
    if len(err):
        print(f"  centroid error: median {np.median(err):.3f} px, "
              f"95th pct {np.quantile(err, 0.95):.3f} px, "
              f"max {err.max():.3f} px")

    # A sub-pixel centroider that is working should land well inside half a
    # pixel on the median, and essentially everything it reports should
    # correspond to a real star.
    ok = (matched.mean() >= 0.9) and len(err) > 0 and np.median(err) < 0.5
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if not os.path.exists(CBIN):
        sys.exit("build the C first:  cd ../c && make")
    a = check_features_and_network()
    b = check_centroids()
    print("\nOVERALL:", "PASS" if (a and b) else "FAIL")
    sys.exit(0 if (a and b) else 1)
