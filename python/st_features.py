"""st_features.py — the reference (ground-truth) feature extractor.

This file is the Python twin of c/src/st_camera.c and c/src/st_feature.c.
Every operation here is mirrored there, in the same order, in float32.

Rule for this file: if you change anything, change the C and re-run the
cross-check. A silent divergence between these two implementations is the
single most likely way for this project to produce a model that scores well in
training and fails in orbit.
"""

import numpy as np

NUM_BINS = 25
MAX_STARS = 24
UNDISTORT_ITERS = 5     # must equal ST_UNDISTORT_ITERS in st_camera.c

F32 = np.float32


class Camera:
    """Pinhole camera with two radial distortion terms.

    fx, fy : focal length in pixels
    cx, cy : principal point in pixels
    k1, k2 : radial distortion coefficients
    """

    def __init__(self, fx, fy, cx, cy, k1=0.0, k2=0.0):
        self.fx, self.fy = F32(fx), F32(fy)
        self.cx, self.cy = F32(cx), F32(cy)
        self.k1, self.k2 = F32(k1), F32(k2)

    def fov_deg(self, width, height):
        """Diagonal field of view in degrees, for sanity-checking a lens."""
        import math
        hx = 0.5 * width / float(self.fx)
        hy = 0.5 * height / float(self.fy)
        return math.degrees(2.0 * math.atan(math.hypot(hx, hy)))

    def project(self, v, jitter=0.0, rng=None):
        """Unit vectors in the CAMERA frame -> pixel coordinates.

        This is the forward model used by the renderer. Vectors with z <= 0 are
        behind the camera and are returned as NaN so the caller can drop them.
        """
        v = np.asarray(v, dtype=np.float64)
        z = v[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = np.where(z > 1e-9, v[:, 0] / z, np.nan)
            yn = np.where(z > 1e-9, v[:, 1] / z, np.nan)
        r2 = xn * xn + yn * yn
        d = 1.0 + float(self.k1) * r2 + float(self.k2) * r2 * r2
        u = float(self.fx) * xn * d + float(self.cx)
        w = float(self.fy) * yn * d + float(self.cy)
        if jitter > 0.0:
            rng = rng or np.random.default_rng()
            u = u + rng.normal(0.0, jitter, u.shape)
            w = w + rng.normal(0.0, jitter, w.shape)
        return np.stack([u, w], axis=1)

    def pixels_to_vectors(self, uv):
        """Pixel coordinates -> unit direction vectors. Mirrors st_camera.c.

        The distortion inversion is a fixed-point iteration with exactly
        UNDISTORT_ITERS steps, in float32, matching the C bit for bit up to
        the compiler's freedom in reassociating the arithmetic.
        """
        uv = np.asarray(uv, dtype=F32)
        xd = ((uv[:, 0] - self.cx) / self.fx).astype(F32)
        yd = ((uv[:, 1] - self.cy) / self.fy).astype(F32)

        xn, yn = xd.copy(), yd.copy()
        for _ in range(UNDISTORT_ITERS):
            r2 = (xn * xn + yn * yn).astype(F32)
            denom = (F32(1.0) + self.k1 * r2 + self.k2 * r2 * r2).astype(F32)
            denom = np.maximum(denom, F32(1e-6))
            xn = (xd / denom).astype(F32)
            yn = (yd / denom).astype(F32)

        norm = np.sqrt(xn * xn + yn * yn + F32(1.0)).astype(F32)
        return np.stack([xn / norm, yn / norm, F32(1.0) / norm],
                        axis=1).astype(F32)


def select_brightest(uv, flux, keep=MAX_STARS):
    """Keep the `keep` brightest stars, in descending flux order.

    Mirrors st_select_brightest(). Ties are broken by index, matching the C
    selection sort, which takes the first strictly-greater element only.
    """
    order = np.argsort(-np.asarray(flux, dtype=np.float64), kind="stable")
    order = order[:keep]
    return np.asarray(uv)[order], np.asarray(flux)[order]


def histogram(vecs, cos_edges):
    """Pairwise angular distance histogram. Mirrors st_histogram().

    vecs      : (n, 3) float32 unit vectors
    cos_edges : (NUM_BINS+1,) float32, DESCENDING (cosine space)
    returns   : (NUM_BINS,) float32, L1-normalized, or None if no pair is in
                range (the C code returns ST_ERR_TOO_FEW in that case)
    """
    vecs = np.asarray(vecs, dtype=F32)
    n = min(len(vecs), MAX_STARS)
    if n < 2:
        return None
    vecs = vecs[:n]

    # All pairs i < j at once. Same set of pairs as the C double loop.
    i, j = np.triu_indices(n, k=1)
    d = np.einsum("ij,ij->i", vecs[i], vecs[j]).astype(F32)
    d = np.clip(d, F32(-1.0), F32(1.0))

    cos_edges = np.asarray(cos_edges, dtype=F32)
    # Bin b holds pairs with cos_edges[b] >= d > cos_edges[b+1]. Because the
    # edges descend, searchsorted needs the reversed array.
    inside = (d <= cos_edges[0]) & (d > cos_edges[NUM_BINS])
    d_in = d[inside]
    if d_in.size == 0:
        return None

    rev = cos_edges[::-1]                       # ascending
    idx_rev = np.searchsorted(rev, d_in, side="left")
    b = NUM_BINS - idx_rev                      # map back to descending bins
    b = np.clip(b, 0, NUM_BINS - 1)

    hist = np.bincount(b, minlength=NUM_BINS).astype(F32)
    return (hist / F32(d_in.size)).astype(F32)


def pairwise_angles(vecs):
    """Every pairwise angle in radians. Used only for fitting the bin edges."""
    vecs = np.asarray(vecs, dtype=np.float64)
    i, j = np.triu_indices(len(vecs), k=1)
    d = np.clip(np.einsum("ij,ij->i", vecs[i], vecs[j]), -1.0, 1.0)
    return np.arccos(d)


def fit_quantile_bins(all_angles, n_bins=NUM_BINS, lo_q=0.005, hi_q=0.995):
    """Quantile (adaptive) bin edges, returned in DESCENDING cosine space.

    Uniform bins waste resolution: pairwise separations in a fixed field of
    view cluster around a characteristic angle, so most uniform bins would sit
    nearly empty while a few carry everything. Placing edges at equally spaced
    quantiles of the observed distribution puts fine bins where the data
    actually is, which is what gives each cell a distinguishable fingerprint.

    The extreme tails are trimmed at lo_q/hi_q so a handful of outlier pairs
    do not stretch the range.
    """
    a = np.asarray(all_angles, dtype=np.float64)
    a = a[np.isfinite(a)]
    lo, hi = np.quantile(a, lo_q), np.quantile(a, hi_q)
    a = a[(a >= lo) & (a <= hi)]
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges_angle = np.quantile(a, qs)

    # Guarantee strict monotonicity even if the distribution has a spike.
    for k in range(1, len(edges_angle)):
        if edges_angle[k] <= edges_angle[k - 1]:
            edges_angle[k] = edges_angle[k - 1] + 1e-7

    # cos is decreasing on [0, pi], so cosine-space edges descend.
    return np.cos(edges_angle).astype(F32), edges_angle
