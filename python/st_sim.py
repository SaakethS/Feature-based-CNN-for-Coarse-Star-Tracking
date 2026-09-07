"""st_sim.py — synthetic star fields, with a physically motivated noise model.

Two levels of simulation live here, and the distinction matters:

  simulate_frame()  — analytic. Projects catalog stars to pixel coordinates and
                      perturbs them. Fast enough to make a hundred thousand
                      training samples. This is what you train on.

  render_image()    — draws an actual 8-bit image with a Gaussian point spread
                      function. Perhaps a thousand times slower. This is what
                      you use to verify that the C centroider recovers the star
                      positions the analytic path assumes, and to check the
                      detection threshold behaviour.

Training on the analytic path alone is only safe because render_image() is used
to validate that the two agree. Skipping that validation is how a model ends up
trained on a star list the flight centroider never actually produces.
"""

import numpy as np
from st_catalog import Catalog
from st_features import Camera


def random_quaternion(rng):
    """Uniformly distributed random rotation (Shoemake's method).

    Naively sampling three Euler angles is NOT uniform — it concentrates
    attitudes near the poles — and a model trained on that distribution is
    quietly better at some parts of the sky than others.
    """
    u1, u2, u3 = rng.uniform(size=3)
    s1, s2 = np.sqrt(1.0 - u1), np.sqrt(u1)
    return np.array([s1 * np.sin(2 * np.pi * u2),
                     s1 * np.cos(2 * np.pi * u2),
                     s2 * np.sin(2 * np.pi * u3),
                     s2 * np.cos(2 * np.pi * u3)])


def quat_to_matrix(q):
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix taking CAMERA-frame
    vectors to INERTIAL-frame vectors. Its transpose does the reverse."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


class NoiseModel:
    """Domain randomization parameters.

    Every field is a range that is resampled per frame, so the network sees a
    spread of conditions rather than one fixed one. Anchor these to real sensor
    data where you can: dark frames give you hot-pixel rate and read noise, the
    detector datasheet gives dark current, and a night of sky images gives a
    realistic centroid scatter.

    The field most people forget is detect_sigma. The set of stars that clears
    the detection threshold depends on where that threshold sits, and the
    histogram depends on which stars were detected. If training always uses one
    threshold and flight drifts to another, the feature shifts underneath the
    model. Vary it.
    """

    def __init__(self):
        self.centroid_sigma_px = (0.05, 0.40)   # sub-pixel centroiding error
        self.detect_sigma      = (4.0, 7.0)     # threshold in background sigma
        self.mag_limit         = (5.5, 6.5)     # effective sensitivity
        self.false_star_rate   = (0.0, 3.0)     # spurious detections per frame
        self.missed_frac       = (0.0, 0.15)    # real stars lost to noise
        self.read_noise        = (1.0, 3.0)     # counts, for render_image
        self.hot_pixel_rate    = (0.0, 2e-5)    # fraction of pixels
        self.psf_sigma_px      = (0.7, 1.6)     # defocus / blur
        self.background        = (8.0, 40.0)    # counts, stray light level

    def sample(self, rng):
        return {k: rng.uniform(*v) for k, v in vars(self).items()}


def visible_stars(cat, R_ci, cam, width, height, mag_limit):
    """Catalog stars that land inside the frame for a given attitude.

    R_ci : camera-to-inertial rotation matrix
    returns (uv pixel coords, magnitudes, camera-frame unit vectors)
    """
    keep_mag = cat.mag <= mag_limit
    v_cam = cat.vec[keep_mag] @ R_ci          # inertial -> camera is R^T v,
    mags = cat.mag[keep_mag]                  # written here as v @ R

    front = v_cam[:, 2] > 0.0
    v_cam, mags = v_cam[front], mags[front]
    if len(v_cam) == 0:
        return np.zeros((0, 2)), np.zeros(0), v_cam

    uv = cam.project(v_cam)
    inside = (np.isfinite(uv).all(axis=1) &
              (uv[:, 0] >= 0) & (uv[:, 0] < width) &
              (uv[:, 1] >= 0) & (uv[:, 1] < height))
    return uv[inside], mags[inside], v_cam[inside]


def mag_to_flux(mag, zero_point=6.0e3):
    """Visual magnitude -> relative flux.

    The magnitude scale is logarithmic and inverted: five magnitudes is a
    factor of exactly 100 in brightness, so flux ~ 10^(-0.4 * mag). Only the
    ordering matters for star selection, but the absolute scale matters for
    render_image().
    """
    return zero_point * 10.0 ** (-0.4 * np.asarray(mag, dtype=np.float64))


def simulate_frame(cat, cam, width, height, rng, noise=None, q=None):
    """One synthetic detection list. Returns dict with uv, flux, quaternion."""
    noise = noise or NoiseModel()
    p = noise.sample(rng)
    q = random_quaternion(rng) if q is None else q
    R = quat_to_matrix(q)

    uv, mags, v_cam = visible_stars(cat, R, cam, width, height, p["mag_limit"])
    flux = mag_to_flux(mags)

    # Approximate the pixel threshold using Gaussian peak intensity. Rendered
    # training uses the real C detector and is preferred for final experiments.
    peak = flux / (2.0 * np.pi * p["psf_sigma_px"] ** 2)
    detected = peak > p["detect_sigma"] * max(p["read_noise"], 0.5)
    uv, flux = uv[detected], flux[detected]

    # Stars lost to noise, occlusion, or a threshold that happened to sit just
    # above them. Dimmer stars are lost preferentially, which is what really
    # happens, so the loss probability is weighted by rank in brightness.
    if len(uv) > 0 and p["missed_frac"] > 0:
        order = np.argsort(-flux)
        rank = np.empty(len(flux)); rank[order] = np.arange(len(flux))
        pmiss = p["missed_frac"] * (rank / max(1, len(flux) - 1))
        keep = rng.uniform(size=len(flux)) > pmiss
        uv, flux = uv[keep], flux[keep]

    # False detections: hot pixels and cosmic ray hits that survive the blob
    # filters. They are uniformly placed and faint.
    n_false = rng.poisson(p["false_star_rate"])
    if n_false > 0:
        fu = rng.uniform(0, width, n_false)
        fv = rng.uniform(0, height, n_false)
        ff = mag_to_flux(rng.uniform(p["mag_limit"] - 1.0, p["mag_limit"],
                                     n_false))
        uv = np.vstack([uv, np.stack([fu, fv], axis=1)]) if len(uv) else \
             np.stack([fu, fv], axis=1)
        flux = np.concatenate([flux, ff])

    # Centroiding error.
    if len(uv) > 0 and p["centroid_sigma_px"] > 0:
        uv = uv + rng.normal(0.0, p["centroid_sigma_px"], uv.shape)

    return {"uv": uv, "flux": flux, "q": q, "R": R, "params": p}


def render_image(cat, cam, width, height, rng, noise=None, q=None):
    """Full 8-bit image, for validating the C centroider. Slow on purpose."""
    noise = noise or NoiseModel()
    p = noise.sample(rng)
    q = random_quaternion(rng) if q is None else q
    R = quat_to_matrix(q)

    uv, mags, _ = visible_stars(cat, R, cam, width, height, p["mag_limit"])
    flux = mag_to_flux(mags, zero_point=6.0e3)

    img = np.full((height, width), p["background"], dtype=np.float64)
    sig = p["psf_sigma_px"]
    half = max(2, int(np.ceil(3 * sig)))

    # Each star is a small Gaussian blob. Rendering only a local window around
    # each star instead of evaluating the PSF over the whole frame is what
    # keeps this merely slow rather than unusable.

    for (u, v), f in zip(uv, flux):
        cu, cv = int(round(u - 0.5)), int(round(v - 0.5))
        x0, x1 = max(0, cu - half), min(width, cu + half + 1)
        y0, y1 = max(0, cv - half), min(height, cv + half + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        # Evaluate the Gaussian at pixel centers using the true fractional
        # star position; do not snap the light distribution to the nearest pixel.
        yy_local, xx_local = np.mgrid[y0:y1, x0:x1]
        weights = np.exp(-((xx_local + 0.5 - u)**2 +
                           (yy_local + 0.5 - v)**2) / (2.0 * sig * sig))
        img[y0:y1, x0:x1] += f * weights / (2.0 * np.pi * sig * sig)

    img += rng.normal(0.0, p["read_noise"], img.shape)
    n_hot = int(p["hot_pixel_rate"] * width * height)
    if n_hot > 0:
        hx = rng.integers(0, width, n_hot)
        hy = rng.integers(0, height, n_hot)
        img[hy, hx] += rng.uniform(60, 255, n_hot)

    return np.clip(img, 0, 255).astype(np.uint8), q, uv, p
