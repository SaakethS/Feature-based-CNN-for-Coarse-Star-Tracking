"""st_catalog.py — the star catalog used to render synthetic frames.

Real use: point this at Hipparcos. Download the catalog, keep the columns
right ascension (degrees), declination (degrees), and visual magnitude, and
save them as a CSV with a header row `ra_deg,dec_deg,mag`. Then:

    cat = Catalog.load("hipparcos.csv", mag_limit=6.5)

If no CSV is present, this module synthesizes a placeholder catalog with a
realistic magnitude distribution and uniform sky positions so the whole
pipeline runs end to end today. A placeholder catalog is fine for validating
plumbing and for the Python/C cross-check. It is NOT fine for training a model
you intend to fly: the real sky is clumpy (the galactic plane above all), and a
uniform fake sky will make cells look more alike than they are.
"""

import os
import hashlib
import warnings
import numpy as np


def radec_to_vec(ra_deg, dec_deg):
    """Right ascension / declination in degrees -> unit vectors, +z to the
    north celestial pole."""
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    c = np.cos(dec)
    return np.stack([c * np.cos(ra), c * np.sin(ra), np.sin(dec)], axis=-1)


class Catalog:
    def __init__(self, vec, mag, identity=None):
        self.vec = np.asarray(vec, dtype=np.float64)   # (N, 3) unit vectors
        self.identity = identity or {"kind": "custom"}
        self.mag = np.asarray(mag, dtype=np.float64)   # (N,) visual magnitude

    def __len__(self):
        return len(self.mag)

    @staticmethod
    def load(path=None, mag_limit=6.5, seed=0):
        if path:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Catalog does not exist: {path}")
            data = np.genfromtxt(path, delimiter=",", names=True)
            data = np.atleast_1d(data)
            if not {"ra_deg", "dec_deg", "mag"}.issubset(data.dtype.names or ()):
                raise ValueError("Catalog requires ra_deg,dec_deg,mag columns")
            if not all(np.isfinite(data[k]).all() for k in ("ra_deg", "dec_deg", "mag")):
                raise ValueError("Catalog contains nonfinite values")
            if np.any(np.abs(data["dec_deg"]) > 90) or np.any((data["ra_deg"] < 0) | (data["ra_deg"] >= 360)):
                raise ValueError("Catalog coordinates are out of range")
            keep = data["mag"] <= mag_limit
            if not keep.any():
                raise ValueError("No catalog stars pass the magnitude limit")
            return Catalog(radec_to_vec(data["ra_deg"][keep],
                                        data["dec_deg"][keep]),
                           data["mag"][keep],
                           {"kind": "csv", "name": os.path.basename(path),
                            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                            "mag_limit": mag_limit})
        warnings.warn("Using synthetic placeholder sky; not real-sky validation", stacklevel=2)
        return Catalog.synthetic(mag_limit=mag_limit, seed=seed)

    @staticmethod
    def synthetic(mag_limit=6.5, seed=0):
        """Placeholder catalog: uniform on the sphere, with a star count that
        grows with limiting magnitude roughly as the real sky does.

        The empirical rule is that each extra magnitude of sensitivity reveals
        about 3x more stars, so N(<= m) ~ 10^(0.48 m). Calibrating so that
        m = 6.5 gives ~9000 naked-eye stars reproduces the right order of
        magnitude, which is what determines how many stars land in a frame and
        therefore how informative the histogram can be.
        """
        rng = np.random.default_rng(seed)
        n = int(9000 * 10 ** (0.48 * (mag_limit - 6.5)))

        # Uniform on the sphere requires uniform z, not uniform declination:
        # sampling declination uniformly would pile stars at the poles.
        z = rng.uniform(-1.0, 1.0, n)
        ra = rng.uniform(0.0, 2.0 * np.pi, n)
        rho = np.sqrt(1.0 - z * z)
        vec = np.stack([rho * np.cos(ra), rho * np.sin(ra), z], axis=1)

        # Magnitudes drawn from the same 10^(0.48 m) cumulative law.
        u = rng.uniform(0.0, 1.0, n)
        mag = mag_limit + np.log10(u) / 0.48
        return Catalog(vec, np.clip(mag, -1.5, mag_limit),
                       {"kind": "synthetic", "seed": seed, "mag_limit": mag_limit})
