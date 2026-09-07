"""st_cells.py — partition the celestial sphere into exactly NUM_CELLS cells.

Why this file exists
--------------------
The classifier's 529 outputs are only meaningful if "cell 314" means the same
patch of sky during training and in orbit. This module is the definition of
that mapping. It is ground-side only: the C code never needs to know the cell
geometry, it only ever emits cell indices for the pattern matcher to consume.

Partition scheme
----------------
Equal-area zonal partition:

  1. Each cell should have area 4*pi / N steradians, so a roughly square cell
     is about theta = sqrt(4*pi/N) radians on a side.
  2. Split the sphere into rings of constant declination, spaced so each ring
     is theta tall. Two polar caps are single cells.
  3. Each ring's area is 2*pi*(sin(dec_hi) - sin(dec_lo)); give it a number of
     cells proportional to that area, then nudge the counts until the total is
     exactly N.

This produces cells of nearly equal area, which matters because a classifier
trained on wildly unequal cells learns the cell size distribution rather than
the sky.

IMPORTANT: if the GLAS document your professor is working from specifies a
different 529-cell convention (for example HEALPix nside=... or a named
tessellation), replace this file with that convention. The rest of the
pipeline does not care how cells are defined, only that the definition is
fixed. Confirm this with Dr. Lee before generating a full training set.
"""

import os
import numpy as np

# The partition size is a design parameter, not a constant of nature: it must
# be chosen together with the field of view so that a frame covers only a few
# cells. Set ST_NUM_CELLS in the environment to sweep it; the default matches
# the architecture document.
NUM_CELLS = int(os.environ.get("ST_NUM_CELLS", "529"))


def _build_partition(n_cells=NUM_CELLS):
    """Return (ring_edges_sin, ring_counts, ring_start_index).

    ring_edges_sin : sin(declination) boundaries, length n_rings+1, ascending
                     from -1 (south pole) to +1 (north pole)
    ring_counts    : number of cells in each ring
    ring_start     : index of the first cell of each ring
    """
    ideal_cell_area = 4.0 * np.pi / n_cells
    theta = np.sqrt(ideal_cell_area)          # cell side length, radians
    n_rings = max(3, int(round(np.pi / theta)))

    # Declination edges evenly spaced in angle, converted to sin for area math.
    dec_edges = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_rings + 1)
    sin_edges = np.sin(dec_edges)

    ring_area = 2.0 * np.pi * np.diff(sin_edges)          # steradians per ring
    raw = ring_area / ideal_cell_area                     # ideal cell count
    counts = np.maximum(1, np.round(raw)).astype(int)

    # Fix the total to land on exactly n_cells. We add or remove cells from
    # whichever ring currently has the worst cells-per-area error, so the
    # correction is spread sensibly instead of dumped on one ring.
    while counts.sum() != n_cells:
        err = counts - raw
        if counts.sum() > n_cells:
            i = int(np.argmax(np.where(counts > 1, err, -np.inf)))
            counts[i] -= 1
        else:
            i = int(np.argmin(err))
            counts[i] += 1

    ring_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return sin_edges, counts, ring_start


SIN_EDGES, RING_COUNTS, RING_START = _build_partition()
N_RINGS = len(RING_COUNTS)
assert RING_COUNTS.sum() == NUM_CELLS


def vectors_to_cells(v):
    """Map unit vectors in the INERTIAL frame to cell indices.

    v : (..., 3) array of unit vectors, with +z toward the north celestial pole
    returns : integer array of the same leading shape
    """
    v = np.asarray(v, dtype=np.float64)
    z = np.clip(v[..., 2], -1.0, 1.0)                 # = sin(declination)
    ra = np.arctan2(v[..., 1], v[..., 0]) % (2.0 * np.pi)

    # searchsorted gives the ring whose sin-declination band contains z.
    ring = np.clip(np.searchsorted(SIN_EDGES, z, side="right") - 1,
                   0, N_RINGS - 1)
    counts = RING_COUNTS[ring]
    col = np.minimum((ra / (2.0 * np.pi) * counts).astype(int), counts - 1)
    return RING_START[ring] + col


def cell_centers():
    """Unit vector at the center of every cell. Shape (NUM_CELLS, 3).

    Useful for sanity plots and for computing which cells neighbour which.
    """
    out = np.zeros((NUM_CELLS, 3))
    for r in range(N_RINGS):
        z = 0.5 * (SIN_EDGES[r] + SIN_EDGES[r + 1])
        rho = np.sqrt(max(0.0, 1.0 - z * z))
        n = RING_COUNTS[r]
        for c in range(n):
            ra = 2.0 * np.pi * (c + 0.5) / n
            out[RING_START[r] + c] = (rho * np.cos(ra), rho * np.sin(ra), z)
    return out


if __name__ == "__main__":
    areas = []
    for r in range(N_RINGS):
        a = 2 * np.pi * (SIN_EDGES[r + 1] - SIN_EDGES[r]) / RING_COUNTS[r]
        areas.extend([a] * RING_COUNTS[r])
    areas = np.array(areas)
    ideal = 4 * np.pi / NUM_CELLS
    print(f"{NUM_CELLS} cells in {N_RINGS} rings")
    print(f"ideal cell area {ideal:.6f} sr "
          f"(equivalent to a {np.degrees(np.sqrt(ideal)):.2f} deg square)")
    print(f"actual area min {areas.min():.6f} max {areas.max():.6f} "
          f"ratio {areas.max()/areas.min():.3f}")
