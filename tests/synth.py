"""Synthetic XANES 2D file in the real layout, for tests and demos.

Two phases: A (edge 8333 eV, left disc) and B (edge 8345 eV, right disc) on
a textured background, a non-uniform beam (flat), references every
`ref_every` energies (flat_index), and a known drift per energy.
"""

import h5py
import numpy as np
from scipy import ndimage


def edge(e, e0, width=2.0, step=1.0, white=0.4):
    """Smooth absorption edge with a small white line."""
    s = 0.5 * (1 + np.tanh((e - e0) / width))
    return step * s + white * np.exp(-((e - e0 - 3) / 3.0) ** 2)


def make_file(path, n_e=31, shape=(120, 160), ref_every=3, drift_px=3.0, seed=1):
    rng = np.random.default_rng(seed)
    energies = np.linspace(8310, 8370, n_e)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    disc_a = ((yy - h * 0.5) ** 2 + (xx - w * 0.3) ** 2) < (h * 0.22) ** 2
    disc_b = ((yy - h * 0.5) ** 2 + (xx - w * 0.7) ** 2) < (h * 0.22) ** 2
    texture = ndimage.gaussian_filter(rng.random(shape), 3) * 0.6   # structure to register on
    beam = 20000 * (0.7 + 0.3 * np.exp(-((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (2 * (w / 2) ** 2)))

    drifts = rng.uniform(-drift_px, drift_px, size=(n_e, 2))
    drifts[n_e - 1] = 0.0                                            # last frame = reference
    data, flats, fidx, e_flat = [], [], [], []
    for i, e in enumerate(energies):
        # particles absorb below their edge too (non-resonant), plus the jump
        od = (0.2 + texture + disc_a * (0.5 + 0.8 * edge(e, 8333))
              + disc_b * (0.4 + 0.8 * edge(e, 8345)))
        od = ndimage.shift(od, drifts[i], order=1, mode="nearest")   # sample drift
        if i % ref_every == 0 or i == n_e - 1:
            flats.append(beam * (1 + 0.005 * rng.standard_normal(shape)))
            e_flat.append(e)
        fidx.append(len(flats) - 1)
        data.append(rng.poisson(beam * np.exp(-od)).astype(np.uint16))
    with h5py.File(path, "w") as f:
        f["exchange/data"] = np.array(data)
        f["exchange/data_flat"] = np.array(flats).astype(np.uint16)
        f["exchange/flat_index"] = np.array(fidx, dtype="i4")
        f["exchange/energy_flat"] = np.array(e_flat)
        f["exchange/energy"] = energies
        f.attrs["format"] = "xanes_gui.gui_2d v1"
    return energies, drifts, disc_a, disc_b
