"""Reading XANES 2D scans and reference spectra.

XANES 2D files (written by the XANES 2D GUI, and by pystream's DataMap):

  /exchange/data        (N, H, W)  sample images, one per energy
  /exchange/data_flat   (M, H, W)  reference (flat) images
  /exchange/flat_index  (N,)       optional: data_flat index for each image
                                   (present when references were taken only
                                   every k-th energy; otherwise M == N, 1:1)
  /exchange/energy      (N,)       energies, eV
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import h5py
import numpy as np


@dataclass
class Scan:
    path: str
    energy: np.ndarray            # (N,) eV, increasing
    od: np.ndarray                # (N, H, W) float32 flat-field corrected
                                  # absorption -ln(I/I0), each image ÷ its
                                  # own reference (computed BEFORE any shift)
    raw: np.ndarray               # (N, H, W) float32 raw intensity I (- dark)
    flat_index: Optional[np.ndarray] = None  # (N,) reference used per image
    binning: int = 1
    dark: float = 0.0
    shifts: Optional[np.ndarray] = None    # (N, 2) dy, dx applied by registration
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.energy)


def _bin2d(img: np.ndarray, b: int) -> np.ndarray:
    if b <= 1:
        return img
    h, w = (img.shape[0] // b) * b, (img.shape[1] // b) * b
    return img[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def load_scan(path: str, binning: int = 1, dark: float = 0.0,
              progress: Optional[Callable[[int, int], None]] = None,
              stop: Optional[Callable[[], bool]] = None) -> Scan:
    """Read a XANES 2D file.

    Returns both the raw images I = data - dark and the flat-field corrected
    absorption OD = -ln(I / I0), where I0 = flat - dark is each image's own
    reference (flat_index; the reference with the same index in older
    files). The correction is done here, in camera coordinates, before any
    registration: the reference is the beam profile, which does not move
    with the sample. Pixels where either value is <= 0 give NaN.
    """
    with h5py.File(path, "r") as f:
        if "exchange/data" not in f or "exchange/data_flat" not in f:
            raise ValueError(f"{os.path.basename(path)} is not a XANES 2D file "
                             "(needs /exchange/data and /exchange/data_flat)")
        data, flat = f["exchange/data"], f["exchange/data_flat"]
        energy = np.asarray(f["exchange/energy"][:], dtype=float) \
            if "exchange/energy" in f else np.arange(data.shape[0], dtype=float)
        # A stopped scan can have one more image than energies (or the
        # reverse); keep only complete energy points.
        n = min(data.shape[0], len(energy))
        if "exchange/flat_index" in f:
            fidx = np.asarray(f["exchange/flat_index"][:n], dtype=int)
            n = min(n, len(fidx))
        else:
            fidx = np.minimum(np.arange(n), flat.shape[0] - 1)
        if n == 0:
            raise ValueError("the file has no complete energy points")

        info = {k: (v.decode() if isinstance(v, bytes) else v)
                for k, v in f.attrs.items() if k != "scan_config"}
        info["n_flats"] = int(flat.shape[0])

        flats = {}                       # cache: a reference may serve many images
        out = raw = None
        for i in range(n):
            if stop and stop():
                raise InterruptedError
            j = int(fidx[i])
            if j not in flats:
                flats[j] = _bin2d(flat[j].astype(np.float32) - dark, binning)
            d = _bin2d(data[i].astype(np.float32) - dark, binning)
            with np.errstate(divide="ignore", invalid="ignore"):
                od = -np.log(d / flats[j])
            od[~np.isfinite(od)] = np.nan
            if out is None:
                out = np.empty((n,) + od.shape, dtype=np.float32)
                raw = np.empty((n,) + od.shape, dtype=np.float32)
            out[i] = od
            raw[i] = d
            if progress:
                progress(i + 1, n)

    order = np.argsort(energy[:n], kind="stable")
    fidx = np.asarray(fidx[:n])
    if np.any(order != np.arange(n)):
        out, raw, energy, fidx = out[order], raw[order], energy[:n][order], fidx[order]
    return Scan(path=path, energy=energy[:n].copy(), od=out, raw=raw,
                flat_index=fidx, binning=binning, dark=dark, info=info)


def save_aligned(scan: Scan, path: str) -> None:
    """Write the (aligned) stacks and energies to HDF5: data_od (flat-field
    corrected absorption) and data_raw (raw intensity, same shifts)."""
    with h5py.File(path, "w") as f:
        for name, arr in (("data_od", scan.od), ("data_raw", scan.raw)):
            f.create_dataset(f"exchange/{name}", data=arr,
                             chunks=(1,) + arr.shape[1:], compression="gzip",
                             compression_opts=3)
        f["exchange/data_od"].attrs["content"] = "-ln(I/I0), each image ÷ its own reference"
        f["exchange/data_raw"].attrs["content"] = "raw intensity I (minus dark), not flat-field corrected"
        f["exchange/energy"] = scan.energy
        f["exchange/energy"].attrs["units"] = "eV"
        if scan.shifts is not None:
            f["registration/shifts"] = scan.shifts
            f["registration/shifts"].attrs["columns"] = "dy, dx (pixels, binned)"
        f.attrs["source"] = scan.path
        f.attrs["binning"] = scan.binning
        f.attrs["dark"] = scan.dark
        f.attrs["content"] = "XANES 2D stacks aligned by xanes2d-analysis"


def load_reference(path: str):
    """Reference spectrum from a 2-column file (energy, intensity).

    Accepts real .npy files and text files (space/comma/tab separated, with
    or without header lines — the XANES GUI 'Calibrated' curves are text
    files with a .npy name). Energies below 100 are taken as keV and
    converted to eV. XDI files (e.g. from the IXAS XAS Data Library) are
    read with their column names, see refs.read_xdi.
    Returns (energy_eV, mu) sorted by energy.
    """
    from . import refs
    if refs.is_xdi(path):
        d = refs.read_xdi(path)
        return d["energy"], d["mu"]
    arr = None
    if path.endswith(".npy"):
        try:
            arr = np.load(path, allow_pickle=False)
        except (ValueError, OSError):
            arr = None
    if arr is None:
        rows = []
        with open(path) as fh:
            for line in fh:
                parts = line.replace(",", " ").replace(";", " ").split()
                try:
                    vals = [float(p) for p in parts]
                except ValueError:
                    continue                    # header / comment line
                if len(vals) >= 2:
                    rows.append(vals[:2])
        arr = np.asarray(rows, dtype=float)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or 2 not in arr.shape:
        raise ValueError(f"expected 2 columns (energy, intensity), got shape {arr.shape}")
    if arr.shape[1] != 2 and arr.shape[0] == 2:
        arr = arr.T
    e, mu = arr[:, 0], arr[:, 1]
    if np.nanmax(e) < 100:                      # keV
        e = e * 1000.0
    order = np.argsort(e)
    return e[order], mu[order]
