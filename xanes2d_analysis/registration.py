"""2D image registration for XANES stacks (sub-pixel phase correlation).

Phase correlation with upsampled-DFT refinement (Guizar-Sicairos et al.,
Opt. Lett. 33, 156 (2008)) — the same method as scikit-image's
phase_cross_correlation, implemented with NumPy only.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from scipy import ndimage


def _upsampled_dft(data, region, factor, offsets):
    """Upsampled DFT of `data` on a `region`×`region` grid around `offsets`."""
    im2pi = 2j * np.pi
    for n_items, ax_offset in list(zip(data.shape, offsets))[::-1]:
        kernel = ((np.arange(region) - ax_offset)[:, None]
                  * np.fft.fftfreq(n_items, factor))
        data = np.tensordot(np.exp(-im2pi * kernel), data, axes=(1, -1))
    return data


def _prepare(img: np.ndarray) -> np.ndarray:
    """NaN-free, zero-mean, Hann-windowed copy (limits edge artefacts)."""
    a = np.array(img, dtype=np.float64)
    bad = ~np.isfinite(a)
    if bad.any():
        a[bad] = np.nanmean(a) if (~bad).any() else 0.0
    a -= a.mean()
    wy = np.hanning(a.shape[0])[:, None]
    wx = np.hanning(a.shape[1])[None, :]
    return a * wy * wx


def _band_weight(shape, smooth: float) -> np.ndarray:
    """Frequency weight for the phase-normalized cross-power spectrum.

    Pure phase correlation weighs all frequencies equally, so pixel noise can
    outvote real structure (pre-edge frames have little contrast). Filtering
    the images first does not help — the normalization undoes it — so the
    band-pass is applied here instead: a Gaussian low-pass (like blurring
    the images by `smooth` px) and a gentle high-pass that ignores slow
    background (beam profile, overall absorption change across the edge)."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    f2 = fy ** 2 + fx ** 2
    low = np.exp(-2 * (np.pi * smooth) ** 2 * f2)
    f_hp = 1.0 / max(20.0 * smooth, 8.0)
    high = 1.0 - np.exp(-f2 / (2 * f_hp ** 2))
    return low * high


def phase_correlation(reference: np.ndarray, moving: np.ndarray,
                      upsample: int = 20, smooth: float = 1.5) -> np.ndarray:
    """Shift (dy, dx) that, applied to `moving` (scipy.ndimage.shift),
    registers it with `reference`. Precision 1/upsample pixel; `smooth`
    (px) sets the noise filtering (0 = plain phase correlation)."""
    ref_f = np.fft.fft2(_prepare(reference))
    mov_f = np.fft.fft2(_prepare(moving))
    cross = ref_f * mov_f.conj()
    cross /= np.abs(cross) + 1e-12                       # phase only
    if smooth > 0:
        cross *= _band_weight(cross.shape, smooth)
    corr = np.fft.ifft2(cross)
    shape = np.array(corr.shape)
    peak = np.array(np.unravel_index(np.argmax(np.abs(corr)), corr.shape), float)
    mid = np.fix(shape / 2)
    peak[peak > mid] -= shape[peak > mid]
    if upsample > 1:
        peak = np.round(peak * upsample) / upsample
        region = int(np.ceil(upsample * 1.5))
        dftshift = np.fix(region / 2.0)
        offsets = dftshift - peak * upsample
        up = _upsampled_dft(cross.conj(), region, upsample, offsets).conj()
        fine = np.array(np.unravel_index(np.argmax(np.abs(up)), up.shape), float)
        peak = peak + (fine - dftshift) / upsample
    return peak


def register_stack(stack: np.ndarray, ref_index: int,
                   region: Optional[Sequence[int]] = None, upsample: int = 20,
                   smooth: float = 1.5, mode: str = "reference",
                   progress: Optional[Callable[[int, int], None]] = None,
                   stop: Optional[Callable[[], bool]] = None) -> np.ndarray:
    """Shifts (N, 2) that register every frame to frame `ref_index`.

    mode "reference": each frame is compared directly with the reference
    frame. mode "neighbour": each frame is compared with the next one
    towards the reference and the shifts are added up — for scans where the
    image changes a lot across the edge (neighbouring energies look alike),
    at the cost of small errors accumulating.

    `region` = (y0, y1, x0, x1) restricts the comparison to that part of the
    image (faster, and avoids areas that change with energy or have no
    structure)."""
    n = stack.shape[0]
    sl = (slice(None), slice(None)) if region is None else \
        (slice(region[0], region[1]), slice(region[2], region[3]))
    shifts = np.zeros((n, 2))
    done = 0
    if mode == "neighbour":
        # walk outwards from the reference in both directions
        for order in (range(ref_index - 1, -1, -1), range(ref_index + 1, n)):
            for i in order:
                if stop and stop():
                    raise InterruptedError
                j = i + 1 if i < ref_index else i - 1        # neighbour towards ref
                shifts[i] = shifts[j] + phase_correlation(
                    stack[j][sl], stack[i][sl], upsample, smooth)
                done += 1
                if progress:
                    progress(done, n - 1)
        return shifts
    ref = stack[ref_index][sl]
    for i in range(n):
        if stop and stop():
            raise InterruptedError
        if i != ref_index:
            shifts[i] = phase_correlation(ref, stack[i][sl], upsample, smooth)
        if progress:
            progress(i + 1, n)
    return shifts


def apply_shifts(stack: np.ndarray, shifts: np.ndarray,
                 progress: Optional[Callable[[int, int], None]] = None) -> None:
    """Shift every frame in place (linear interpolation; edges repeat)."""
    n = stack.shape[0]
    for i in range(n):
        dy, dx = shifts[i]
        if dy or dx:
            frame = stack[i]
            bad = ~np.isfinite(frame)
            if bad.any():
                frame = frame.copy()
                frame[bad] = np.nanmedian(frame)
            stack[i] = ndimage.shift(frame, (dy, dx), order=1, mode="nearest")
        if progress:
            progress(i + 1, n)
