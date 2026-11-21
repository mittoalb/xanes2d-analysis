"""Sample / background segmentation and per-energy background removal.

The sample is found in the mean of the post-edge images (highest contrast:
the element absorbs most above its edge). Background removal subtracts, at
every energy, the median absorption of the background pixels, so empty
areas become ~0. That also corrects intensity changes of the beam between
each image and its reference (they show up as an offset common to all
pixels of that energy).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import ndimage


def contrast_image(e: np.ndarray, stack: np.ndarray, post: Sequence[float]) -> np.ndarray:
    """Mean of the images whose energy lies in the post-edge range (the last
    20% of the energies if the range holds none)."""
    m = (e >= post[0]) & (e <= post[1])
    if not m.any():
        m = np.zeros(len(e), bool)
        m[-max(1, len(e) // 5):] = True
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack[m], axis=0)


def otsu_threshold(img: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold of the finite values of img."""
    v = img[np.isfinite(img)]
    if v.size == 0:
        return float("nan")
    lo, hi = np.percentile(v, (0.5, 99.5))
    if hi <= lo:
        return float(lo)
    hist, edges = np.histogram(np.clip(v, lo, hi), bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * centers)
    m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    between = (w0 * w1 * (m0 - m1) ** 2)[:-1]
    # Across an empty gap between two well-separated populations the score
    # is flat (every cut splits the pixels the same way); take the middle of
    # that plateau, not its first bin (which would hug one population).
    top = np.flatnonzero(between >= between.max() * (1 - 1e-9))
    return float(0.5 * (centers[top[0]] + centers[top[-1]]))


def segment_sample(img: np.ndarray, threshold: float, smooth: float = 2.0,
                   min_size_frac: float = 0.001) -> np.ndarray:
    """Boolean sample mask: smoothed image above threshold, with small specks
    removed (smaller than min_size_frac of the image) and holes filled."""
    a = np.array(img, dtype=np.float64)
    bad = ~np.isfinite(a)
    if bad.any():
        a[bad] = np.nanmedian(a) if (~bad).any() else 0.0
    if smooth > 0:
        a = ndimage.gaussian_filter(a, smooth)
    mask = a > threshold
    mask = ndimage.binary_opening(mask, iterations=1)
    lab, n = ndimage.label(mask)
    if n:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= max(1, min_size_frac * mask.size)
        keep[0] = False
        mask = keep[lab]
    return ndimage.binary_fill_holes(mask)


def background_pixels(mask: np.ndarray, margin: int = 3) -> np.ndarray:
    """Background = not sample, keeping `margin` px away from the sample edge
    (blurred edges would bias the level)."""
    grown = ndimage.binary_dilation(mask, iterations=margin) if margin > 0 else mask
    return ~grown


def background_levels(stack: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Median of every image over the background pixels (N,). NaN if there
    is no background."""
    if not bg.any():
        return np.full(stack.shape[0], np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmedian(stack[:, bg], axis=1)
