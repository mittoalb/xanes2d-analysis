"""XANES spectrum helpers: edge-step normalization, comparison, export."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def edge_energy(e: np.ndarray, mu: np.ndarray) -> float:
    """E0 = energy of the steepest rise (maximum of the smoothed derivative)."""
    ok = np.isfinite(mu)
    e, mu = e[ok], mu[ok]
    if len(e) < 3:
        return float(e[len(e) // 2]) if len(e) else float("nan")
    k = min(5, max(1, len(mu) // 20))
    smooth = np.convolve(mu, np.ones(k) / k, mode="same")
    d = np.gradient(smooth, e)
    # ignore the ends, where the moving average is biased
    lo, hi = k, len(d) - k
    i = lo + int(np.argmax(d[lo:hi])) if hi > lo else int(np.argmax(d))
    return float(e[i])


def _linfit(e, mu, rng):
    m = (e >= rng[0]) & (e <= rng[1]) & np.isfinite(mu)
    if m.sum() < 2:
        return None
    return np.polyfit(e[m], mu[m], 1)


def normalize(e: np.ndarray, mu: np.ndarray, pre: Sequence[float],
              post: Sequence[float], e0: Optional[float] = None
              ) -> Tuple[np.ndarray, float, float]:
    """Standard XANES normalization.

    A line fitted in the pre-edge range is subtracted; the result is divided
    by the edge step = (post-edge line − pre-edge line) at E0. Returns
    (normalized mu, E0, edge step). If a range has fewer than 2 points the
    spectrum is returned unchanged (step = 1)."""
    e = np.asarray(e, float)
    mu = np.asarray(mu, float)
    if e0 is None:
        e0 = edge_energy(e, mu)
    p_pre, p_post = _linfit(e, mu, pre), _linfit(e, mu, post)
    if p_pre is None or p_post is None:
        return mu.copy(), e0, 1.0
    step = np.polyval(p_post, e0) - np.polyval(p_pre, e0)
    if not np.isfinite(step) or abs(step) < 1e-12:
        step = 1.0
    return (mu - np.polyval(p_pre, e)) / step, e0, float(step)


def r_factor(e: np.ndarray, mu: np.ndarray, e_ref: np.ndarray,
             mu_ref: np.ndarray) -> float:
    """sum((mu - ref)^2) / sum(mu^2) with the reference interpolated onto e
    (only where both are defined)."""
    ref = np.interp(e, e_ref, mu_ref, left=np.nan, right=np.nan)
    m = np.isfinite(ref) & np.isfinite(mu)
    if m.sum() < 2:
        return float("nan")
    den = np.sum(mu[m] ** 2)
    return float(np.sum((mu[m] - ref[m]) ** 2) / den) if den else float("nan")


def write_csv(path: str, energy: np.ndarray, columns: dict,
              header_lines: Sequence[str] = ()) -> None:
    """Energy plus one column per entry of `columns` ({name: values})."""
    names = ["energy_eV"] + list(columns)
    data = np.column_stack([energy] + [np.asarray(v, float) for v in columns.values()])
    with open(path, "w") as fh:
        for line in header_lines:
            fh.write(f"# {line}\n")
        fh.write(",".join(names) + "\n")
        for row in data:
            fh.write(",".join(f"{v:.6g}" for v in row) + "\n")


def _line_fit_all(e: np.ndarray, y: np.ndarray, rng) -> Optional[np.ndarray]:
    """Least-squares line a + b·E for every column of y (N, P) using only
    energies inside rng. Returns (2, P) [a, b] or None if < 2 energies."""
    m = (e >= rng[0]) & (e <= rng[1])
    if m.sum() < 2:
        return None
    A = np.column_stack([np.ones(m.sum()), e[m]])
    yy = np.where(np.isfinite(y[m]), y[m], 0.0)
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    return coef


def edge_maps(e: np.ndarray, stack: np.ndarray, pre: Sequence[float],
              post: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel edge step and edge energy for a (N, H, W) absorption stack.

    Every pixel's spectrum is normalised like `normalize` (pre-edge line
    subtracted, divided by the step between the post- and pre-edge lines,
    taken in the middle of the gap between the two ranges). The edge energy
    is where the normalised spectrum (3-point smoothed along energy) first
    crosses 0.5 between the end of the pre-edge range and the start of the
    post-edge range, linearly interpolated. Pixels without a crossing get
    NaN. Returns (step (H, W), edge energy (H, W) in eV)."""
    e = np.asarray(e, float)
    n, h, w = stack.shape
    y = stack.reshape(n, -1).astype(np.float64)
    c_pre, c_post = _line_fit_all(e, y, pre), _line_fit_all(e, y, post)
    if c_pre is None or c_post is None:
        nan = np.full((h, w), np.nan)
        return nan, nan
    e_mid = 0.5 * (pre[1] + post[0])
    step = (c_post[0] + c_post[1] * e_mid) - (c_pre[0] + c_pre[1] * e_mid)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = (y - (c_pre[0] + c_pre[1] * e[:, None])) / step
    if n >= 3:                                     # light smoothing along energy
        sm = norm.copy()
        sm[1:-1] = (norm[:-2] + norm[1:-1] + norm[2:]) / 3.0
        norm = sm
    win = np.where((e >= pre[1]) & (e <= post[0]))[0]
    e_half = np.full(h * w, np.nan)
    if len(win) >= 2:
        lo, hi = win[0], win[-1]
        seg = norm[lo:hi + 1]
        above = seg >= 0.5
        has = above.any(axis=0) & ~above[0]        # starts below, crosses later
        i = np.argmax(above, axis=0)               # first index at/above 0.5
        cols = np.where(has)[0]
        i1 = i[cols]
        i0 = i1 - 1
        y0, y1 = seg[i0, cols], seg[i1, cols]
        ee = e[lo:hi + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.clip((0.5 - y0) / (y1 - y0), 0, 1)
        e_half[cols] = ee[i0] + frac * (ee[i1] - ee[i0])
    return step.reshape(h, w), e_half.reshape(h, w)
