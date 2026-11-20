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
