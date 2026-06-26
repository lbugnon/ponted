"""Inference-time score smoothing (anti-fragmentation post-process).

A per-protein median filter on the final ensemble score eliminates isolated
peaks
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def median_filter_1d(x: np.ndarray, window: int) -> np.ndarray:
    """1-D median filter, odd `window`, edge-extended ('nearest') at the bounds."""
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or x.size == 0:
        return x
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(xp, window)
    return np.median(windows, axis=-1)


def smooth(scores: np.ndarray, method: Optional[str], window: int) -> np.ndarray:
    """Apply the configured smoothing to one protein's score track."""
    if not method or method == "none" or window <= 1:
        return np.asarray(scores, dtype=np.float64)
    if method == "median":
        return median_filter_1d(scores, window)
    raise ValueError(f"unsupported smoothing method {method!r} (inference supports 'median')")
