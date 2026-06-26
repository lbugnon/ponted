"""Per-residue sequence-only biophysics features (b2a).

Every channel is oriented so that **higher value = more disorder / more
linker-like**. Non-standard residues (X/U/B/Z/J/O and
gaps) are masked: they contribute neither numerator nor denominator to windowed
averages, so an ambiguous-char FASTA still gets a finite score at every position.

Construction of the 7 b2a channels (windows are literature defaults):
  seq_hydro_polar_w7      = -mean(Kyte-Doolittle hydrophobicity) over w=7
  seq_ncpr_abs_w5         = |mean(signed charge, K/R=+1 D/E=-1)| over w=5
  seq_fcr_w5              = mean(|charge|) over w=5
  seq_fpro_w5             = fraction Pro over w=5
  seq_fgly_w5             = fraction Gly over w=5
  seq_uversky_disprom_w11 = fraction disorder-promoting AAs (ARSQEGKP) over w=11
  seq_lowcomplexity_w12   = 1 - normalized Shannon entropy (log base 20) over w=12
Values are z-scored at inference with the bundled per-fold training mean/std.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

LOG = logging.getLogger("sequence_features")

_STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX = {aa: i for i, aa in enumerate(_STANDARD_AA)}

_KD_SCALE: Dict[str, float] = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}
_CHARGE: Dict[str, float] = {"K": 1.0, "R": 1.0, "D": -1.0, "E": -1.0}
_UVERSKY_DISPROM = frozenset("ARSQEGKP")


@dataclass
class SequenceCfg:
    hydro: bool = False
    ncpr_abs: bool = False
    fcr: bool = False
    fpro: bool = False
    fgly: bool = False
    uversky: bool = False
    low_complexity: bool = False

    hydro_window: int = 7
    ncpr_window: int = 5
    fcr_window: int = 5
    fpro_window: int = 5
    fgly_window: int = 5
    uversky_window: int = 11
    low_complexity_window: int = 12

    @classmethod
    def from_config(cls, cfg: Any) -> "SequenceCfg":
        """Build from a plain dict (method.yaml `sequence` block). None -> all-off.

        Accepts a `variant` shortcut (`b2a`/`compositional` -> all 7 channels at
        default windows; `none`/`off`/`false` -> empty) or explicit per-channel
        booleans + window ints.
        """
        if cfg is None:
            return cls()
        get = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))
        variant = get("variant")
        if variant is not None:
            v = str(variant).lower().strip()
            if v in {"none", "off", "false"}:
                return cls()
            if v in {"b2a", "compositional"}:
                return cls(hydro=True, ncpr_abs=True, fcr=True, fpro=True,
                           fgly=True, uversky=True, low_complexity=True)
            raise ValueError(f"unknown sequence.variant {variant!r} "
                             "(expected: b2a, compositional, none, off, false)")
        return cls(
            hydro=bool(get("hydro", False)),
            ncpr_abs=bool(get("ncpr_abs", False)),
            fcr=bool(get("fcr", False)),
            fpro=bool(get("fpro", False)),
            fgly=bool(get("fgly", False)),
            uversky=bool(get("uversky", False)),
            low_complexity=bool(get("low_complexity", False)),
            hydro_window=int(get("hydro_window", 7)),
            ncpr_window=int(get("ncpr_window", 5)),
            fcr_window=int(get("fcr_window", 5)),
            fpro_window=int(get("fpro_window", 5)),
            fgly_window=int(get("fgly_window", 5)),
            uversky_window=int(get("uversky_window", 11)),
            low_complexity_window=int(get("low_complexity_window", 12)),
        )

    @property
    def enabled(self) -> bool:
        return any([self.hydro, self.ncpr_abs, self.fcr, self.fpro,
                    self.fgly, self.uversky, self.low_complexity])


def channel_names(cfg: SequenceCfg) -> List[str]:
    names: List[str] = []
    if cfg.hydro:
        names.append(f"seq_hydro_polar_w{cfg.hydro_window}")
    if cfg.ncpr_abs:
        names.append(f"seq_ncpr_abs_w{cfg.ncpr_window}")
    if cfg.fcr:
        names.append(f"seq_fcr_w{cfg.fcr_window}")
    if cfg.fpro:
        names.append(f"seq_fpro_w{cfg.fpro_window}")
    if cfg.fgly:
        names.append(f"seq_fgly_w{cfg.fgly_window}")
    if cfg.uversky:
        names.append(f"seq_uversky_disprom_w{cfg.uversky_window}")
    if cfg.low_complexity:
        names.append(f"seq_lowcomplexity_w{cfg.low_complexity_window}")
    return names


def channel_dim(cfg: SequenceCfg) -> int:
    return len(channel_names(cfg))


def _validity_mask(sequence: str) -> np.ndarray:
    return np.fromiter((c in _AA_INDEX for c in sequence), dtype=bool, count=len(sequence))


def _per_residue_scalar(sequence: str, mapping: Dict[str, float], default: float = 0.0) -> np.ndarray:
    return np.fromiter((mapping.get(c, default) for c in sequence),
                       dtype=np.float32, count=len(sequence))


def _windowed_mean(values: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    L = values.shape[0]
    if L == 0:
        return np.zeros(0, dtype=np.float32)
    half_lo = window // 2
    half_hi = (window + 1) // 2
    cs_val = np.concatenate(([0.0], np.cumsum(values.astype(np.float64) * valid.astype(np.float64))))
    cs_cnt = np.concatenate(([0.0], np.cumsum(valid.astype(np.float64))))
    idx = np.arange(L)
    lo = np.maximum(0, idx - half_lo)
    hi = np.minimum(L, idx + half_hi)
    num = cs_val[hi] - cs_val[lo]
    den = cs_cnt[hi] - cs_cnt[lo]
    out = np.zeros(L, dtype=np.float32)
    mask = den > 0
    out[mask] = (num[mask] / den[mask]).astype(np.float32)
    return out


def _windowed_count(sequence: str, accept: frozenset, valid: np.ndarray, window: int) -> np.ndarray:
    hits = np.fromiter((1.0 if c in accept else 0.0 for c in sequence),
                       dtype=np.float32, count=len(sequence))
    return _windowed_mean(hits, valid, window)


def _wootton_federhen(sequence: str, valid: np.ndarray, window: int) -> np.ndarray:
    L = len(sequence)
    if L == 0:
        return np.zeros(0, dtype=np.float32)
    half_lo = window // 2
    half_hi = (window + 1) // 2
    onehot = np.zeros((L, 20), dtype=np.float64)
    for i, c in enumerate(sequence):
        j = _AA_INDEX.get(c)
        if j is not None:
            onehot[i, j] = 1.0
    cs_onehot = np.concatenate(([np.zeros(20)], np.cumsum(onehot, axis=0)), axis=0)
    cs_cnt = np.concatenate(([0.0], np.cumsum(valid.astype(np.float64))))
    idx = np.arange(L)
    lo = np.maximum(0, idx - half_lo)
    hi = np.minimum(L, idx + half_hi)
    counts = cs_onehot[hi] - cs_onehot[lo]
    n = cs_cnt[hi] - cs_cnt[lo]
    out = np.zeros(L, dtype=np.float32)
    log20 = math.log(20.0)
    for i in range(L):
        ni = n[i]
        if ni <= 0:
            continue
        p = counts[i] / ni
        nz = p > 0
        if not np.any(nz):
            continue
        H = -float((p[nz] * np.log(p[nz])).sum())
        out[i] = float(1.0 - H / log20)
    return out


def _raw_blocks(sequence: str, cfg: SequenceCfg) -> np.ndarray:
    L = len(sequence)
    if L == 0 or not cfg.enabled:
        return np.zeros((L, 0), dtype=np.float32)
    valid = _validity_mask(sequence)
    cols: List[np.ndarray] = []
    if cfg.hydro:
        kd = _per_residue_scalar(sequence, _KD_SCALE, default=0.0)
        cols.append(-_windowed_mean(kd, valid, cfg.hydro_window))
    if cfg.ncpr_abs:
        signed = _per_residue_scalar(sequence, _CHARGE, default=0.0)
        cols.append(np.abs(_windowed_mean(signed, valid, cfg.ncpr_window)).astype(np.float32))
    if cfg.fcr:
        absq = np.abs(_per_residue_scalar(sequence, _CHARGE, default=0.0))
        cols.append(_windowed_mean(absq, valid, cfg.fcr_window))
    if cfg.fpro:
        cols.append(_windowed_count(sequence, frozenset("P"), valid, cfg.fpro_window))
    if cfg.fgly:
        cols.append(_windowed_count(sequence, frozenset("G"), valid, cfg.fgly_window))
    if cfg.uversky:
        cols.append(_windowed_count(sequence, _UVERSKY_DISPROM, valid, cfg.uversky_window))
    if cfg.low_complexity:
        cols.append(_wootton_federhen(sequence, valid, cfg.low_complexity_window))
    return np.stack(cols, axis=1).astype(np.float32)


def build_features(sequence: str, cfg: SequenceCfg,
                   stats: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """Return (L, n_extra) float32 sequence features. With `stats`, z-scored."""
    raw = _raw_blocks(sequence, cfg)
    if raw.shape[1] == 0 or stats is None:
        return raw
    mean = stats["mean"]
    std = stats["std"]
    return ((raw - mean[None, :]) / std[None, :]).astype(np.float32)
