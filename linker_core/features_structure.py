"""Per-residue structure feature for inference: AF2 disorder proxy.

Channel:  af2_disorder_proxy = 1 - pLDDT      (higher = more disordered)

Expected per-protein AF2 file, keyed by the FASTA id (organizers precompute it):
    <af2_dir>/<id>.npy           pLDDT array, shape (L,), values in [0, 1]
  or
    <af2_dir>/<id>.npz           with key "plddt", same shape/range

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

AF2_CHANNEL = "af2_disorder_proxy"


class FeatureError(RuntimeError):
    """Raised when a required feature is missing or malformed."""


def _af2_precompute_hint(fasta_id: str, af2_dir: Path) -> str:
    return (
        f"AF2 pLDDT for '{fasta_id}' not found under {af2_dir} "
        f"(expected {fasta_id}.npy or {fasta_id}.npz[plddt], shape (L,), values in [0,1]).\n"
        "Provide it via --af2-plddt (precompute it; see README 'Stage 1')."
    )


def _load_plddt(fasta_id: str, af2_dir: Path) -> np.ndarray:
    """Return pLDDT (L,) in [0,1] for `fasta_id`. Raises if the file is absent."""
    npy = af2_dir / f"{fasta_id}.npy"
    npz = af2_dir / f"{fasta_id}.npz"
    if npy.is_file():
        plddt = np.load(npy).astype(np.float32)
    elif npz.is_file():
        with np.load(npz) as fh:
            if "plddt" not in fh:
                raise FeatureError(f"{npz}: missing 'plddt' key (has {list(fh.keys())})")
            plddt = fh["plddt"].astype(np.float32)
    else:
        raise FeatureError(_af2_precompute_hint(fasta_id, af2_dir))
    plddt = np.asarray(plddt).reshape(-1)
    if plddt.size and float(np.nanmax(plddt)) > 1.5:
        raise FeatureError(
            f"AF2 pLDDT for '{fasta_id}' looks like a 0-100 scale (max={float(np.nanmax(plddt)):.1f}); "
            "expected [0,1]. Divide by 100 when precomputing."
        )
    return plddt


def build_structure_features(
    fasta_id: str,
    length: int,
    structure_channels: List[str],
    stats: Optional[Dict[str, np.ndarray]],
    af2_dir: Optional[Path],
) -> np.ndarray:
    """Return (L, n_struct) z-scored structure features for one protein.

    `structure_channels` is the method's exact channel list (from method.yaml);
    must be `[]` or `["af2_disorder_proxy"]`. Raises FeatureError on an
    unsupported channel, a missing AF2 file, a length mismatch, or an
    out-of-range pLDDT.
    """
    if not structure_channels:
        return np.zeros((length, 0), dtype=np.float32)
    unsupported = [c for c in structure_channels if c != AF2_CHANNEL]
    if unsupported:
        raise FeatureError(
            f"unsupported structure channel(s) {unsupported} for inference "
            f"(only {AF2_CHANNEL!r} is available; Boltz/domain are training-only)."
        )
    if af2_dir is None:
        raise FeatureError(
            f"method needs the AF2 channel but --af2-plddt was not provided.\n"
            + _af2_precompute_hint(fasta_id, Path("<af2-plddt>"))
        )
    af2_dir = Path(af2_dir)

    plddt = _load_plddt(fasta_id, af2_dir)
    if plddt.shape[0] != length:
        raise FeatureError(
            f"AF2 length {plddt.shape[0]} != sequence length {length} for '{fasta_id}' "
            "(id/sequence mismatch — the precomputed structure is for a different sequence)."
        )
    raw = (1.0 - plddt).astype(np.float32).reshape(length, 1)

    if stats is None:
        raise FeatureError(
            f"method uses structure channels but no structure_stats were loaded for '{fasta_id}'."
        )
    mean = stats["mean"]
    std = stats["std"]
    if mean.shape[0] != 1 or std.shape[0] != 1:
        raise FeatureError(
            f"structure_stats has {mean.shape[0]} channels, expected 1 (af2)."
        )
    return ((raw - mean[None, :]) / std[None, :]).astype(np.float32)
