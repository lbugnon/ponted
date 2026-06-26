"""
Embeddings are precomputed outside the container (see `precompute/`) and mounted,
keyed by the FASTA id. This module resolves an id to its (L, H) embedding matrix from a directory of per-id files:

    <dir>/<id>.npy   or   <dir>/<id>.npz[embeddings]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class EmbeddingError(RuntimeError):
    """Raised when an embedding cannot be loaded or fails validation."""


def _precompute_hint(fasta_id: str, source: Path) -> str:
    return (
        f"embedding for '{fasta_id}' not found in {source}.\n"
        f"Expected {fasta_id}.npy or {fasta_id}.npz[embeddings]. "
        "Precompute it on the host first (see precompute/ in README)."
    )


class EmbeddingSource:
    """A directory of per-id embedding files (.npy or .npz[embeddings])."""

    def __init__(self, source: Path):
        self.source = Path(source)
        if not self.source.is_dir():
            raise EmbeddingError(
                f"--embeddings must be a directory of <id>.npy/.npz files: {self.source}"
            )

    def load(self, fasta_id: str) -> np.ndarray:
        """Return (L, H) float32 embedding for `fasta_id`. Raises EmbeddingError."""
        npy = self.source / f"{fasta_id}.npy"
        npz = self.source / f"{fasta_id}.npz"
        if npy.is_file():
            arr = np.load(npy)
        elif npz.is_file():
            with np.load(npz) as fh:
                key = "embeddings" if "embeddings" in fh else list(fh.keys())[0]
                arr = fh[key]
        else:
            raise EmbeddingError(_precompute_hint(fasta_id, self.source))
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise EmbeddingError(
                f"embedding for '{fasta_id}' has shape {arr.shape}, expected 2-D (L, H)."
            )
        return arr

    def close(self) -> None:
        """No-op (kept so callers can `try/finally: source.close()` uniformly)."""


def validate_embedding(fasta_id: str, arr: np.ndarray, expected_dim: int, seq_len: int) -> None:
    """hidden dim must match the method backbone; rows must match L."""
    if arr.shape[1] != expected_dim:
        raise EmbeddingError(
            f"embedding for '{fasta_id}' has hidden dim {arr.shape[1]}, but the method "
            f"expects {expected_dim} (wrong PLM provided? check backbone.embed_dim)."
        )
    if arr.shape[0] != seq_len:
        raise EmbeddingError(
            f"embedding for '{fasta_id}' has {arr.shape[0]} rows but the FASTA sequence "
            f"is {seq_len} residues (id/sequence mismatch)."
        )
