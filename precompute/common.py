"""Shared helpers for the host-side precompute scripts. These scripts run *outside* the container.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

from Bio import SeqIO


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    """Return [(id, sequence), ...] from a FASTA, parsed with Biopython.

    The id is the record id (first whitespace token after '>'). Sequences are
    uppercased so the (L, H) embedding aligns 1:1 with what the container's
    `caid_io.parse_fasta` produces for the same FASTA.
    """
    path = Path(path)
    out: List[Tuple[str, str]] = []
    for rec in SeqIO.parse(str(path), "fasta"):
        seq = str(rec.seq).upper()
        if not rec.id or not seq:
            raise ValueError(f"{path}: record {rec.id!r} has no id or sequence")
        out.append((rec.id, seq))
    if not out:
        raise ValueError(f"{path}: no sequences found")
    return out


def chunk_indices(length: int, chunk: int, overlap: int) -> Iterable[Tuple[int, int]]:
    """Yield (start, end) spans covering [0, length) with `overlap` between chunks."""
    if length <= chunk:
        yield 0, length
        return
    step = chunk - overlap
    if step <= 0:
        raise ValueError("chunk size must be greater than overlap")
    start = 0
    while True:
        end = min(start + chunk, length)
        yield start, end
        if end == length:
            return
        start += step
