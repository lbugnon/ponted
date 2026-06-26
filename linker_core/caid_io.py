"""FASTA input + CAID output IO.

Input: a multi-sequence FASTA, parsed with Biopython (`Bio.SeqIO`). The id is the
record id (first whitespace-delimited token after '>'). Sequences are uppercased;
ambiguous residues (B Z J U O X) are kept verbatim — downstream feature builders
mask them, so every position still gets a score (CAID rule: never crash on
ambiguous chars, always emit a score).

Output (per CAID rules):
  <out>/<id>.caid     4-column: position<TAB>residue<TAB>score<TAB>state
  <out>/timings.csv   one row per sequence: sequence,milliseconds
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from Bio import SeqIO


class FastaError(RuntimeError):
    """Raised on a malformed FASTA."""


def parse_fasta(path: Path) -> List[Tuple[str, str]]:
    """Return [(id, sequence), ...] from a FASTA, parsed with Biopython.

    The id is the record id (first whitespace-delimited token after '>').
    Sequences are uppercased; ambiguous residues are preserved.
    """
    path = Path(path)
    out: List[Tuple[str, str]] = []
    seen: set = set()
    for rec in SeqIO.parse(str(path), "fasta"):
        acc = rec.id
        seq = str(rec.seq).upper()
        if not acc:
            raise FastaError(f"{path}: record with empty identifier")
        if not seq:
            raise FastaError(f"{path}: record {acc!r} has no sequence")
        if acc in seen:
            raise FastaError(f"{path}: duplicate FASTA id {acc!r}")
        seen.add(acc)
        out.append((acc, seq))
    if not out:
        raise FastaError(f"{path}: no sequences found")
    return out


def write_caid(path: Path, fasta_id: str, sequence: str,
               scores: Sequence[float], threshold: float) -> None:
    """Write one .caid file (4-column: position, residue, score, state).

    `scores` length must equal `sequence`. The binary state is `1` where
    `score >= threshold`, else `0`.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape[0] != len(sequence):
        raise ValueError(
            f"{fasta_id}: {scores.shape[0]} scores for {len(sequence)} residues"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(f">{fasta_id}\n")
        for pos, (aa, score) in enumerate(zip(sequence, scores), start=1):
            state = 1 if score >= threshold else 0
            fh.write(f"{pos}\t{aa}\t{score:.3f}\t{state}\n")


def write_timings(path: Path, method: str, rows: Sequence[Tuple[str, int]]) -> None:
    """Write timings.csv: a comment header then `sequence,milliseconds` rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(f"# Running {method}\n")
        fh.write("sequence,milliseconds\n")
        for seq_id, ms in rows:
            fh.write(f"{seq_id},{ms}\n")
