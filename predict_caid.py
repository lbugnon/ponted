#!/usr/bin/env python3
"""CAID linker predictor.

Reads a multi-sequence FASTA, runs a prediction method (a 5-fold ensemble), and writes:

    <out>/<id>.caid      4-column CAID predictions
    <out>/timings.csv    per-sequence wall time (ms)

Runtime inputs are precomputed (see precompute/): One folder per feature type,
one `<fasta_id>.npy` per sequence inside it:

    PonTED       --embeddings esm2_folder
    PonTED-XL    --embeddings prott5_folder
    Ponte-S      --embeddings esm2_folder --af2-plddt af2_folder

The af2 (1) + b2a (7) channels are appended after the PLM block; af2 = 1-pLDDT
from --af2-plddt, b2a = 7 sequence-biophysics channels computed in-container. All
embeddings are (L, H) aligned 1:1 with the sequence (CLS/EOS already stripped).

Example:
    python predict_caid.py \
        --method PonTED \
        --fasta input.fasta \
        --embeddings /mnt/embeddings \
        --out predictions \
        --threads 8
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

LOG = logging.getLogger("predict_caid")

_HERE = Path(__file__).resolve().parent


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True,
                   help="Method (model) name; resolved under --methods-dir.")
    p.add_argument("--fasta", type=Path, required=True, help="Input multi-sequence FASTA.")
    p.add_argument("--out", type=Path, required=True, help="Output directory.")
    p.add_argument("--embeddings", type=Path, required=True,
                   help="Directory of precomputed pLM embeddings, one <id>.npy/.npz per sequence.")
    p.add_argument("--af2-plddt", type=Path, default=None,
                   help="Directory of precomputed AF2 pLDDT <id>.npy (required for af2 methods).")
    p.add_argument("--methods-dir", type=Path, default=_HERE / "methods",
                   help="Directory containing one subdir per served method.")
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1),
                   help="CPU threads for torch (CAID cap is 24).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # CPU-only, thread-capped. Force before importing heavy numerics elsewhere.
    import torch
    torch.set_num_threads(max(1, args.threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, args.threads)))

    from linker_core.caid_io import parse_fasta, write_caid, write_timings
    from linker_core.embeddings_io import EmbeddingSource, validate_embedding
    from linker_core.methods import Method

    method_dir = args.methods_dir / args.method
    if not (method_dir / "method.yaml").is_file():
        raise SystemExit(f"method {args.method!r} not found under {args.methods_dir}")
    method = Method.load(method_dir)

    if method.needs_af2 and args.af2_plddt is None:
        raise SystemExit(
            f"method {method.name!r} requires AF2 but --af2-plddt not given.")

    records = parse_fasta(args.fasta)
    LOG.info("parsed %d sequences from %s", len(records), args.fasta)

    args.out.mkdir(parents=True, exist_ok=True)
    emb_source = EmbeddingSource(args.embeddings)
    timings = []
    for fasta_id, seq in records:
        t0 = time.perf_counter()
        emb = emb_source.load(fasta_id)
        validate_embedding(fasta_id, emb, method.backbone_dim, len(seq))
        scores = method.score(fasta_id, seq, emb, args.af2_plddt)
        write_caid(args.out / f"{fasta_id}.caid", fasta_id, seq, scores, method.binary_threshold)
        ms = int(round((time.perf_counter() - t0) * 1000))
        timings.append((fasta_id, ms))
        LOG.debug("%s/%s L=%d mean=%.3f %dms", method.name, fasta_id, len(seq),
                  float(scores.mean()), ms)

    write_timings(args.out / "timings.csv", method.name, timings)
    LOG.info("method %s: wrote %d predictions -> %s", method.name, len(records), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
