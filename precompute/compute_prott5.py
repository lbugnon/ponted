#!/usr/bin/env python3
"""ProtT5-XL encoder (Rostlab/prot_t5_xl_half_uniref50-enc) embeddings. Host-side.

Usage:
  pip install -r precompute/requirements-precompute.txt
  python precompute/compute_prott5.py --fasta input.fasta --output-dir emb_prott5/
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import chunk_indices, read_fasta  # noqa: E402

LOG = logging.getLogger("precompute_prott5")
MODEL_ID = "Rostlab/prot_t5_xl_half_uniref50-enc"
_RARE_AA = re.compile(r"[UZOB]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", type=Path, required=True, help="Input FASTA (2-line or CAID 3-line).")
    p.add_argument("--output-dir", type=Path, required=True, help="Where <id>.npy files are written.")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--chunk-size", type=int, default=4000,
                   help="Max residues per forward (ProtT5 tolerates long inputs; bounds memory).")
    p.add_argument("--overlap", type=int, default=256, help="Residue overlap between chunks.")
    p.add_argument("--batch-size", type=int, default=1, help="Chunks per forward.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def embed_one(seq: str, model, tokenizer, device: str,
              chunk: int, overlap: int, batch_size: int) -> np.ndarray:
    L = len(seq)
    spans = list(chunk_indices(L, chunk, overlap))
    hidden = model.config.d_model
    sum_emb = np.zeros((L, hidden), dtype=np.float32)
    count = np.zeros((L,), dtype=np.float32)
    for i in range(0, len(spans), batch_size):
        batch_spans = spans[i:i + batch_size]
        batch_seqs = [" ".join(_RARE_AA.sub("X", seq[a:b])) for a, b in batch_spans]
        enc = tokenizer.batch_encode_plus(
            batch_seqs, add_special_tokens=True, padding="longest", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        hs = out.last_hidden_state.float().cpu().numpy()  # (B, T, H)
        attn = enc["attention_mask"].cpu().numpy()
        for (a, b), h, am in zip(batch_spans, hs, attn):
            n_tokens = int(am.sum())
            residue_h = h[:n_tokens - 1]  # drop the single trailing EOS
            if residue_h.shape[0] != (b - a):
                raise RuntimeError(f"chunk {a}:{b} got {residue_h.shape[0]} residue tokens, expected {b - a}")
            sum_emb[a:b] += residue_h
            count[a:b] += 1
    return sum_emb / np.maximum(count, 1.0)[:, None]


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import T5EncoderModel, T5Tokenizer
    LOG.info("loading %s on %s", args.model, args.device)
    tokenizer = T5Tokenizer.from_pretrained(args.model, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(args.model).eval().to(args.device)

    n_done = n_skipped = 0
    for fasta_id, seq in read_fasta(args.fasta):
        out_path = args.output_dir / f"{fasta_id}.npy"
        if args.skip_existing and out_path.is_file():
            n_skipped += 1
            continue
        LOG.info("embedding %s (L=%d)", fasta_id, len(seq))
        emb = embed_one(seq, model, tokenizer, args.device,
                        args.chunk_size, args.overlap, args.batch_size)
        np.save(out_path, emb.astype(np.float32))
        n_done += 1
    LOG.info("done: %d embedded, %d skipped -> %s", n_done, n_skipped, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
