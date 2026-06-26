#!/usr/bin/env python3
"""AlphaFold-DB per-residue pLDDT.

Host-side (needs internet for the EBI AlphaFold DB). For each FASTA record it resolves a UniProt accession, fetches the AFDB model, and writes `<fasta_id>.npy` of shape (L,) in [0,1] (CA pLDDT / 100) into `--output-dir`. The FASTA id must be a UniProt accession, else pass `--id-map map.tsv` (`fasta_id<TAB>uniprot_acc`); a record whose model length != the sequence length is skipped with a warning. See the README ("Stage 1") for the output layout and dimensions.

Usage:
  pip install -r precompute/requirements-precompute.txt
  python precompute/compute_af2.py --fasta input.fasta --output-dir af2/ [--id-map map.tsv]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_fasta  

LOG = logging.getLogger("precompute_af2")
AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
USER_AGENT = "idpfun-linkers/precompute_af2"
UNIPROT_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$")
ISOFORM_RE = re.compile(r"^[A-Z0-9]+-\d+$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", type=Path, required=True, help="Input FASTA")
    p.add_argument("--output-dir", type=Path, required=True, help="Where <id>.npy pLDDT files are written.")
    p.add_argument("--id-map", type=Path, default=None,
                   help="Optional TSV: 'fasta_id<TAB>uniprot_acc' per line, for ids that are not UniProt accs.")
    p.add_argument("--sleep", type=float, default=0.2, help="Seconds between HTTP calls.")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def load_id_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    out: Dict[str, str] = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t") if "\t" in ln else ln.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    LOG.info("loaded %d id->uniprot mappings", len(out))
    return out


_NO_ID = {"", "-", "nan", "na", "none", "null"}


def resolve_acc(fasta_id: str, id_map: Dict[str, str]) -> Optional[str]:
    """UniProt accession for a FASTA id, or None if unknown.

    The host-provided id_map may mark a sequence as having no UniProt id with a
    sentinel ('-', 'nan', ...); such entries (and unmapped non-accession ids)
    return None so the sequence is skipped — partial coverage is expected.
    """
    if fasta_id in id_map:
        acc = id_map[fasta_id].strip()
        return acc if acc.lower() not in _NO_ID else None
    if UNIPROT_RE.match(fasta_id) or ISOFORM_RE.match(fasta_id):
        return fasta_id
    return None


def _http_get(url: str, timeout: float, retries: int, accept_json: bool = False):
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
        except requests.RequestException as e:
            LOG.warning("HTTP error %d/%d on %s: %s", attempt + 1, retries, url, e)
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            return None
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json() if accept_json else r.text
    return None


def fetch_pdb(acc: str, timeout: float, retries: int) -> Optional[str]:
    meta = _http_get(AF_API.format(acc=acc), timeout, retries, accept_json=True)
    if not meta:
        return None
    pdb_url = meta[0].get("pdbUrl")
    return _http_get(pdb_url, timeout, retries) if pdb_url else None


def parse_plddt_from_pdb(text: str) -> np.ndarray:
    """CA B-factors (pLDDT) ordered by residue number, /100 -> [0,1]. Empty on gaps."""
    plddt: Dict[int, float] = {}
    for line in text.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        try:
            plddt[int(line[22:26].strip())] = float(line[60:66].strip())
        except ValueError:
            continue
    if not plddt:
        return np.empty(0, dtype=np.float32)
    nums = sorted(plddt)
    if nums[0] != 1 or nums[-1] != len(nums):
        return np.empty(0, dtype=np.float32)  # fragmented numbering
    return np.asarray([plddt[i] for i in nums], dtype=np.float32) / 100.0


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    id_map = load_id_map(args.id_map)

    n_ok = n_skip = n_unresolved = n_404 = n_lenmismatch = 0
    for fasta_id, seq in read_fasta(args.fasta):
        out_path = args.output_dir / f"{fasta_id}.npy"
        if args.skip_existing and out_path.is_file():
            n_skip += 1
            continue
        acc = resolve_acc(fasta_id, id_map)
        if acc is None:
            LOG.warning("%s: no UniProt acc (not an accession; add it to --id-map) — skipping", fasta_id)
            n_unresolved += 1
            continue
        text = fetch_pdb(acc, args.timeout, args.retries)
        time.sleep(args.sleep)
        if text is None:
            LOG.info("%s (acc=%s): no AFDB entry — skipping", fasta_id, acc)
            n_404 += 1
            continue
        plddt = parse_plddt_from_pdb(text)
        if plddt.size != len(seq):
            LOG.warning("%s (acc=%s): AFDB len %d != sequence %d — skipping",
                        fasta_id, acc, plddt.size, len(seq))
            n_lenmismatch += 1
            continue
        np.save(out_path, plddt.astype(np.float32))
        n_ok += 1
    LOG.info("done: ok=%d skipped_existing=%d unresolved=%d 404=%d len_mismatch=%d -> %s",
             n_ok, n_skip, n_unresolved, n_404, n_lenmismatch, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
