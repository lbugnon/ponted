"""
`Method.load(method_dir)` reads `method.yaml` and the `fold{0..N}/` 
(plain head state_dicts + per-fold standardization stats). `Method.score()`
assembles features per fold (backbone embedding ++ AF2 ++ b2a, each z-scored with
that fold's training stats), runs the fold head, and averages the per-residue
sigmoid scores across folds — the 5-fold ensemble.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from .features_sequence import SequenceCfg, build_features as build_seq_features, channel_names as seq_channel_names
from .features_structure import AF2_CHANNEL, FeatureError, build_structure_features
from .model import load_head
from .smoothing import smooth

LOG = logging.getLogger("methods")


def _load_stats(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as fh:
        return {"mean": fh["mean"].astype(np.float32), "std": fh["std"].astype(np.float32)}


@dataclass
class _Fold:
    head: torch.nn.Module
    struct_stats: Optional[Dict[str, np.ndarray]]
    seq_stats: Optional[Dict[str, np.ndarray]]


def _load_fold_stats(fold_dir: Path, filename: str, channels: List[str], label: str
                     ) -> Optional[Dict[str, np.ndarray]]:
    """A fold's standardization stats for one feature group, or None if it has no such channel."""
    if not channels:
        return None
    path = fold_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"{fold_dir}: method uses {label} but {filename} missing")
    return _load_stats(path)


class Method:
    def __init__(self, name: str, spec: dict, folds: List[_Fold]):
        self.name = name
        self.description = spec.get("description", "")
        self.backbone_id = spec["backbone"]["id"]
        self.backbone_dim = int(spec["backbone"]["embed_dim"])
        self.input_dim = int(spec["input_dim"])
        self.head_name = spec["head"]["name"]
        self.head_kwargs = {k: v for k, v in spec["head"].items() if k != "name"}
        self.structure_channels: List[str] = list(spec.get("structure_channels", []))
        self.sequence_channels: List[str] = list(spec.get("sequence_channels", []))
        self.seq_cfg: SequenceCfg = SequenceCfg.from_config(spec.get("sequence"))
        self.binary_threshold = float(spec.get("binary_threshold", 0.5))
        sm = spec.get("smoothing") or {}
        self.smoothing_method: Optional[str] = sm.get("method")
        self.smoothing_window: int = int(sm.get("window", 0) or 0)
        self.needs_af2 = AF2_CHANNEL in self.structure_channels
        self.folds = folds
        self._check_consistency()

    def _check_consistency(self) -> None:
        derived = seq_channel_names(self.seq_cfg)
        if derived != self.sequence_channels:
            raise ValueError(
                f"{self.name}: sequence config yields channels {derived} but method.yaml "
                f"lists {self.sequence_channels} — config/stat drift.")
        bad = [c for c in self.structure_channels if c != AF2_CHANNEL]
        if bad:
            raise ValueError(f"{self.name}: unsupported structure channel(s) {bad} (only af2 at inference).")
        expected = self.backbone_dim + len(self.structure_channels) + len(self.sequence_channels)
        if expected != self.input_dim:
            raise ValueError(
                f"{self.name}: input_dim {self.input_dim} != backbone {self.backbone_dim} + "
                f"{len(self.structure_channels)} struct + {len(self.sequence_channels)} seq = {expected}.")

    @classmethod
    def load(cls, method_dir: Path) -> "Method":
        method_dir = Path(method_dir)
        spec_path = method_dir / "method.yaml"
        if not spec_path.is_file():
            raise FileNotFoundError(f"{method_dir}: missing method.yaml")
        spec = yaml.safe_load(spec_path.read_text())
        method = cls(spec.get("name", method_dir.name), spec, folds=[])  # parses + validates spec
        method.folds = method._load_folds(method_dir)
        LOG.info("loaded method %s (%d folds, input_dim=%d, af2=%s, b2a=%d ch, smoothing=%s)",
                 method.name, len(method.folds), method.input_dim, method.needs_af2,
                 len(method.sequence_channels),
                 f"{method.smoothing_method}/{method.smoothing_window}" if method.smoothing_method else "off")
        return method

    def _load_folds(self, method_dir: Path) -> List[_Fold]:
        """Load each fold's head + standardization stats from the fold*/ subdirectories."""
        fold_dirs = sorted(d for d in method_dir.glob("fold*") if d.is_dir())
        if not fold_dirs:
            raise FileNotFoundError(f"{method_dir}: no fold*/ subdirectories")
        folds = []
        for fd in fold_dirs:
            ckpt = fd / "head.pt"
            if not ckpt.is_file():
                raise FileNotFoundError(f"{fd}: missing head.pt")
            head = load_head(ckpt, self.head_name, self.input_dim, **self.head_kwargs)
            struct_stats = _load_fold_stats(fd, "structure_stats.npz", self.structure_channels, "AF2")
            seq_stats = _load_fold_stats(fd, "sequence_stats.npz", self.sequence_channels, "b2a")
            folds.append(_Fold(head=head, struct_stats=struct_stats, seq_stats=seq_stats))
        return folds

    def score(self, fasta_id: str, sequence: str, embedding: np.ndarray,
              af2_dir: Optional[Path]) -> np.ndarray:
        """Return (L,) scores in [0,1]: the per-fold sigmoid mean, then smoothed."""
        emb = embedding.astype(np.float32, copy=False)
        per_fold = [self._fold_scores(fold, fasta_id, sequence, emb, af2_dir)
                    for fold in self.folds]
        scores = np.mean(per_fold, axis=0).astype(np.float64)
        return smooth(scores, self.smoothing_method, self.smoothing_window)

    def _fold_scores(self, fold: _Fold, fasta_id: str, sequence: str,
                     emb: np.ndarray, af2_dir: Optional[Path]) -> np.ndarray:
        """One fold's per-residue sigmoid scores: assemble [emb | struct | seq], run the head."""
        extras = []
        if self.structure_channels:
            extras.append(build_structure_features(
                fasta_id, len(sequence), self.structure_channels, fold.struct_stats, af2_dir))
        if self.sequence_channels:
            extras.append(build_seq_features(sequence, self.seq_cfg, fold.seq_stats))
        feats = np.concatenate([emb, *extras], axis=-1) if extras else emb
        if feats.shape[1] != self.input_dim:
            raise FeatureError(
                f"{fasta_id}: assembled feature width {feats.shape[1]} != input_dim {self.input_dim}")
        x = torch.from_numpy(feats).unsqueeze(0).float()
        attn = torch.ones((1, x.shape[1]), dtype=torch.float32)
        with torch.inference_mode():
            logits = fold.head(x, attn).squeeze(0).numpy()
        if logits.ndim == 2:  # multi-task head: linker task is channel 0
            logits = logits[:, 0]
        return 1.0 / (1.0 + np.exp(-logits))
