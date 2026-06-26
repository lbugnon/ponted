"""
Input: (B, L, H) embeddings + (B, L) attention mask (1 = valid, 0 = pad).
Output: (B, L) logits for n_outputs=1, else (B, L, n_outputs).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn


def load_head(state_dict_path: Path, name: str, input_dim: int, **head_kwargs) -> nn.Module:
    """Build the head at `input_dim` and load a plain state_dict from disk.

    `input_dim` is the FULL per-residue feature width fed to the head (backbone +
    structure + sequence channels), i.e. `method.yaml:input_dim` — NOT the bare
    backbone dim. `head_kwargs` must match training (d_model, num_layers, …) or
    the load fails loudly. Only the `transformer_lite` head is served.
    """
    if name != "transformer_lite":
        raise KeyError(f"unknown head {name!r} (only 'transformer_lite' is supported)")
    state = torch.load(Path(state_dict_path), map_location="cpu", weights_only=True)
    head = _TransformerHead(input_dim, **head_kwargs)
    head.load_state_dict(state, strict=True)
    head.eval()
    return head


class _SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE on the (B, L, D) tensor. Buffer grows on demand."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", self._build(d_model, max_len), persistent=False)

    @staticmethod
    def _build(d_model: int, length: int) -> torch.Tensor:
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        if L > self.pe.size(0):
            self.pe = self._build(self.d_model, L).to(x.device, dtype=self.pe.dtype)
        return x + self.pe[:L].to(dtype=x.dtype)


class _TransformerHead(nn.Module):
    """Small transformer encoder on top of frozen pLM embeddings."""

    def __init__(self, embed_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 1,
                 dim_feedforward: int = 256, dropout: float = 0.2, n_outputs: int = 1,
                 input_norm: bool = False, **_unused):
        super().__init__()
        self.input_norm = nn.LayerNorm(embed_dim) if input_norm else nn.Identity()
        self.proj_in = nn.Linear(embed_dim, d_model)
        self.pos = _SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj_out = nn.Linear(d_model, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(self.input_norm(embeddings))
        x = self.pos(x)
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.proj_out(x)
        if self.n_outputs == 1:
            x = x.squeeze(-1)
            return x * attention_mask.to(x.dtype)
        return x * attention_mask.unsqueeze(-1).to(x.dtype)
