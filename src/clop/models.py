"""CLOP encoders and the dual-encoder model.

DNA encoders (all take N-padded code batches and lengths, so they are interchangeable):

* ``cnn``: dilated residual CNN on one-hot DNA, strand-summed so the embedding is exactly reverse-complement invariant
  (default; fast on CPU, exportable to ONNX and to the in-browser reader). Pooling takes the first two moments of the
  features over interior positions only (``pooling="moments"``): both are densities whose expectation does not change
  with sequence length, unlike a max (which grows with length) or edge-affected positions (whose share is ~1/length).
  ``pooling="meanmax"`` is the length-sensitive CLOP 2.0 pooling, kept for comparison.
* ``transformer``: k-mer tokens, pre-norm Transformer, masked mean pooling, optional strand averaging.
* ``lstm``: k-mer tokens and a bidirectional LSTM, the CLOP v1 encoder.
* ``fcgr``: frequency chaos game representation (a 2^k x 2^k image of k-mer frequencies) and a small 2D CNN;
  length-agnostic, strand-symmetric when ``rc`` is set.

Text encoders: ``bow`` (mean word embedding + MLP, default) and ``transformer``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn.functional as F
from torch import nn

from .tokenizers import N_CODE, kmer_ids, kmer_vocab_size, reverse_complement_codes


@dataclass
class ModelConfig:
    dna_encoder: str = "cnn"
    text_encoder: str = "bow"
    embed_dim: int = 128
    width: int = 128
    depth: int = 3
    kernel: int = 15
    k: int = 6
    stride: int = 1
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    rc: bool = True
    pooling: str = "moments"
    text_vocab_size: int = 2
    text_width: int = 128
    max_length: int = 4096
    window: int = 0  # inputs longer than this are embedded as the mean of overlapping windows (0: never tile)
    temperature: float = 0.07

    @classmethod
    def from_dict(cls, data: dict) -> ModelConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


def masked_mean_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean and max over the valid positions of (B, C, T) features; returns (B, 2C)."""
    m = mask[:, None, :].to(x.dtype)
    mean = (x * m).sum(-1) / m.sum(-1).clamp(min=1.0)
    peak = x.masked_fill(~mask[:, None, :], torch.finfo(x.dtype).min).amax(-1)
    return torch.cat([mean, torch.where(mask.any(-1, keepdim=True), peak, torch.zeros_like(peak))], 1)


def masked_moments(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean and mean square over the valid positions of (B, C, T) features; returns (B, 2C)."""
    m = mask[:, None, :].to(x.dtype)
    count = m.sum(-1).clamp(min=1.0)
    return torch.cat([(x * m).sum(-1) / count, (x * x * m).sum(-1) / count], 1)


def interior(mask: torch.Tensor, margin: int) -> torch.Tensor:
    """Positions at least ``margin`` steps from both ends of each (prefix) mask row: those whose receptive field lies
    inside the sequence. Rows too short to have any keep their whole mask."""
    n = mask.sum(-1, keepdim=True)
    steps = torch.arange(mask.shape[-1], device=mask.device)[None, :]
    inner = (steps >= margin) & (steps < n - margin)
    return inner | (mask & ~inner.any(-1, keepdim=True))  # logical form: ONNX Runtime has no boolean Where


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid positions of (B, T, C) features."""
    m = mask[..., None].to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


def one_hot(codes: torch.Tensor) -> torch.Tensor:
    """(B, L) codes -> (B, 4, L) float one-hot with N (and padding) as the zero vector."""
    return F.one_hot(codes.long(), N_CODE + 1)[..., :N_CODE].transpose(1, 2).float()


class Projection(nn.Sequential):
    def __init__(self, d_in: int, d_out: int, dropout: float):
        super().__init__(nn.Linear(d_in, d_in), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_in, d_out))


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.norm(F.gelu(self.conv(x))))


class ConvDNAEncoder(nn.Module):
    """Stem convolution, 2x max-pool, dilated residual blocks, length-invariant pooling, MLP head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.pooling not in ("moments", "meanmax"):
            raise ValueError(f"Unknown pooling {cfg.pooling!r}")
        self.rc, self.pooling = cfg.rc, cfg.pooling
        # receptive-field half-width in pooled steps: the stem's (kernel // 2 bases, halved) plus the dilated blocks'
        self.margin = (cfg.kernel // 2 + 1) // 2 + 2 * (2**cfg.depth - 1)
        self.stem = nn.Sequential(
            nn.Conv1d(4, cfg.width, cfg.kernel, padding=cfg.kernel // 2), nn.GELU(), nn.BatchNorm1d(cfg.width)
        )
        self.blocks = nn.Sequential(*[ResidualBlock(cfg.width, 2**i, cfg.dropout) for i in range(cfg.depth)])
        self.head = Projection(2 * cfg.width, cfg.embed_dim, cfg.dropout)

    def features(self, codes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Padding is kept inert: activations past each sequence's end are reset to zero (what a lone, unpadded
        sequence sees through convolution padding) and excluded from max-pooling, so batch composition never changes
        an embedding."""
        valid = (torch.arange(codes.shape[1], device=codes.device)[None, :] < lengths[:, None])[:, None, :]
        x = self.stem(one_hot(codes)).masked_fill(~valid, float("-inf"))
        x = F.max_pool1d(x, 2, ceil_mode=True)
        mask = torch.arange(x.shape[-1], device=codes.device)[None, :] < ((lengths + 1) // 2)[:, None]
        x = x.masked_fill(~mask[:, None, :], 0.0)
        for block in self.blocks:
            x = block(x).masked_fill(~mask[:, None, :], 0.0)
        if self.pooling == "meanmax":
            return masked_mean_max(x, mask)
        return masked_moments(x, interior(mask, self.margin))

    def pooled(self, codes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Pooled features before the projection head, summed over both strands when ``rc``."""
        pooled = self.features(codes, lengths)
        if self.rc:
            pooled = pooled + self.features(reverse_complement_codes(codes, lengths), lengths)
        return pooled

    def forward(self, codes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.head(self.pooled(codes, lengths))


class SinusoidalPositions(nn.Module):
    def __init__(self, width: int, max_length: int):
        super().__init__()
        position = torch.arange(max_length)[:, None]
        scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        table = torch.zeros(max_length, width)
        table[:, 0::2], table[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale[: width // 2])
        self.register_buffer("table", table, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.table[: x.shape[1]][None]


class KmerSequenceEncoder(nn.Module):
    """k-mer tokens through a Transformer or a bidirectional LSTM, masked mean pooling, MLP head."""

    def __init__(self, cfg: ModelConfig, kind: str):
        super().__init__()
        self.k, self.stride, self.rc, self.kind = cfg.k, cfg.stride, cfg.rc, kind
        self.embed = nn.Embedding(kmer_vocab_size(cfg.k), cfg.width, padding_idx=0)
        if kind == "transformer":
            self.positions = SinusoidalPositions(cfg.width, cfg.max_length)
            layer = nn.TransformerEncoderLayer(
                cfg.width, cfg.heads, 2 * cfg.width, cfg.dropout, batch_first=True, norm_first=True, activation="gelu"
            )
            self.body = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
            out = cfg.width
        else:
            # bidirectional LSTM as forward/backward pairs on length-reversed inputs: exact for padded batches without
            # packing, whose ever-changing per-step batch sizes make CPU LSTM kernels orders of magnitude slower
            self.body = nn.ModuleList(
                nn.ModuleList([nn.LSTM(cfg.width if i == 0 else 2 * cfg.width, cfg.width, batch_first=True) for _ in range(2)])
                for i in range(cfg.layers)
            )
            self.dropout = nn.Dropout(cfg.dropout)
            out = 2 * cfg.width
        self.head = Projection(out, cfg.embed_dim, cfg.dropout)

    def features(self, codes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        ids, counts = kmer_ids(codes, lengths, self.k, self.stride)
        mask = ids > 0
        x = self.embed(ids)
        if self.kind == "transformer":
            h = self.body(self.positions(x), src_key_padding_mask=~mask)
        else:
            steps = torch.arange(ids.shape[1], device=ids.device)
            order = (counts[:, None] - 1 - steps[None, :]).clamp(min=0)[..., None]  # reverses each row's valid prefix
            flip = lambda t: torch.gather(t, 1, order.expand(-1, -1, t.shape[-1]))
            h = x
            for i, (forward, backward) in enumerate(self.body):
                if i:
                    h = self.dropout(h)
                h = torch.cat([forward(h)[0], flip(backward(flip(h))[0])], -1)
        return masked_mean(h, mask)

    def pooled(self, codes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Pooled features before the projection head, averaged over both strands when ``rc``."""
        pooled = self.features(codes, lengths)
        if self.rc:
            pooled = 0.5 * (pooled + self.features(reverse_complement_codes(codes, lengths), lengths))
        return pooled

    def forward(self, codes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.head(self.pooled(codes, lengths))


def fcgr(codes: torch.Tensor, lengths: torch.Tensor, k: int, rc: bool) -> torch.Tensor:
    """Frequency chaos game representation: (B, 1, 2^k, 2^k) log-scaled k-mer frequencies (both strands if ``rc``).

    Base b contributes bit (b & 1) to x and (b >> 1) to y, the most recent base being the most significant, which is
    the classical CGR layout (A bottom-left, C bottom-right, G top-left, T top-right)."""

    def counts(c: torch.Tensor) -> torch.Tensor:
        ids, n = kmer_ids(c, lengths, k)
        valid = ids >= 2
        index = (ids - 2).clamp(min=0)
        digits = torch.stack([(index >> (2 * (k - 1 - j))) & 3 for j in range(k)], -1)  # (B, T, k), oldest first
        weights = 2 ** torch.arange(k, device=c.device)  # newest base -> highest bit
        x, y = ((digits & 1) * weights).sum(-1), ((digits >> 1) * weights).sum(-1)
        flat = torch.where(valid, y * (2**k) + x, torch.zeros_like(x))
        image = torch.zeros(c.shape[0], 4**k, device=c.device)
        image.scatter_add_(1, flat, valid.float())
        return image

    image = counts(codes)
    if rc:
        image = image + counts(reverse_complement_codes(codes, lengths))
    total = image.sum(1, keepdim=True).clamp(min=1.0)
    return torch.log1p(image / total * (4**k)).view(-1, 1, 2**k, 2**k)


class FCGREncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.k, self.rc = cfg.k, cfg.rc
        w = cfg.width // 4
        layers, channels = [], 1
        for i in range(max(cfg.k - 2, 1)):
            out = min(w * 2**i, cfg.width)
            layers += [nn.Conv2d(channels, out, 3, padding=1), nn.GELU(), nn.BatchNorm2d(out), nn.MaxPool2d(2)]
            channels = out
        self.body = nn.Sequential(*layers)
        self.head = Projection(2 * channels, cfg.embed_dim, cfg.dropout)

    def pooled(self, codes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Pooled image features before the projection head."""
        h = self.body(fcgr(codes, lengths, self.k, self.rc)).flatten(2)
        return torch.cat([h.mean(-1), h.amax(-1)], 1)

    def forward(self, codes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.head(self.pooled(codes, lengths))


class TextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.kind = cfg.text_encoder
        self.embed = nn.Embedding(cfg.text_vocab_size, cfg.text_width, padding_idx=0)
        if self.kind == "transformer":
            self.positions = SinusoidalPositions(cfg.text_width, 256)
            layer = nn.TransformerEncoderLayer(
                cfg.text_width, cfg.heads, 2 * cfg.text_width, cfg.dropout, batch_first=True, norm_first=True, activation="gelu"
            )
            self.body = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.head = Projection(cfg.text_width, cfg.embed_dim, cfg.dropout)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths[:, None]
        x = self.embed(tokens)
        if self.kind == "transformer":
            x = self.body(self.positions(x), src_key_padding_mask=~mask)
        return self.head(masked_mean(x, mask))


def build_dna_encoder(cfg: ModelConfig) -> nn.Module:
    if cfg.dna_encoder == "cnn":
        return ConvDNAEncoder(cfg)
    if cfg.dna_encoder in ("transformer", "lstm"):
        return KmerSequenceEncoder(cfg, cfg.dna_encoder)
    if cfg.dna_encoder == "fcgr":
        return FCGREncoder(cfg)
    raise ValueError(f"Unknown DNA encoder {cfg.dna_encoder!r}")


class CLOP(nn.Module):
    """Dual encoder: L2-normalised DNA and text embeddings in one space, with a learnable temperature (and bias,
    used by the sigmoid loss)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.dna = build_dna_encoder(cfg)
        self.text = TextEncoder(cfg)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / cfg.temperature)))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    def encode_dna(self, codes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.dna(codes, lengths), dim=-1)

    def encode_text(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text(tokens, lengths), dim=-1)

    def scale(self) -> torch.Tensor:
        return self.logit_scale.clamp(max=math.log(100.0)).exp()

    def forward(self, codes, lengths, tokens, token_lengths) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_dna(codes, lengths), self.encode_text(tokens, token_lengths)

    def n_parameters(self) -> dict[str, int]:
        count = lambda m: sum(p.numel() for p in m.parameters())
        return {"dna": count(self.dna), "text": count(self.text), "total": count(self)}
