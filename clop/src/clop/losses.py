"""Contrastive objectives.

Many sequences share one caption (every human promoter reads "homo sapiens ... promoter"). ``multi_positive``
(default) contrasts each sequence with the *distinct* captions of its batch, which states the task directly: name the
caption. ``clip`` (CLOP v1) pairs every sequence with its own caption row; identical rows give identical logits, so it
equals the multi-positive loss with logits shifted by the in-batch caption log-frequencies plus a constant, and trains
an equivalent model (it only never reaches zero). ``sigmoid`` scores every sequence-caption pair independently (SigLIP).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_loss(dna: torch.Tensor, text: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE over a batch of paired rows (one text row per sequence, duplicates included)."""
    logits = scale * dna @ text.T
    labels = torch.arange(len(dna), device=dna.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def multi_positive_loss(dna: torch.Tensor, text: torch.Tensor, owner: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """InfoNCE between B sequences and U distinct captions; ``owner[i]`` is the caption index of sequence i.

    DNA -> text is a U-way classification; text -> DNA spreads the target uniformly over each caption's sequences."""
    logits = scale * dna @ text.T  # (B, U)
    to_text = F.cross_entropy(logits, owner)
    positives = F.one_hot(owner, text.shape[0]).T.float()  # (U, B)
    targets = positives / positives.sum(1, keepdim=True)
    to_dna = -(targets * F.log_softmax(logits.T, dim=1)).sum(1).mean()
    return 0.5 * (to_text + to_dna)


def sigmoid_loss(
    dna: torch.Tensor, text: torch.Tensor, owner: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Pairwise sigmoid loss (SigLIP) with every sequence-caption pair labelled +1 if the caption is the sequence's."""
    logits = scale * dna @ text.T + bias
    labels = 2.0 * F.one_hot(owner, text.shape[0]).float() - 1.0
    return -F.logsigmoid(labels * logits).sum(1).mean()


LOSSES = ("multi_positive", "sigmoid", "clip")
