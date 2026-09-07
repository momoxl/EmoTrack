"""Losses for Position-Shuffled Augmentation and Position Self-Distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossOutput:
    """Components of the joint PoSA + PoSD objective."""

    total: torch.Tensor
    posa: torch.Tensor
    posd: torch.Tensor
    teacher_correct: bool


def position_self_distillation_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    gold_index: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, bool]:
    """Compute correctness-gated KL(sg[p_teacher] || p_student).

    Logits are restricted to the shared answer options before this function is
    called. Gradients flow through the student distribution only.
    """
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    if teacher_logits.ndim != 1:
        raise ValueError("this reference implementation expects one logit vector per view")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= gold_index < teacher_logits.numel():
        raise ValueError("gold_index is outside the answer vocabulary")

    teacher_correct = int(teacher_logits.detach().argmax().item()) == gold_index
    if not teacher_correct:
        return student_logits.sum() * 0.0, False

    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="sum")
    return loss * (temperature**2), True


def posa_posd_loss(
    teacher_logits: torch.Tensor,
    student_logits: Sequence[torch.Tensor],
    gold_index: int,
    posd_weight: float = 1.0,
    temperature: float = 1.0,
) -> LossOutput:
    """Return L_PoSA + lambda * L_PoSD for one group of positional views."""
    if not student_logits:
        raise ValueError("at least one non-final student view is required")
    if posd_weight < 0:
        raise ValueError("posd_weight cannot be negative")

    gold = torch.tensor([gold_index], device=teacher_logits.device)
    posa = F.cross_entropy(teacher_logits.unsqueeze(0), gold)
    posd_terms: list[torch.Tensor] = []
    teacher_correct = int(teacher_logits.detach().argmax().item()) == gold_index

    for logits in student_logits:
        posa = posa + F.cross_entropy(logits.unsqueeze(0), gold)
        term, _ = position_self_distillation_loss(
            teacher_logits, logits, gold_index, temperature=temperature
        )
        posd_terms.append(term)

    posd = torch.stack(posd_terms).sum()
    return LossOutput(
        total=posa + posd_weight * posd,
        posa=posa,
        posd=posd,
        teacher_correct=teacher_correct,
    )

