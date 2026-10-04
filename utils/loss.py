import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import math


def prototype_relation_kl(current, previous, current_prototypes, previous_prototypes, temperature=0.2):
    """Cosine relation retention, matching the inference head's geometry.

    Only current features receive gradients: the teacher and both prototype
    anchors are detached. This is a PRD-inspired no-exemplar ablation, not the
    replay-based CCLIS algorithm.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("relation_temperature must be positive and finite.")
    if current_prototypes.shape != previous_prototypes.shape or current.shape != previous.shape:
        raise ValueError("Teacher/student relation shapes must match.")
    if len(current_prototypes) < 2:
        return current.sum() * 0.0
    teacher_scores = F.normalize(previous.detach(), dim=1) @ F.normalize(previous_prototypes.detach(), dim=1).T
    student_scores = F.normalize(current, dim=1) @ F.normalize(current_prototypes.detach(), dim=1).T
    teacher = F.softmax(teacher_scores / temperature, dim=1)
    student = F.log_softmax(student_scores / temperature, dim=1)
    return F.kl_div(student, teacher, reduction="batchmean") * temperature ** 2

class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='cosface', eps=1e-7, s=20, m=0):
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface', 'crossentropy']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 20.0 if not s else s
            self.m = 0.0 if not m else m
        self.loss_type = loss_type
        self.eps = eps

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wf, labels):
        if self.loss_type == 'crossentropy':
            return self.cross_entropy(wf, labels)
        else:
            if self.loss_type == 'cosface':
                numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
            if self.loss_type == 'arcface':
                numerator = self.s * torch.cos(torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
            if self.loss_type == 'sphereface':
                numerator = self.s * torch.cos(self.m * torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

            excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
            # Algebraically identical to log(exp(target)+sum(exp(others))),
            # without overflow for high scale / low-precision training.
            all_scores = torch.cat((numerator[:, None], self.s * excl), dim=1)
            return (torch.logsumexp(all_scores, dim=1) - numerator).mean()
