from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_logits(query: torch.Tensor, prototypes: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    query = F.normalize(query, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    return query @ prototypes.t() / temperature


def build_prototypes(support_embeddings: torch.Tensor, support_labels: torch.Tensor, n_classes: int = 2) -> torch.Tensor:
    prototypes = []
    for cls in range(n_classes):
        cls_features = support_embeddings[support_labels == cls]
        if len(cls_features) == 0:
            raise ValueError(f"Support set has no examples for class {cls}.")
        prototypes.append(F.normalize(cls_features.mean(dim=0, keepdim=True), dim=-1).squeeze(0))
    return torch.stack(prototypes, dim=0)


def prototypical_logits(
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    temperature: float | torch.Tensor = 0.07,
    prototype_enricher: callable | None = None,
):
    prototypes = build_prototypes(support_embeddings, support_labels)
    if prototype_enricher is not None:
        prototypes = prototype_enricher(prototypes)
    logits = cosine_logits(query_embeddings, prototypes, temperature)
    return logits, prototypes


def prototypical_loss(
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    temperature: float | torch.Tensor = 0.07,
    prototype_enricher: callable | None = None,
):
    logits, prototypes = prototypical_logits(
        support_embeddings,
        support_labels,
        query_embeddings,
        temperature,
        prototype_enricher=prototype_enricher,
    )
    return F.cross_entropy(logits, query_labels), logits, prototypes
