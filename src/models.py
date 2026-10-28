from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenCLIPEncoder(nn.Module):
    """Frozen CLIP ViT-B/32 visual encoder."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | torch.device = "cpu"):
        super().__init__()
        try:
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        except ImportError as exc:
            raise ImportError(
                "FrozenCLIPEncoder requires transformers. Install requirements.txt before loading CLIP."
            ) from exc
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.output_dim = int(self.model.config.projection_dim)
        self.to(device)

    @torch.no_grad()
    def encode_pixels(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        output = self.model(pixel_values=pixel_values)
        return F.normalize(output.image_embeds, dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encode_pixels(pixel_values)


class ForensicBottleneckAdapter(nn.Module):
    """Lightweight d -> r -> d adapter with residual projection."""

    def __init__(self, input_dim: int, bottleneck_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.norm = nn.LayerNorm(bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, input_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forensic_embedding(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dropout(self.act(self.norm(self.down(x))))
        return F.normalize(z, dim=-1)

    def residual_projection(self, z: torch.Tensor) -> torch.Tensor:
        """Projects forensic bottleneck representation back to input CLIP space."""
        return self.up(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forensic_embedding(x)
        residual = self.residual_projection(z)
        return F.normalize(x + self.residual_scale * residual, dim=-1)


class ContentRepresentationPredictor(nn.Module):
    """Predicts a frozen face-recognition embedding from forensic features."""

    def __init__(self, forensic_dim: int, content_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(forensic_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, content_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambd)


class AdversarialContentSuppressor(nn.Module):
    """ACS = GRL + content representation predictor."""

    def __init__(self, forensic_dim: int, content_dim: int, grl_lambda: float = 1.0, hidden_dim: int = 256):
        super().__init__()
        self.grl = GradientReversal(grl_lambda)
        self.predictor = ContentRepresentationPredictor(forensic_dim, content_dim, hidden_dim)

    def set_lambda(self, value: float) -> None:
        self.grl.lambd = float(value)

    def forward(self, forensic: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.grl(forensic))


class StandardForensicClassifier(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


class IdentityForensicAdapter(nn.Module):
    """Pass-through adapter for the no-bottleneck ablation.

    Keeps the same interface as ForensicBottleneckAdapter so the CFD-FSA
    pipeline can call ``forensic_embedding`` without special-casing the model.
    """

    def __init__(self):
        super().__init__()

    def forensic_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)

    def residual_projection(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forensic_embedding(x)


class FBAEncoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.adapter = ForensicBottleneckAdapter(input_dim, bottleneck_dim, dropout)

    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        return self.adapter.forensic_embedding(clip_features)


class OptionalFrequencyBranch(nn.Module):
    """Secondary frequency branch used only for the optional ablation."""

    def __init__(self, input_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ManipulationSelectivePrototypeMemory(nn.Module):
    """Novelty 3: Manipulation-Selective Prototype Memory (MSPM).

    Maintains M learnable prototype memory slots representing diverse
    manipulation artifact signatures. Episodic prototypes attend to memory
    slots to retrieve relevant pattern priors, forming enriched prototypes.
    """

    def __init__(self, dim: int = 256, num_slots: int = 16, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        # Learnable memory slots initialized orthogonally for diverse coverage
        memory_init = torch.empty(num_slots, dim)
        nn.init.orthogonal_(memory_init)
        self.slots = nn.Parameter(memory_init)

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.enrich_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
        """Enriches episodic prototypes (C, D) using manipulation memory."""
        # prototypes: (C, D)
        c_in = prototypes.unsqueeze(0)  # (1, C, D)
        m_in = self.slots.unsqueeze(0)  # (1, M, D)

        # Cross-attention: query=prototypes, key/value=memory_slots
        retrieved, _ = self.attn(query=c_in, key=m_in, value=m_in)  # (1, C, D)
        retrieved = retrieved.squeeze(0)  # (C, D)

        # Residual gating & projection
        combined = torch.cat([prototypes, retrieved], dim=-1)  # (C, 2*D)
        delta = self.enrich_net(combined)
        enriched = self.norm(prototypes + delta)
        return F.normalize(enriched, dim=-1)


class CFD_FSA(nn.Module):
    """CLIP -> FBA -> ACS -> forensic embedding + optional MSPM."""

    def __init__(
        self,
        clip_dim: int,
        forensic_dim: int,
        content_dim: int,
        grl_lambda: float = 1.0,
        dropout: float = 0.1,
        use_frequency: bool = False,
        frequency_dim: int = 64,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
        use_mspm: bool = False,
        mspm_slots: int = 16,
    ):
        super().__init__()
        self.fba = ForensicBottleneckAdapter(clip_dim, forensic_dim, dropout)
        self.use_frequency = use_frequency
        self.frequency = OptionalFrequencyBranch(frequency_dim, forensic_dim) if use_frequency else None
        self.acs = AdversarialContentSuppressor(forensic_dim, content_dim, grl_lambda)
        self.learnable_temperature = learnable_temperature
        init_log_t = float(math.log(max(1e-4, temperature)))
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(init_log_t, dtype=torch.float32))
        else:
            self.register_buffer("log_temperature", torch.tensor(init_log_t, dtype=torch.float32))

        self.use_mspm = use_mspm
        self.mspm = (
            ManipulationSelectivePrototypeMemory(forensic_dim, num_slots=mspm_slots)
            if use_mspm
            else None
        )

    def get_temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(min=0.01, max=1.0)

    def forensic(self, clip_features: torch.Tensor, frequency_features: torch.Tensor | None = None) -> torch.Tensor:
        z = self.fba.forensic_embedding(clip_features)
        if self.use_frequency:
            if frequency_features is None:
                raise ValueError("Frequency features are required when use_frequency=True.")
            z = F.normalize(z + self.frequency(frequency_features), dim=-1)
        return z

    def residual_projection(self, z: torch.Tensor) -> torch.Tensor:
        """Projects forensic representation back to CLIP feature space for ORC."""
        return self.fba.residual_projection(z)

    def enrich_prototypes(self, prototypes: torch.Tensor) -> torch.Tensor:
        """Novelty 3: Enriches prototypes using MSPM if enabled."""
        if self.use_mspm and self.mspm is not None:
            return self.mspm(prototypes)
        return prototypes

    def forward(self, clip_features: torch.Tensor, frequency_features: torch.Tensor | None = None):
        z = self.forensic(clip_features, frequency_features)
        content_pred = self.acs(z)
        return z, content_pred
