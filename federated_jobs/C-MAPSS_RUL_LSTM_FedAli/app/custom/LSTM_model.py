"""FedAli LSTM model for sequence-to-one RUL regression.

The LSTM produces a sequence of embeddings. An Alignment with Prototypes (ALP)
layer aligns all timestep embeddings with either the shared global prototype
bank (during training) or the personalized local bank (during local evaluation),
and the aligned final timestep drives RUL regression. Local prototypes are
non-persistent buffers: they are updated by EMA, are never optimized by gradient
descent, and are not part of checkpoints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


GLOBAL_PROTOTYPE_KEY = "alp.global_prototypes"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class AlignmentWithPrototypes(nn.Module):
    """Single-layer FedAli alignment module with global and local prototypes."""

    def __init__(
        self,
        embedding_dim: int,
        num_prototypes: int = 16,
        beta: float = 0.2,
        gamma: float = 0.999,
        epsilon: float = 0.05,
        sinkhorn_iterations: int = 3,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_prototypes <= 0:
            raise ValueError("num_prototypes must be positive")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive")

        self.embedding_dim = int(embedding_dim)
        self.num_prototypes = int(num_prototypes)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        global_initial = torch.randn(self.num_prototypes, self.embedding_dim, generator=generator)
        local_initial = torch.randn(self.num_prototypes, self.embedding_dim, generator=generator)

        # Global prototypes are part of the federated state and server checkpoint.
        self.register_buffer("global_prototypes", global_initial, persistent=True)
        # Personalized prototypes stay only on the client and are uploaded through
        # a temporary copy under GLOBAL_PROTOTYPE_KEY after local training.
        self.register_buffer("local_prototypes", local_initial, persistent=False)
        self.local_initialized = False

        # Linear + GLU implements the prototype regulation gate described by FedAli.
        self.prototype_gate = nn.Linear(self.embedding_dim, self.embedding_dim * 2)
        self.glu = nn.GLU(dim=-1)

    @torch.no_grad()
    def initialize_local_from_global(self, force: bool = False) -> None:
        if force or not self.local_initialized:
            self.local_prototypes.copy_(self.global_prototypes)
            self.local_initialized = True

    @torch.no_grad()
    def set_local_prototypes(self, prototypes: torch.Tensor) -> None:
        prototypes = torch.as_tensor(
            prototypes,
            device=self.local_prototypes.device,
            dtype=self.local_prototypes.dtype,
        )
        if tuple(prototypes.shape) != tuple(self.local_prototypes.shape):
            raise ValueError(
                f"Expected local prototypes with shape {tuple(self.local_prototypes.shape)}, "
                f"got {tuple(prototypes.shape)}"
            )
        self.local_prototypes.copy_(prototypes)
        self.local_initialized = True

    def get_local_prototypes(self) -> torch.Tensor:
        return self.local_prototypes.detach().clone()

    def _sinkhorn(self, embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Return a balanced batch-by-prototype transport assignment."""
        x = F.normalize(embeddings, dim=-1)
        p = F.normalize(prototypes, dim=-1)
        logits = (x @ p.transpose(0, 1)) / self.epsilon
        logits = logits - logits.max()

        # Work as (prototypes, batch), following balanced Sinkhorn-Knopp.
        transport = torch.exp(logits).transpose(0, 1)
        transport = transport / transport.sum().clamp_min(1e-12)
        num_prototypes, batch_size = transport.shape
        for _ in range(self.sinkhorn_iterations):
            transport = transport / transport.sum(dim=1, keepdim=True).clamp_min(1e-12)
            transport = transport / float(num_prototypes)
            transport = transport / transport.sum(dim=0, keepdim=True).clamp_min(1e-12)
            transport = transport / float(batch_size)

        # Rows now sum to one, making them convenient assignment probabilities.
        return (transport * float(batch_size)).transpose(0, 1)

    @torch.no_grad()
    def _update_local_ema(self, embeddings: torch.Tensor, assignments: torch.Tensor) -> None:
        """Apply the paper's per-prototype Top-K weighted EMA update."""
        num_embeddings = embeddings.shape[0]
        top_k = min(num_embeddings, (num_embeddings + self.num_prototypes - 1) // self.num_prototypes)
        prototype_scores = assignments.detach().transpose(0, 1)
        scores, indices = prototype_scores.topk(top_k, dim=1, largest=True, sorted=False)
        weights = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
        selected = embeddings.detach()[indices]
        targets = (selected * weights.unsqueeze(-1)).sum(dim=1)
        updated = self.gamma * self.local_prototypes + (1.0 - self.gamma) * targets
        self.local_prototypes.copy_(updated)

    def _align(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
        assignments: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if assignments is None:
            assignments = self._sinkhorn(embeddings.detach(), prototypes.detach())
        matched_indices = assignments.argmax(dim=1)
        matched = prototypes.detach().index_select(0, matched_indices)
        regulated = self.glu(self.prototype_gate(matched))
        aligned = self.beta * regulated + (1.0 - self.beta) * embeddings
        return F.normalize(aligned, dim=-1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embeddings shaped (batch, {self.embedding_dim}), got {tuple(embeddings.shape)}"
            )

        if self.training:
            # A new client starts from its independently initialized local bank;
            # subsequent rounds restore that client's retained bank explicitly.
            self.local_initialized = True
            # Algorithm 1 computes one balanced transport plan over the combined
            # local/global bank and then partitions it into O_local and O_global.
            combined = torch.cat((self.local_prototypes, self.global_prototypes), dim=0)
            transport = self._sinkhorn(embeddings.detach(), combined.detach())
            local_assignments = transport[:, : self.num_prototypes]
            global_assignments = transport[:, self.num_prototypes :]
            aligned = self._align(embeddings, self.global_prototypes, global_assignments)
            self._update_local_ema(embeddings, local_assignments)
            return aligned

        # Personalized client evaluation uses local prototypes.  A freshly loaded
        # global checkpoint has no local memory, so it falls back to the global bank.
        bank = self.local_prototypes if self.local_initialized else self.global_prototypes
        return self._align(embeddings, bank)


class LSTMForecaster(nn.Module):
    """LSTM RUL regressor with one FedAli ALP layer."""

    def __init__(
        self,
        input_size: int = 25,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_prototypes: Optional[int] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        epsilon: Optional[float] = None,
        sinkhorn_iterations: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        num_prototypes = num_prototypes or _env_int("FEDALI_NUM_PROTOTYPES", 16)
        beta = _env_float("FEDALI_BETA", 0.2) if beta is None else beta
        gamma = _env_float("FEDALI_GAMMA", 0.999) if gamma is None else gamma
        epsilon = _env_float("FEDALI_EPSILON", 0.05) if epsilon is None else epsilon
        sinkhorn_iterations = sinkhorn_iterations or _env_int("FEDALI_SINKHORN_ITERATIONS", 3)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.alp = AlignmentWithPrototypes(
            embedding_dim=hidden_size,
            num_prototypes=num_prototypes,
            beta=beta,
            gamma=gamma,
            epsilon=epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            seed=seed,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        batch_size, sequence_length, hidden_size = lstm_out.shape
        # The paper applies ALP to all B*Z embeddings. For the LSTM adaptation,
        # Z is the sequence length and the aligned final timestep drives RUL.
        aligned = self.alp(lstm_out.reshape(batch_size * sequence_length, hidden_size))
        aligned = aligned.reshape(batch_size, sequence_length, hidden_size)
        return self.regressor(aligned[:, -1, :])

    def set_local_prototypes(self, prototypes: torch.Tensor) -> None:
        self.alp.set_local_prototypes(prototypes)

    def get_local_prototypes(self) -> torch.Tensor:
        return self.alp.get_local_prototypes()

    def initialize_local_from_global(self, force: bool = False) -> None:
        self.alp.initialize_local_from_global(force=force)

    def federation_state_dict(self, upload_local_prototypes: bool = False):
        state = self.state_dict()
        if upload_local_prototypes:
            state[GLOBAL_PROTOTYPE_KEY] = self.get_local_prototypes()
        return state


@dataclass
class ModelConfig:
    input_size: int = 25
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    num_prototypes: int = 16
    beta: float = 0.2
    gamma: float = 0.999
    epsilon: float = 0.05
    sinkhorn_iterations: int = 3
    seed: int = 42


def build_model(cfg: ModelConfig) -> nn.Module:
    return LSTMForecaster(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        num_prototypes=cfg.num_prototypes,
        beta=cfg.beta,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon,
        sinkhorn_iterations=cfg.sinkhorn_iterations,
        seed=cfg.seed,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    model = build_model(ModelConfig())
    output = model(torch.randn(32, 512, 25))
    print("Output shape:", tuple(output.shape))
    print("Trainable parameters:", count_parameters(model))
    print("Global prototypes:", tuple(model.alp.global_prototypes.shape))
