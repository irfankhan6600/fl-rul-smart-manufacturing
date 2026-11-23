"""
PatchTST-style forecasting model for the FL C-MAPSS pipeline.

Input shape:
    (batch_size, window_size, num_features)

Output shape:
    (batch_size, 1)
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


class PatchTSTForecaster(nn.Module):
    """Patch-based Transformer encoder for sequence-to-one regression."""

    def __init__(
        self,
        input_size: int = 25,
        input_window: int = 512,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        patch_len: int = 16,
        stride: int = 8,
        num_heads: int = 4,
        forecast_size: int = 1,
    ):
        super().__init__()

        self.input_size = input_size
        self.input_window = input_window
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = 1 + max(0, (input_window - patch_len) // stride)

        self.patch_projection = nn.Linear(input_size * patch_len, hidden_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_patches * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_size),
        )

    def forward(self, x):
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().view(x.shape[0], self.num_patches, -1)
        hidden = self.patch_projection(patches) + self.position_embedding
        hidden = self.encoder(hidden)
        hidden = self.norm(hidden)
        return self.head(hidden)


@dataclass
class ModelConfig:
    input_size: int = 25
    input_window: int = 512
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    patch_len: int = 16
    stride: int = 8
    num_heads: int = 4


def build_model(cfg: ModelConfig) -> nn.Module:
    return PatchTSTForecaster(
        input_size=cfg.input_size,
        input_window=cfg.input_window,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        patch_len=cfg.patch_len,
        stride=cfg.stride,
        num_heads=cfg.num_heads,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = ModelConfig()
    model = build_model(cfg)
    x = torch.randn(32, 512, 25)
    y = model(x)
    print("Output shape:", y.shape)
    print("Trainable parameters:", count_parameters(model))
