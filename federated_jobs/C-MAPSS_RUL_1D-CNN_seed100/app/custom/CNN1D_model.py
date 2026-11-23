"""Compact 1D-CNN/TCN-style sequence-to-one RUL regression model."""
from dataclasses import dataclass

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class CNN1DForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 25,
        input_window: int = 512,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        kernel_size: int = 7,
        forecast_size: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_window = input_window
        self.input_projection = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                TemporalBlock(
                    channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=2 ** idx,
                    dropout=dropout,
                )
                for idx in range(num_layers)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_size),
        )

    def forward(self, x):
        x = torch.nan_to_num(x).transpose(1, 2)
        hidden = self.input_projection(x)
        hidden = self.blocks(hidden)
        pooled = self.pool(hidden)
        return self.head(pooled)


@dataclass
class ModelConfig:
    input_size: int = 25
    input_window: int = 512
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    kernel_size: int = 7


def build_model(cfg: ModelConfig) -> nn.Module:
    return CNN1DForecaster(
        input_size=cfg.input_size,
        input_window=cfg.input_window,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
