"""GRU sequence-to-one RUL regression model."""
from dataclasses import dataclass

import torch.nn as nn


class GRUForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 25,
        input_window: int = 512,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_size: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_window = input_window
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_size),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


@dataclass
class ModelConfig:
    input_size: int = 25
    input_window: int = 512
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2


def build_model(cfg: ModelConfig) -> nn.Module:
    return GRUForecaster(
        input_size=cfg.input_size,
        input_window=cfg.input_window,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
