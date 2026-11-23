"""
N-BEATS-style forecasting model for the FL C-MAPSS pipeline.

Input shape:
    (batch_size, window_size, num_features)

Example:
    (256, 96, 25)

Output shape:
    (batch_size, 1)
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


class NBEATSBlock(nn.Module):
    """
    Generic N-BEATS block.

    The block learns a backcast used to refine the residual input and a
    one-step forecast used by the final prediction head.
    """

    def __init__(
        self,
        backcast_size: int,
        forecast_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_features = backcast_size
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_features, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden_size

        self.layers = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden_size, backcast_size)
        self.forecast = nn.Linear(hidden_size, forecast_size)

    def forward(self, x):
        hidden = self.layers(x)
        return self.backcast(hidden), self.forecast(hidden)


class NBEATSForecaster(nn.Module):
    """
    N-BEATS-style sequence-to-one regression model.

    The multivariate window is flattened before entering stacked generic
    N-BEATS blocks. Each block explains part of the residual input and
    contributes to the final one-step forecast.
    """

    def __init__(
        self,
        input_size: int = 25,
        input_window: int = 512,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_blocks: int = 4,
        dropout: float = 0.2,
        forecast_size: int = 1,
    ):
        super().__init__()

        self.input_size = input_size
        self.input_window = input_window
        self.backcast_size = input_window * input_size
        self.forecast_size = forecast_size

        self.blocks = nn.ModuleList(
            [
                NBEATSBlock(
                    backcast_size=self.backcast_size,
                    forecast_size=forecast_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        residual = x.reshape(x.shape[0], -1)
        forecast = torch.zeros(x.shape[0], self.forecast_size, device=x.device, dtype=x.dtype)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return forecast


@dataclass
class ModelConfig:
    input_size: int = 25
    input_window: int = 512
    hidden_size: int = 128
    num_layers: int = 2
    num_blocks: int = 4
    dropout: float = 0.2


def build_model(cfg: ModelConfig) -> nn.Module:
    return NBEATSForecaster(
        input_size=cfg.input_size,
        input_window=cfg.input_window,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        num_blocks=cfg.num_blocks,
        dropout=cfg.dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


if __name__ == "__main__":
    cfg = ModelConfig()
    model = build_model(cfg)
    x = torch.randn(32, 512, 25)
    y = model(x)
    print("Output shape:", y.shape)
    print("Trainable parameters:", count_parameters(model))
