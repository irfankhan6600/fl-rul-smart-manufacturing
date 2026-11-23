"""
DLinear forecasting model for the FL C-MAPSS pipeline.

Input shape:
    (batch_size, window_size, num_features)

Output shape:
    (batch_size, 1)
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.transpose(1, 2)).transpose(1, 2)
        return x


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x):
        trend = self.moving_average(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearForecaster(nn.Module):
    """Decomposition-linear sequence-to-one regression model."""

    def __init__(
        self,
        input_size: int = 25,
        input_window: int = 512,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        moving_avg: int = 25,
        forecast_size: int = 1,
    ):
        super().__init__()

        self.input_size = input_size
        self.input_window = input_window
        self.decomposition = SeriesDecomposition(moving_avg)
        self.seasonal_linear = nn.Linear(input_window, 1)
        self.trend_linear = nn.Linear(input_window, 1)
        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            *[
                layer
                for _ in range(max(num_layers - 1, 0))
                for layer in (
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            ],
            nn.Linear(hidden_size, forecast_size),
        )

    def forward(self, x):
        seasonal, trend = self.decomposition(x)
        seasonal = self.seasonal_linear(seasonal.transpose(1, 2)).squeeze(-1)
        trend = self.trend_linear(trend.transpose(1, 2)).squeeze(-1)
        return self.head(seasonal + trend)


@dataclass
class ModelConfig:
    input_size: int = 25
    input_window: int = 512
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    moving_avg: int = 25


def build_model(cfg: ModelConfig) -> nn.Module:
    return DLinearForecaster(
        input_size=cfg.input_size,
        input_window=cfg.input_window,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        moving_avg=cfg.moving_avg,
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
