"""Model definitions and helpers for inference."""

import torch
import torch.nn as nn


class MediumMLP(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 4096):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridConvNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.input_dim = 32 * 32 * 3  # For reshape & dummy compatibility

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # ↓ resolution
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.mlp(x)
        return x



class ToyHybrid(nn.Module):
    def __init__(self, input_dim=2048):
        super().__init__()
        self.input_dim = input_dim

        self.linear1 = nn.Linear(input_dim, input_dim)  # ~4M ops
        self.relu = nn.ReLU()                           # ~4k ops (1D input)
        self.linear2 = nn.Linear(input_dim, input_dim)  # ~4M ops

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x





def build_model(model, input_dim: int = 2048, hidden_dim: int = 4096) -> nn.Module:
    """Create the MediumMLP model.""" 
    match model:
        case 'HybridConvNet':
            return HybridConvNet(input_dim=input_dim, hidden_dim=hidden_dim)
        case 'MediumMLP':
            return MediumMLP(input_dim=input_dim, hidden_dim=hidden_dim)
        case 'ToyHybrid':
            return ToyHybrid()
        case _:
            raise ValueError(f"Invalid model type: {model}")


def make_dummy_input(batch_size: int, input_dim: int, device: torch.device) -> torch.Tensor:
    """Generate dummy input for timing/inference."""

    return torch.randn(batch_size, input_dim, device=device)


def run_inference(model: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    """Run a single forward pass with gradients disabled."""

    model.eval()
    with torch.no_grad():
        return model(input_tensor)
