"""Simple environment-driven configuration for hardware/tests."""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    device_preference: str = "auto"
    batch_size: int = 512
    runs: int = 5
    model_name: str = "MediumMLP"
    input_dim: int = 2048
    hidden_dim: int = 4096


def load_config() -> Settings:
    """Load settings from environment variables with sane defaults."""

    return Settings(
        device_preference=os.getenv("DEVICE_PREFERENCE", "auto"),
        batch_size=int(os.getenv("BATCH_SIZE", "512")),
        runs=int(os.getenv("RUNS", "5")),
        model_name=os.getenv("MODEL_NAME", "MediumMLP"),
        input_dim=int(os.getenv("INPUT_DIM", "2048")),
        hidden_dim=int(os.getenv("HIDDEN_DIM", "4096")),
    )
