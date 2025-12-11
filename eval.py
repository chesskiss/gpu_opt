"""Inference timing utilities."""

import time
import torch
import torch.nn as nn

from model import make_dummy_input, run_inference


def _sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def time_inference(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 512,
    runs: int = 5,
    input_dim: int = 2048,
) -> float:
    """Time forward passes for a model on the specified device."""

    if not hasattr(model, "device_map"):
        model.to(device)
        dummy = torch.randn(batch_size, input_dim).to(device)
    else:
        dummy = torch.randn(batch_size, input_dim)  # leave on CPU for FX routing

    # Warmup
    for _ in range(2):
        _ = run_inference(model, dummy)

    _sync_device(device)

    start = time.time()
    for _ in range(runs):
        _ = run_inference(model, dummy)
    _sync_device(device)

    return (time.time() - start) / runs
