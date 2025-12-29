"""Run accelerator vs CPU inference timing using modular components."""

import torch

from env_config import load_config
from gpu_alloc import pick_device
from model import build_model
from eval import time_inference
from router import optimize


def run_once(model, device_preference: str, batch_size: int = 512, runs: int = 5) -> None:
    device = pick_device(device_preference)
    print(f"\nRunning on: {device}")

    avg_time = time_inference(model, device, batch_size=batch_size, runs=runs)
    print(f"Average forward pass time over {runs} runs: {avg_time:.4f} seconds")

    if device.type == "cpu":
        print("⚠️ CPU path; expect slower inference compared to accelerator.")
    else:
        print("✅ Accelerator path; expect faster inference.")


def optimized_run(model, device_preference: str = "auto", batch_size: int = 512, runs: int = 5):
    input_dim = getattr(model, "input_dim", 2048)

    # No device sent into optimize — it will route ops on its own
    model_fx = optimize(model, batch_size=batch_size, input_dim=input_dim)

    # Evaluate on requested device (CUDA/MPS if available) or fall back to CPU
    device = pick_device(device_preference)

    avg_time_fx = time_inference(model_fx, device, batch_size=batch_size, runs=runs)
    print(f"\nFX-Routed avg time on {device}: {avg_time_fx:.4f} seconds")


def main() -> None:
    cfg = load_config()

    print("Model speed test")
    model = build_model(cfg.model_name, input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim)
    run_once(model, cfg.device_preference, batch_size=cfg.batch_size, runs=cfg.runs)
    run_once(model, "cpu", batch_size=cfg.batch_size, runs=cfg.runs)
    optimized_run(model, device_preference=cfg.device_preference, batch_size=cfg.batch_size, runs=cfg.runs)


if __name__ == "__main__":
    main()
