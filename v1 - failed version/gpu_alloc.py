"""Device selection helpers for CPU/CUDA/MPS."""

import torch


def pick_device(preference: str = "auto") -> torch.device:
    """Choose a device based on user preference and availability (CUDA, MPS, CPU)."""

    pref = preference.lower()
    if pref == "cpu":
        return torch.device("cpu")

    if pref in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("Requested CUDA/GPU but it is not available; trying MPS...")
        pref = "mps"

    if pref in ("mps", "metal"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("Requested MPS but it is not available; falling back to CPU.")
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
