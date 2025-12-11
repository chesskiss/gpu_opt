"""Device selection helpers for CPU/MPS (macOS)."""

import torch


def pick_device(preference: str = "auto") -> torch.device:
    """Choose a device based on user preference and availability (MPS or CPU)."""

    pref = preference.lower()
    if pref == "cpu":
        return torch.device("cpu")

    if pref in ("mps", "metal", "gpu"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("Requested MPS but it is not available; falling back to CPU.")
        return torch.device("cpu")

    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
