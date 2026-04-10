import torch


def resolve_device(device_arg: str | None) -> torch.device:
    """Resolve torch device from CLI or auto-detect."""
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
