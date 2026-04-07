import torch


def get_device_info():
    device = "cpu"
    vram = None

    try:
        if torch.cuda.is_available():
            device = "cuda"
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        device = "cpu"
        vram = None

    return {
        "device": device,
        "vram_gb": round(vram, 2) if vram else None,
    }