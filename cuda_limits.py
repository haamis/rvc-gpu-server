"""Per-process cap on torch's CUDA allocator.

RVC_CUDA_MEMORY_FRACTION=0.375 caps every torch process in the container to
37.5% of the card (6GB of the 16GB RTX 4080), matching ifrit's 6GB 2060 so the
same chunk sizes and margins hold on both machines. Unset = no cap.
"""
import logging
import os

log = logging.getLogger("rvc_server.cuda")


def cuda_memory_fraction() -> float | None:
    """The configured fraction of device memory, or None when unset."""
    raw = os.environ.get("RVC_CUDA_MEMORY_FRACTION", "").strip()
    if not raw:
        return None
    fraction = float(raw)
    if not 0 < fraction <= 1:
        raise ValueError(f"RVC_CUDA_MEMORY_FRACTION must be in (0, 1], got {raw!r}")
    return fraction


def apply_cuda_memory_fraction() -> None:
    """Cap torch allocations to RVC_CUDA_MEMORY_FRACTION of the device.

    Call before the first CUDA allocation. A no-op without the env var; a
    machine without CUDA only logs (the cap has nothing to do).
    """
    fraction = cuda_memory_fraction()
    if fraction is None:
        return
    import torch

    if not torch.cuda.is_available():
        log.warning("RVC_CUDA_MEMORY_FRACTION=%s ignored: CUDA is not available", fraction)
        return
    torch.cuda.set_per_process_memory_fraction(fraction)
    log.info("Torch CUDA memory capped at %.1f%% of device memory", fraction * 100)
