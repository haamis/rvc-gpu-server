"""Container-wide budget for torch's CUDA allocator.

RVC_CUDA_MEMORY_FRACTION=0.375 is the budget for the whole container: 37.5% of
the 16GB RTX 4080 = 6GB, matching ifrit's 6GB 2060 so the same chunk sizes and
memory margins hold on both machines. Unset = no cap.

Torch can only cap itself per process, and this container runs two CUDA torch
processes (the server with its engines, the RVC worker), so the budget is
split: RVC_CUDA_SERVER_SHARE (default 1/2) of it for the server, the rest for
the worker. The two caps always add up to RVC_CUDA_MEMORY_FRACTION.
"""
import logging
import os

log = logging.getLogger("rvc_server.cuda")

TOTAL_ENV = "RVC_CUDA_MEMORY_FRACTION"
SERVER_SHARE_ENV = "RVC_CUDA_SERVER_SHARE"
DEFAULT_SERVER_SHARE = 0.5
ROLES = ("server", "worker")


def _fraction(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    value = float(raw)
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {raw!r}")
    return value


def cuda_memory_fraction(role: str = "server") -> float | None:
    """This process's share of the container's torch CUDA budget.

    None when unset. The server and worker shares add up to
    RVC_CUDA_MEMORY_FRACTION, so the container as a whole stays within it.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    total = _fraction(TOTAL_ENV)
    if total is None:
        return None
    share = _fraction(SERVER_SHARE_ENV)
    if share is None:
        share = DEFAULT_SERVER_SHARE
    elif share >= 1:
        raise ValueError(f"{SERVER_SHARE_ENV} must be in (0, 1), got {share!r}")
    return total * (share if role == "server" else 1 - share)


def apply_cuda_memory_fraction(role: str = "server") -> None:
    """Cap torch allocations to this role's share of the container budget.

    Call before the first CUDA allocation. A no-op without the env var; a
    machine without CUDA only logs (the cap has nothing to do).
    """
    fraction = cuda_memory_fraction(role)
    if fraction is None:
        return
    import torch

    if not torch.cuda.is_available():
        log.warning("%s ignored (%s): CUDA is not available", TOTAL_ENV, role)
        return
    torch.cuda.set_per_process_memory_fraction(fraction)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    log.info(
        "Torch CUDA (%s) capped at %.2f GiB: %.1f%% of the %.1f GiB device",
        role, fraction * total_gib, fraction * 100, total_gib,
    )
