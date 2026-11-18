"""The torch VRAM cap must be opt-in, validated, and harmless without CUDA."""
import sys
import types

import pytest

from cuda_limits import apply_cuda_memory_fraction, cuda_memory_fraction


class FakeCuda:
    def __init__(self, available=True):
        self.available = available
        self.calls = []

    def is_available(self):
        return self.available

    def set_per_process_memory_fraction(self, fraction):
        self.calls.append(fraction)


def fake_torch(monkeypatch, available=True) -> FakeCuda:
    cuda = FakeCuda(available)
    module = types.ModuleType("torch")
    module.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", module)
    return cuda


def test_fraction_is_opt_in(monkeypatch):
    monkeypatch.delenv("RVC_CUDA_MEMORY_FRACTION", raising=False)
    assert cuda_memory_fraction() is None
    monkeypatch.setenv("RVC_CUDA_MEMORY_FRACTION", "  ")
    assert cuda_memory_fraction() is None


@pytest.mark.parametrize("raw, expected", [("0.375", 0.375), ("1", 1.0)])
def test_fraction_is_parsed(monkeypatch, raw, expected):
    monkeypatch.setenv("RVC_CUDA_MEMORY_FRACTION", raw)
    assert cuda_memory_fraction() == expected


@pytest.mark.parametrize("raw", ["0", "-0.1", "1.5", "six"])
def test_out_of_range_fraction_is_rejected(monkeypatch, raw):
    monkeypatch.setenv("RVC_CUDA_MEMORY_FRACTION", raw)
    with pytest.raises(ValueError):
        cuda_memory_fraction()


def test_cap_is_applied_when_configured(monkeypatch):
    cuda = fake_torch(monkeypatch)
    monkeypatch.setenv("RVC_CUDA_MEMORY_FRACTION", "0.375")
    apply_cuda_memory_fraction()
    assert cuda.calls == [0.375]


def test_unset_fraction_touches_nothing(monkeypatch):
    cuda = fake_torch(monkeypatch)
    monkeypatch.delenv("RVC_CUDA_MEMORY_FRACTION", raising=False)
    apply_cuda_memory_fraction()
    assert cuda.calls == []


def test_cpu_host_only_warns(monkeypatch):
    cuda = fake_torch(monkeypatch, available=False)
    monkeypatch.setenv("RVC_CUDA_MEMORY_FRACTION", "0.375")
    apply_cuda_memory_fraction()
    assert cuda.calls == []
