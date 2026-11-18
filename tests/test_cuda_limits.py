"""The torch VRAM cap is a container budget split across its two torch processes."""
import sys
import types

import pytest

from cuda_limits import apply_cuda_memory_fraction, cuda_memory_fraction


class FakeCuda:
    def __init__(self, available=True):
        self.available = available
        self.calls = []
        self.properties = types.SimpleNamespace(total_memory=16 * 2**30)

    def is_available(self):
        return self.available

    def get_device_properties(self, index):
        return self.properties

    def set_per_process_memory_fraction(self, fraction):
        self.calls.append(fraction)


def fake_torch(monkeypatch, available=True) -> FakeCuda:
    cuda = FakeCuda(available)
    module = types.ModuleType("torch")
    module.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", module)
    return cuda


def test_budget_is_opt_in(monkeypatch):
    monkeypatch.delenv("RVC_CUDA_MEMORY_FRACTION", raising=False)
    assert cuda_memory_fraction("server") is None
    assert cuda_memory_fraction("worker") is None
    monkeypatch.setenv("RVC_CUDA_MEMORY_FRACTION", "  ")
    assert cuda_memory_fraction("server") is None


def test_default_split_fills_the_whole_budget(monkeypatch):
    monkeypatch.setenv("RVC_CUDA_MEMORY_FRACTION", "0.375")
    monkeypatch.delenv("RVC_CUDA_SERVER_SHARE", raising=False)
    server = cuda_memory_fraction("server")
    worker = cuda_memory_fraction("worker")
    assert server == pytest.approx(0.1875)
    assert worker == pytest.approx(0.1875)
    assert server + worker == pytest.approx(0.375)


def test_explicit_server_share(monkeypatch):
    monkeypatch.setenv("RVC_CUDA_MEMORY_FRACTION", "0.375")
    monkeypatch.setenv("RVC_CUDA_SERVER_SHARE", "0.25")
    assert cuda_memory_fraction("server") == pytest.approx(0.09375)
    assert cuda_memory_fraction("worker") == pytest.approx(0.28125)


@pytest.mark.parametrize("raw, expected", [("0.375", 0.375), ("1", 1.0)])
def test_budget_is_parsed(monkeypatch, raw, expected):
    monkeypatch.setenv("RVC_CUDA_MEMORY_FRACTION", raw)
    monkeypatch.setenv("RVC_CUDA_SERVER_SHARE", "0.5")
    assert cuda_memory_fraction() == pytest.approx(expected * 0.5)


@pytest.mark.parametrize("raw", ["0", "-0.1", "1.5", "six"])
def test_out_of_range_budget_is_rejected(monkeypatch, raw):
    monkeypatch.setenv("RVC_CUDA_MEMORY_FRACTION", raw)
    with pytest.raises(ValueError):
        cuda_memory_fraction()


@pytest.mark.parametrize("raw", ["0", "1", "1.5", "half"])
def test_split_must_leave_both_processes_a_share(monkeypatch, raw):
    monkeypatch.setenv("RVC_CUDA_MEMORY_FRACTION", "0.375")
    monkeypatch.setenv("RVC_CUDA_SERVER_SHARE", raw)
    with pytest.raises(ValueError):
        cuda_memory_fraction()


def test_role_must_be_known(monkeypatch):
    monkeypatch.setenv("RVC_CUDA_MEMORY_FRACTION", "0.375")
    with pytest.raises(ValueError):
        cuda_memory_fraction("gpu")


@pytest.mark.parametrize("role, expected", [("server", 0.09375), ("worker", 0.28125)])
def test_each_process_applies_its_own_share(monkeypatch, role, expected):
    cuda = fake_torch(monkeypatch)
    monkeypatch.setenv("RVC_CUDA_MEMORY_FRACTION", "0.375")
    monkeypatch.setenv("RVC_CUDA_SERVER_SHARE", "0.25")
    apply_cuda_memory_fraction(role)
    assert cuda.calls == [pytest.approx(expected)]


def test_unset_budget_touches_nothing(monkeypatch):
    cuda = fake_torch(monkeypatch)
    monkeypatch.delenv("RVC_CUDA_MEMORY_FRACTION", raising=False)
    apply_cuda_memory_fraction("server")
    apply_cuda_memory_fraction("worker")
    assert cuda.calls == []


def test_cpu_host_only_warns(monkeypatch):
    cuda = fake_torch(monkeypatch, available=False)
    monkeypatch.setenv("RVC_CUDA_MEMORY_FRACTION", "0.375")
    apply_cuda_memory_fraction("worker")
    assert cuda.calls == []
