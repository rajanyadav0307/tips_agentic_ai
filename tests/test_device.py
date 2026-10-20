import pytest

from agent_tips.utils import device
from agent_tips.utils.seed import set_seed


def test_auto_returns_known_device() -> None:
    assert device.select_device("auto") in {"cpu", "cuda", "mps"}


def test_cpu_is_always_available() -> None:
    assert device.select_device("cpu") == "cpu"


def test_without_torch_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "torch_available", lambda: False)
    assert device.select_device("auto") == "cpu"
    with pytest.raises(RuntimeError):
        device.select_device("cuda")


def test_unknown_preference_raises() -> None:
    with pytest.raises(ValueError):
        device.select_device("tpu")


def test_seed_is_reproducible() -> None:
    a = set_seed(7).integers(0, 1_000_000, size=5)
    b = set_seed(7).integers(0, 1_000_000, size=5)
    assert (a == b).all()
