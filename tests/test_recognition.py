import pytest

from utils.recognition import CaptureGate


@pytest.fixture
def gate():
    return CaptureGate(threshold=0.7, stable_frames=3)


def test_capture_requires_consecutive_confident_frames(gate):
    for _ in range(2):
        assert not gate.observe("A", 0.9).ready
        assert gate.capture() == ""
    assert gate.observe("A", 0.9).ready
    assert gate.capture() == "A"
    assert gate.capture() == ""
    assert not gate.observe("A", 0.9).ready


@pytest.mark.parametrize("interruption", [None, "B", "low-confidence"])
def test_interrupted_streak_cannot_capture(gate, interruption):
    gate.observe("A", 0.9)
    gate.observe("A", 0.9)
    if interruption == "low-confidence":
        assert gate.observe("A", 0.6).status == "UNCERTAIN"
    else:
        gate.observe(interruption, 0.9)
    assert not gate.observe("A", 0.9).ready
    assert gate.capture() == ""


def test_stale_prediction_and_reset_cannot_capture():
    now = [0.0]
    gate = CaptureGate(stable_frames=1, clock=lambda: now[0])
    gate.observe("A", 0.9)
    now[0] = 1
    assert gate.capture() == ""
    gate.observe("B", 0.9)
    gate.reset()
    assert gate.capture() == ""


def test_nonfinite_confidence_is_uncertain(gate):
    assert gate.observe("A", float("nan")).status == "UNCERTAIN"


@pytest.mark.parametrize("kwargs", [{"threshold": -1}, {"threshold": 2},
                                     {"stable_frames": 0}, {"max_age": 0}])
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CaptureGate(**kwargs)
