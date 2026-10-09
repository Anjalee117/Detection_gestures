import numpy as np
import pytest

from utils.landmarks import pre_process_landmark


def test_normalization_is_translation_and_scale_invariant_without_mutation():
    points = np.arange(42).reshape(21, 2).astype(float)
    original = points.copy()
    result = pre_process_landmark(points)
    np.testing.assert_allclose(result, pre_process_landmark(points * 3 + [90, -20]))
    np.testing.assert_array_equal(points, original)
    assert len(result) == 42 and result[:2] == [0, 0]
    assert max(map(abs, result)) == 1


def test_coincident_landmarks_do_not_divide_by_zero():
    assert pre_process_landmark([[7, 9] for _ in range(21)]) == [0] * 42


@pytest.mark.parametrize("points", [[], [[0, 0]] * 20, [[0, 0, 0]] * 21,
                                      [[float("nan"), 0]] * 21, [[float("inf"), 0]] * 21])
def test_invalid_landmarks_fail_loudly(points):
    with pytest.raises(ValueError, match="21 finite"):
        pre_process_landmark(points)
