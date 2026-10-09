import numpy as np
import pytest

from utils.preview import fit_frame


@pytest.mark.parametrize("viewport", [(800, 600), (400, 100), (1, 1), (0, 0)])
def test_preview_fits_viewport_and_converts_bgr_to_rgb(viewport):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = [10, 30, 240]
    rgb = fit_frame(frame, *viewport)
    assert 1 <= rgb.shape[1] <= max(1, viewport[0])
    assert 1 <= rgb.shape[0] <= max(1, viewport[1])
    assert tuple(rgb[0, 0]) == (240, 30, 10)


@pytest.mark.parametrize("frame", [np.zeros((2, 2)), np.zeros((2, 2, 4), dtype=np.uint8),
                                    np.zeros((0, 2, 3), dtype=np.uint8)])
def test_bad_preview_frames_report_errors(frame):
    with pytest.raises(ValueError):
        fit_frame(frame, 100, 100)
