import pytest

from app import get_args


@pytest.mark.parametrize("arguments", [["--device", "-1"], ["--width", "0"],
                                         ["--height", "-10"], ["--stable_frames", "0"],
                                         ["--confidence_threshold", "nan"],
                                         ["--confidence_threshold", "1.1"],
                                         ["--min_tracking_confidence", "-.1"],
                                         ["--min_detection_confidence", "inf"]])
def test_invalid_cli_arguments_are_rejected(arguments):
    with pytest.raises(SystemExit) as error:
        get_args(arguments)
    assert error.value.code == 2


def test_decimal_confidences_and_camera_selection():
    args = get_args(["--device", "2", "--min_tracking_confidence", ".6",
                     "--confidence_threshold", ".8", "--stable_frames", "7"])
    assert (args.device, args.min_tracking_confidence, args.confidence_threshold, args.stable_frames) == (2, .6, .8, 7)
