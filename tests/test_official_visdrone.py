from pathlib import Path

import numpy as np
import pytest

from sdcl.evaluation.official_visdrone import detection_rows, write_detection, write_runner


def test_export_preserves_fractional_coordinates_score_and_category(tmp_path):
    boxes = np.array([[0.25, 2.75, 10.125, 23.875, .1234567890123456, 9]])
    target = tmp_path / "predictions.txt"
    write_detection(target, boxes, (50, 50))
    rows = np.loadtxt(target, delimiter=",", ndmin=2)
    assert np.array_equal(rows, [[.25, 2.75, 9.875, 21.125, boxes[0, 4], 10, -1, -1]])


@pytest.mark.parametrize("box", [
    [-1, 0, 2, 2, .1, 0],
    [0, 0, 101, 2, .1, 0],
    [0, 0, 0, 2, .1, 0],
    [0, 0, 2, 2, .0001, 0],
    [0, 0, 2, 2, .1, 10],
    [0, 0, 2, 2, .1, .5],
    [0, 0, 2, 2, np.nan, 0],
])
def test_reject_invalid_export(box):
    with pytest.raises(ValueError):
        detection_rows([box], (100, 100))


def test_reject_unsorted_or_more_than_500_detections():
    with pytest.raises(ValueError):
        detection_rows([[0, 0, 2, 2, .1, 0], [0, 0, 2, 2, .2, 0]], (100, 100))
    with pytest.raises(ValueError):
        detection_rows(np.tile([0, 0, 2, 2, .1, 0], (501, 1)), (100, 100))


def test_runner_calls_original_official_functions_and_preserves_results(tmp_path):
    write_runner(tmp_path)
    code = (tmp_path / "run_official.m").read_text()
    assert "saveAnnoRes(" in code and "calcAccuracy(" in code
    assert "assert(numImgs == 548)" in code
    assert "Refusing to overwrite result" in code
    assert "dropObjectsInIgr" not in code
    assert (tmp_path / "compat/mean2.m").exists()
