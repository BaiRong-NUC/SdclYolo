import numpy as np
import pytest

from sdcl.evaluation.official_visdrone import METRICS
from sdcl.evaluation.testdev_transfer import global_predictions, transfer_decision


def metrics():
    result = {a: {k: 30. for k in METRICS} for a in ("baseline", "soft_nms", "calibration")}
    result["baseline"]["AP"] = 29.
    result["calibration"]["AP"] = 31.
    return result


def test_transfer_pass_and_single_failed_gate_stops():
    values = metrics()
    assert transfer_decision(values, True)["decision"] == "TRANSFER_PASS"
    values["calibration"]["AR500"] = 29.8
    assert transfer_decision(values, True)["decision"] == "TRANSFER_STOP"
    assert transfer_decision(values, False)["decision"] == "SMOKE_ONLY"


def test_transfer_rejects_missing_or_nonfinite_metrics():
    values = metrics()
    values["calibration"]["AP"] = np.nan
    with pytest.raises(ValueError):
        transfer_decision(values, True)
    with pytest.raises(ValueError):
        transfer_decision({"baseline": {}}, True)


def test_global_policy_clips_drops_degenerate_and_preserves_scores_classes():
    boxes = np.array([[-1, -1, 12, 9, .4, 2], [5, 5, 5, 6, .9, 1],
                      [0, 0, 2, 2, .7, 3]])
    original = boxes.copy()
    actual = global_predictions(boxes, (10, 8))
    np.testing.assert_array_equal(actual, [[0, 0, 2, 2, .7, 3], [0, 0, 10, 8, .4, 2]])
    np.testing.assert_array_equal(boxes, original)
    assert global_predictions([], (10, 8)).shape == (0, 6)
