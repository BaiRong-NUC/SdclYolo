import numpy as np
import pytest

from sdcl.evaluation.localization_official_audit import (
    ARMS, METRICS, audit_summary, interference_diagnostic,
)


def metrics():
    return {arm: {key: 30. for key in METRICS} for arm in ARMS}


def test_audit_keeps_primary_comparison_and_does_not_relabel_stop():
    values = metrics()
    values["uniform"]["AP"] = 30.3
    values["regret"]["AP"] = 30.2
    result = audit_summary(values)
    assert result["status"] == "NO_TRAINING_AUDIT_ONLY"
    assert not result["ranking_positive_point_estimates"]
    assert result["comparisons"]["regret_vs_calibration"]["AP_pp"] > 0
    assert result["comparisons"]["regret_vs_uniform"]["AP_pp"] < 0
    assert "Unchanged" in result["historical_gates"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1, 101])
def test_audit_rejects_invalid_official_values(bad):
    values = metrics()
    values["regret"]["AP"] = bad
    with pytest.raises(ValueError):
        audit_summary(values)


def test_audit_requires_all_matched_controls():
    values = metrics()
    del values["uniform"]
    with pytest.raises(ValueError):
        audit_summary(values)


def test_harm_is_not_reported_as_proven_identity_switch():
    record = {"stem": "crowd", "objects": [
        {"box_xyxy": [0, 0, 10, 10], "class_id": 0},
        {"box_xyxy": [5, 0, 15, 10], "class_id": 0},
    ]}
    calibration = np.array([[0, 0, 10, 10, .9, 0], [5, 0, 15, 10, .8, 0]])
    changed = calibration.copy()
    changed[0, :4] = [5, 0, 15, 10]
    predictions = {arm: changed for arm in ("gain", "uniform", "regret")}
    selected = {arm: np.array([1, 1]) for arm in predictions}
    boxes = np.repeat(calibration[:, None, :4], 6, axis=1)
    boxes[:, 1] = [5, 0, 15, 10]
    result = interference_diagnostic(record, calibration, predictions, selected, np.zeros(2), boxes)
    assert all(row["owned"] == 2 for row in result)
    assert all(row["worsened_owned"] == 1 for row in result)
    assert all(row["other_gt_better_after_harm"] == 1 for row in result)
    assert all(row["peer_reused_distinct_owners"] == 2 for row in result)
