import pytest

from sdcl.evaluation.candidate_series import CANDIDATES, METRICS, confirm_decision


def metrics(candidate):
    return {a: {k: 30. for k in METRICS} for a in CANDIDATES[candidate]["arms"]}


def test_positive_total_gain_does_not_establish_relative_mechanism():
    values = metrics("relative")
    values["gain"]["AP"] += .3
    values["absolute"]["AP"] += .4
    result = confirm_decision("relative", values)
    assert result["decision"] == "NO_TRANSFERRED_MECHANISM_ADVANTAGE"
    assert result["values"]["AP_vs_calibration_pp"] > 0


def test_modest_positive_effect_can_pass_without_auxiliary_gates():
    values = metrics("ranking")
    values["regret"]["AP"] += .15
    values["uniform"]["AP"] += .1
    values["regret"]["AR500"] -= .3
    assert confirm_decision("ranking", values)["decision"] == "PROMISING_POINT_ESTIMATE"


def test_small_signal_is_not_called_no_effect():
    values = metrics("relative")
    values["gain"]["AP"] += .05
    assert confirm_decision("relative", values)["decision"] == "SMALL_POSITIVE_SIGNAL"


def test_recall_tradeoff_reported_even_with_positive_ap():
    values = metrics("relative")
    values["gain"]["AP"] += .2
    values["gain"]["AR500"] -= .6
    assert confirm_decision("relative", values)["decision"] == "AP_RECALL_TRADEOFF_EXCEEDS_BUDGET"


def test_retention_does_not_need_large_ap_improvement():
    values = metrics("retention")
    values["gain"]["AR500"] += .2
    assert confirm_decision("retention", values)["decision"] == "PROMISING_POINT_ESTIMATE"


def test_missing_matched_control_and_nonfinite_metrics_rejected():
    values = metrics("relative")
    del values["absolute"]
    with pytest.raises(ValueError):
        confirm_decision("relative", values)
    values = metrics("ranking")
    values["regret"]["AP"] = float("nan")
    with pytest.raises(ValueError):
        confirm_decision("ranking", values)
