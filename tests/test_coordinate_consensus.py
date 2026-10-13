import numpy as np
import pytest

from sdcl.evaluation.coordinate_consensus import consensus, decide
from sdcl.evaluation.official_visdrone import METRICS


def test_coordinate_disagreement_falls_back_without_score_change():
    c = np.array([[0., 0., 10., 10., .4, 0.], [20., 20., 30., 30., .5, 1.]])
    p = c.copy()
    p[:, 0] += 1
    q = p.copy()
    q[1, 0] += .000001
    result = consensus(c, [p, p, q])
    assert np.array_equal(result[0], p[0])
    assert np.array_equal(result[1], c[1])
    assert np.array_equal(result[:, 4:], c[:, 4:])


def test_missing_head_or_changed_score_rejected():
    c = np.array([[0., 0., 10., 10., .4, 0.]])
    with pytest.raises(ValueError):
        consensus(c, [c, c])
    p = c.copy()
    p[0, 4] += .01
    with pytest.raises(ValueError):
        consensus(c, [c, c, p])


def test_small_ap_tradeoff_is_allowed_but_must_improve_ar_and_matched_method():
    m = {a: {k: 30. for k in METRICS} for a in ("uniform_consensus", "regret_consensus")}
    original = {a: {k: 30. for k in METRICS} for a in ("calibration", "regret")}
    original["regret"]["AP"] += .24
    m["regret_consensus"]["AP"] += .2
    m["regret_consensus"]["AR500"] += .01
    assert decide(m, original)["decision"] == "PROMISING_CONSENSUS_TRADEOFF"
    m["uniform_consensus"]["AP"] += .21
    assert decide(m, original)["decision"] == "ORDINARY_CONSENSUS_TRADEOFF_ONLY"
    m["regret_consensus"]["AR500"] = 30.
    assert decide(m, original)["decision"] == "NO_BUDGETED_AP_RECALL_TRADEOFF"
