import numpy as np

from sdcl.postprocess.class_choice import choose_labels, make_features
from sdcl.training.class_choice_preflight import pilot_decision


def test_context_features_are_tied_to_actual_view_class_evidence():
    candidate = np.array([[10, 10, 30, 30, .8, 1]], float)
    score, raw, context = np.zeros((1, 10, 5)), np.zeros((1, 10, 5)), np.ones((1, 5, 4))
    score[0, 2, 3], raw[0, 2, 3] = .4, .5
    features = make_features(candidate, score, raw, context, [100, 100])
    assert features.shape == (1, 10, 62)
    assert not features[0, 1, -20:].any()
    np.testing.assert_allclose(features[0, 2, -20:].reshape(5, 4)[3], .4)


def test_class_change_requires_supported_class_probability_gain_and_score_floor():
    candidates = np.array([[0, 0, 10, 10, .8, 1], [0, 0, 10, 10, .01, 1]], float)
    valid = np.zeros((2, 10), bool)
    valid[:, 1:3] = True
    utility = np.full((2, 10), 100.)
    utility[:, 1], utility[:, 2] = 0., 2.
    assert choose_labels(utility, candidates, valid).tolist() == [2, 1]


def test_context_must_beat_matched_evidence_head_not_only_baseline():
    metrics = {
        "uniform": {"small_all_recall_percent": 60., "all_fppi": 10.},
        "evidence_only": {"small_all_recall_percent": 61., "all_fppi": 9.},
        "observation_context": {"small_all_recall_percent": 60.5, "all_fppi": 9.5},
    }
    policy = {"context_small_all_recall_vs_uniform_min_pp": .1,
              "context_small_all_recall_vs_evidence_only_min_exclusive_pp": 0.,
              "context_all_fppi_vs_uniform_max_increase": .1}
    assert pilot_decision(metrics, policy)["decision"] == "STOP_THIS_CONTEXT_CLASS_CHOICE_PILOT"
