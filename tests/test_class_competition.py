import numpy as np

from sdcl.postprocess.class_competition import competition_evidence, suppress_scores


def views(probability, boxes=None):
    probability = np.asarray(probability, float)
    if boxes is None:
        boxes = np.tile([0., 0, 10, 10], (len(probability), 1))
    full = np.zeros((len(probability), 10))
    full[:, :probability.shape[1]] = probability
    return [(boxes, full) for _ in range(5)]


def test_stronger_class_supported_at_common_site_downranks_only_lower_row():
    predictions = np.array([[0., 0, 10, 10, .9, 0], [0, 0, 10, 10, .8, 1]])
    result = competition_evidence(predictions, views([[.8, .2]]))
    np.testing.assert_array_equal(result["pairs"], [[1, 0]])
    np.testing.assert_array_equal(result["witnesses"], [[0]*5])
    np.testing.assert_allclose(result["penalty_shared"], [0, 1.2])
    output, rows = suppress_scores(predictions, result["penalty_shared"])
    np.testing.assert_array_equal(output[:, :4], predictions[rows, :4])
    np.testing.assert_array_equal(output[:, 5], predictions[rows, 5])
    assert output[1, 4] < .8 and output[0, 4] == .9


def test_common_site_favoring_lower_row_vetoes_suppression():
    predictions = np.array([[0., 0, 10, 10, .9, 0], [0, 0, 10, 10, .8, 1]])
    result = competition_evidence(predictions, views([[.2, .8]]))
    np.testing.assert_array_equal(result["penalty_shared"], [0., 0])
    np.testing.assert_array_equal(suppress_scores(predictions, result["penalty_shared"])[0], predictions)


def test_same_class_and_separate_instances_do_not_compete():
    predictions = np.array([[0., 0, 10, 10, .9, 0], [0, 0, 10, 10, .8, 0], [20, 0, 30, 10, .7, 1]])
    result = competition_evidence(predictions, views([[.8, .2]]))
    assert result["pairs"].shape == (0, 2)
    for arm in ("shared", "marginal", "affinity", "geometry"):
        np.testing.assert_array_equal(result["penalty_"+arm], np.zeros(3))


def test_independent_sites_do_not_equal_a_shared_site_witness():
    predictions = np.array([[0., 0, 10, 10, .9, 0], [0, 0, 10, 10, .8, 1]])
    result = competition_evidence(predictions, views([[.9, .1], [.2, .8]]))
    assert result["contrast_shared"][0] > result["contrast_marginal"][0] > 0
    np.testing.assert_array_equal(result["witnesses"], [[0]*5])


def test_empty_native_evidence_has_no_semantic_suppression():
    predictions = np.array([[0., 0, 10, 10, .9, 0], [0, 0, 10, 10, .8, 1]])
    result = competition_evidence(predictions, views(np.empty((0, 2))))
    for arm in ("shared", "marginal", "affinity"):
        np.testing.assert_array_equal(result["penalty_"+arm], [0, 0])
    assert result["penalty_geometry"][1] == 2.


def test_score_floor_stable_ties_and_empty_predictions():
    predictions = np.array([[0., 0, 10, 10, .5, 0], [0, 0, 10, 10, .5, 1], [0, 0, 10, 10, .001, 2]])
    output, rows = suppress_scores(predictions, [0., 0, 1])
    np.testing.assert_array_equal(rows, [0, 1])
    assert len(output) == 2
    result = competition_evidence(np.empty((0, 6)), views([[.8, .2]]))
    assert result["pairs"].shape == (0, 2)
