import numpy as np

from sdcl.evaluation.class_choice_signal import weighted_auc


def test_weighted_auc_handles_ties_reverse_ranking_and_missing_class():
    assert weighted_auc([True, False], [1., 1.], [2., 5.]) == .5
    assert weighted_auc([True, False], [0., 1.], [2., 5.]) == 0.
    assert weighted_auc([True, False], [1., 0.], [2., 5.]) == 1.
    assert weighted_auc([True, True], [1., 0.], [2., 5.]) is None


def test_weighted_auc_counts_probability_mass_not_row_counts():
    labels, scores, weights = [True, False, False], [1., 0., 2.], [1., 3., 1.]
    np.testing.assert_allclose(weighted_auc(labels, scores, weights), .75)
