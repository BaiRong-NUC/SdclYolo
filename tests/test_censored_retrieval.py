import numpy as np

from sdcl.evaluation.censored_retrieval import score_image
from sdcl.postprocess.censored_retrieval import append_selected, peer_scores
from sdcl.evaluation.size_analysis import box_areas


def test_other_view_must_prefer_candidate_to_its_suppressor():
    candidate = np.array([[0, 0, 10, 10, .4, 0]], float)
    suppressor = np.array([[2, 0, 12, 10, .5, 0]], float)
    # Identical evidence in the same view cannot establish independent support.
    ordinary, contrast = peer_scores(candidate, np.array([0]), suppressor, candidate, np.array([0]))
    assert ordinary[0] == contrast[0] == 0
    ordinary, contrast = peer_scores(candidate, np.array([0]), suppressor, candidate, np.array([1]))
    assert ordinary[0] > 0 and contrast[0] > 0
    _, reverse = peer_scores(candidate, np.array([0]), suppressor, suppressor, np.array([1]))
    assert reverse[0] == 0


def test_duplicate_addition_is_fp_not_an_extra_recovered_object():
    baseline = np.array([[0, 0, 10, 10, .8, 0]], float)
    candidate = np.array([[0, 0, 10, 10, .1, 0]], float)
    merged, extra = append_selected(baseline, candidate, np.array([0]))
    result = score_image(merged, extra, baseline[:, :4], np.array([0]), box_areas(baseline[:, :4]))
    assert result["groups"]["all"]["tp50"] == 1
    assert result["added_tp50"] == 0
    assert result["fp50"] == result["added_duplicate_fp50"] == 1
