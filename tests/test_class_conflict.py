import numpy as np

from sdcl.evaluation.class_conflict_preflight import owned_oracle
from sdcl.postprocess.class_conflict import apply_labels, class_evidence, vote_labels


def test_view_duplicates_do_not_manufacture_additional_class_votes():
    anchor = np.array([[10, 10, 30, 30, .8, 0]], float)
    peer = np.array([[10, 10, 30, 30, .7, 1]], float)
    windows = [[0, 0, 100, 100]] * 5
    valid, score, _, _ = class_evidence(
        anchor, [[], np.repeat(peer, 10, axis=0), [], [], []], windows, [100, 100])
    assert valid[0, 1]
    np.testing.assert_allclose(score[0, 1].sum(), .7)


def test_tied_vote_retains_original_and_labels_preserve_scores_and_coordinates():
    anchor = np.array([[10, 10, 30, 30, .8, 3]], float)
    valid, score = np.ones((1, 10), bool), np.zeros((1, 10, 5))
    score[0, 1, 0] = score[0, 3, 0] = .8
    assert vote_labels(anchor, valid, score).tolist() == [3]
    changed = apply_labels(anchor, [1])
    np.testing.assert_array_equal(changed[:, :5], anchor[:, :5])


def test_oracle_cannot_introduce_class_absent_from_prediction_actions():
    record = {"objects": [{"box_xyxy": [10, 10, 30, 30], "class_id": 2}]}
    anchor = np.array([[10, 10, 30, 30, .8, 1]], float)
    valid = np.zeros((1, 10), bool)
    valid[0, 1] = True
    labels, count = owned_oracle(record, anchor, valid)
    assert labels.tolist() == [1] and count == 0
    valid[0, 2] = True
    labels, count = owned_oracle(record, anchor, valid)
    assert labels.tolist() == [2] and count == 1
