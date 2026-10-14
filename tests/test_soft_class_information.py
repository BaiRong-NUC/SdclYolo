import numpy as np

from sdcl.evaluation.soft_class_information import class_statistics, decide


def test_class_rank_coverage_and_correctness_gap():
    logits = np.zeros((3, 10), np.float32)
    logits[0, :3] = [5, 1, 0]
    logits[1, :3] = [3, 2, 1]
    logits[2, :4] = [3, 2, 1, .5]
    targets = np.eye(10, dtype=np.float32)[[0, 2, 3]]
    result = class_statistics(logits, targets, np.ones(3, bool))
    assert result["wrong_rows"] == 2
    assert result["wrong_top3_coverage"] == .5
    assert result["wrong_target_rank_histogram"][:3] == [0, 1, 1]
    assert result["top2_gap_correctness_auc"] == 1.


def test_only_active_tiny_positive_rows_enter_and_no_wrong_blocks_pass():
    logits = np.zeros((4, 10), np.float32)
    logits[1] = -10
    targets = np.eye(10, dtype=np.float32)[[0, 1, 2, 3]]
    targets[3] = 0
    result = class_statistics(logits, targets, [True, True, False, True])
    assert result["tiny_positive_rows_at_score025"] == 1
    assert result["wrong_top3_coverage"] is None
    policy = dict(holdout_tiny_wrong_count_min=20, holdout_tiny_wrong_top3_coverage_min=.6,
                  holdout_tiny_gap_auc_min=.65)
    assert decide({"holdout": result}, policy)["decision"] == "STOP_THIS_SOFT_CLASS_SIGNAL"
