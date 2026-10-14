import numpy as np

from sdcl.training.unowned_supervision import supervision_targets


def test_new_labels_are_fit_only_unowned_and_uniqueness_sees_detected_GT():
    gt = np.array([[0., 0, 10, 10], [.5, 0, 10.5, 10], [20, 0, 30, 10]])
    reference = np.array([[0., 0, 10, 10, .9, 1], [4, 0, 14, 10, .8, 1],
                          [23.5, 0, 33.5, 10, .7, 1]], np.float32)
    owner = np.array([0, -1, -1])
    original = np.zeros((3, 4), np.float32)
    original[0] = .02
    targets, owners, added = supervision_targets(gt, np.array([1, 1, 1]), reference, np.full((3, 4), 10.),
                                                np.ones(3, bool), owner, original, reference, (640, 640), "fixture")
    np.testing.assert_array_equal(added["naive"], [False, True, True])
    np.testing.assert_array_equal(added["unique"], [False, False, True])
    assert added["random_count"].sum() == added["unique"].sum() == 1
    assert np.all(~added["random_count"] | added["naive"])
    np.testing.assert_array_equal(owners["unique"], [0, -1, 2])
    np.testing.assert_allclose(targets["unique"][2], [-.1, 0, -.1, 0])
    np.testing.assert_array_equal(targets["unique"][:2], original[:2])
    for arm in targets:
        np.testing.assert_array_equal(targets[arm][0], original[0])
    np.testing.assert_array_equal(targets["original"], original)
    replay = supervision_targets(gt, np.array([1, 1, 1]), reference, np.full((3, 4), 10.), np.ones(3, bool),
                                 owner, original, reference, (640, 640), "fixture")
    np.testing.assert_array_equal(replay[2]["random_count"], added["random_count"])


def test_inactive_low_confidence_or_wrong_class_rows_do_not_gain_supervision():
    gt = np.array([[0., 0, 10, 10]])
    reference = np.array([[3.5, 0, 13.5, 10, .8, 2], [3.5, 0, 13.5, 10, .2, 1],
                          [3.5, 0, 13.5, 10, .8, 1]], np.float32)
    _, owners, added = supervision_targets(gt, np.array([1]), reference, np.full((3, 4), 10.),
                                          np.array([True, True, False]), np.full(3, -1), np.zeros((3, 4), np.float32),
                                          reference, (640, 640), "isolation")
    for arm in owners:
        np.testing.assert_array_equal(owners[arm], -1)
        assert not added[arm].any()


def test_primary_policy_rejects_tiny_regression_despite_absolute_AP_gain():
    import json

    from sdcl.config import project_path
    from sdcl.training.unowned_supervision import REGISTRATION, decide

    policy = json.loads(project_path(REGISTRATION).read_text())["policy"]
    values = {"regret": .4, "original": .402, "naive": .401, "random_count": .401, "unique": .403}
    metrics = {a: {g: {"ap50_95": x} for g in ("tiny", "small_all", "all")} for a, x in values.items()}
    metrics["unique"]["tiny"]["ap50_95"] = .4019
    budget = {a: [{"delta_recall_pp": {"small_all": 0.}}] for a in ("regret", "original")}
    decision = decide(metrics, budget, policy)
    assert decision["checks"]["tiny_vs_regret"]
    assert not decision["checks"]["tiny_vs_original"]
    assert not decision["passed"]
