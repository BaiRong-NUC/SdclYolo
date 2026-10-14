import numpy as np

from sdcl.evaluation.identity_preflight import classify_drift, decide, threshold_recovery


def test_drift_requires_same_class_and_original_nearest_identity():
    old = np.array([[.8, .5, .8], [.3, .7, .3], [.9, .9, .9]])
    new = np.array([[.4, .4, .4], [.7, .8, .3], [.95, .95, .95]])
    _, _, harm, suspect, strict = classify_drift(
        old, new, np.array([0, 0, 0]), np.array([0, 0, 1]))
    assert harm.tolist() == [True, True, True]
    assert suspect.tolist() == [True, True, False]
    assert strict.tolist() == [True, False, False]


def test_recovery_is_fixed_owner_threshold_average_not_iou_gain():
    old, new = np.array([.8, .6]), np.array([.6, .8])
    assert threshold_recovery(old, new, np.array([True, False])) == .4
    assert threshold_recovery(old, new, np.array([False, True])) == 0.


def test_primary_arm_cannot_be_replaced_by_better_secondary():
    small = {"strict_drift_count": 9, "strict_drift_fixed_owner_recovery_pp": .04}
    big = {"strict_drift_count": 100, "strict_drift_fixed_owner_recovery_pp": 1.}
    summary = {"uniform": {"all": small, "small_all": small}, "regret": {"all": big}}
    policy = {"primary_strict_drift_count_min": 10,
              "primary_fixed_owner_threshold_recovery_all_or_small_all_min_pp": .05}
    assert decide(summary, policy)["decision"] == "STOP_IDENTITY_REPAIR_RECIPE"
