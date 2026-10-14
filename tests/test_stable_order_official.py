from sdcl.evaluation.stable_order_official import ARMS, decide


POLICY = {
    "mean_stable_AP_vs_uniform_min_pp": .10,
    "mean_stable_AP_vs_regret_min_exclusive_pp": 0.,
    "mean_stable_AP_vs_nominal_min_exclusive_pp": 0.,
    "mean_stable_AP_vs_thinned_min_exclusive_pp": 0.,
    "positive_vs_both_matched_seeds_min": 2,
    "mean_stable_AR500_vs_uniform_max_decrease_pp": .10,
}


def test_positive_mean_cannot_hide_failure_to_beat_matched_seeds_or_recall_budget():
    metrics = {arm: {"AP": 30., "AR500": 45.} for arm in ARMS}
    metrics["regret"]["AP"] = 30.05
    for seed, ap in enumerate([30.11, 30.12, 30.13]):
        metrics[f"stable_order_seed{seed}"]["AP"] = ap
    assert decide(metrics, POLICY)["decision"] == "WIDER_TRAIN_ONLY_PILOT_ALLOWED"
    for seed in (0, 1):
        metrics[f"nominal_order_seed{seed}"]["AP"] = 30.2
    result = decide(metrics, POLICY)
    assert not result["checks"]["positive_matched_seeds"]
    assert result["decision"] == "STOP_THIS_STABLE_ORDER_OFFICIAL_RECIPE"
    for seed in range(3):
        metrics[f"stable_order_seed{seed}"]["AR500"] = 44.8
    assert not decide(metrics, POLICY)["checks"]["AR500_budget"]
