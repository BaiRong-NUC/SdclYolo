from dataclasses import asdict

import numpy as np
import pytest
import torch

from sdcl.evaluation.fusion_screen import soft_merge_predictions
from sdcl.postprocess.rescue import (
    RescueRecipe, RescueScorer, candidate_features, feature_layout, load_recipe,
    normalized_features, opportunity, rescue_labels, sample_candidates, scene_holdout,
    score_correction, soft_nms_with_origins, variant_labels,
)
from sdcl.evaluation.rescue_screen import decide_rescue
from sdcl.evaluation.slicing_screen import tile_windows


def test_origin_tracking_reproduces_strong_soft_nms_with_ignore_and_final_cap():
    generator = np.random.default_rng(4)
    views = []
    for _ in range(5):
        low = generator.uniform(0, 80, (160, 2))
        high = low + generator.uniform(2, 15, (160, 2))
        views.append(np.column_stack([low, high, generator.uniform(0.001, 0.95, 160),
                                      generator.integers(0, 3, 160)]))
    ignores = [[0, 0, 20, 20]]
    expected, _ = soft_merge_predictions(views, (100, 100), ignores)
    actual, origins, raw_scores = soft_nms_with_origins(views, (100, 100), ignores)
    np.testing.assert_array_equal(actual, expected)
    assert len(actual) <= 500
    assert len(origins) == len(raw_scores) == len(actual)
    assert np.all((origins >= 0) & (origins < 5))
    assert np.all(raw_scores >= actual[:, 4])


def fixture(side=5):
    record = {"stem": "scene_frame", "size": [1000, 1000], "objects": [
        {"object_id": 0, "class_id": 0, "box_xyxy": [100, 100, 100 + side, 100 + side]}]}
    windows = [[0, 0, 1000, 1000], *tile_windows(1000, 1000)]
    empty = np.empty((0, 6))
    candidate = np.array([[100, 100, 100 + side, 100 + side, 0.8, 0]])
    views = [empty, candidate.copy(), empty, empty, empty]
    return record, windows, candidate, np.array([1]), views


def test_labels_distinguish_missing_observable_peer_from_sole_observer():
    record, windows, candidate, origins, views = fixture(5)
    assert rescue_labels(record, candidate, origins, views, windows, []).tolist() == [3]
    record, windows, candidate, origins, views = fixture(10)
    assert rescue_labels(record, candidate, origins, views, windows, []).tolist() == [2]
    views[0] = candidate.copy()
    views[0][:, 4] = 0.3
    assert rescue_labels(record, candidate, origins, views, windows, []).tolist() == [1]


def test_observed_detection_overrides_geometric_opportunity_proxy():
    record, windows, candidate, origins, views = fixture(5)
    assert not opportunity(candidate[:, :4], windows[0])[0][0]
    views[0] = candidate.copy()
    assert rescue_labels(record, candidate, origins, views, windows, []).tolist() == [1]


def test_duplicate_false_positive_does_not_get_supported_positive_label():
    record, windows, candidate, origins, views = fixture(10)
    candidates = np.vstack([candidate, candidate])
    candidates[1, 4] = 0.2
    labels = rescue_labels(record, candidates, np.array([1, 1]), views, windows, [])
    assert labels.tolist() == [2, 0]


def test_opportunity_handles_crop_coverage_and_small_projected_scale():
    boxes = [[100, 100, 110, 110], [550, 100, 570, 110], [100, 100, 102, 102]]
    available, coverage, _ = opportunity(boxes, [0, 0, 556, 556])
    assert available.tolist() == [True, False, False]
    assert coverage[1] == pytest.approx(0.3)


def test_features_are_finite_predict_only_and_ablation_removes_all_opportunity_columns():
    record, windows, candidate, origins, views = fixture()
    features = candidate_features(candidate, origins, candidate[:, 4],
                                  views, windows, record["size"], [])
    names, removed = feature_layout()
    assert features.shape == (1, len(names))
    assert np.isfinite(features).all()
    # Origin view contributes no peer evidence, including its own matching prediction.
    offset = names.index("peer_1_opportunity")
    assert (features[:, offset:offset + 6] == 0).all()
    masked = normalized_features(features, np.zeros(len(names)), np.ones(len(names)),
                                 "no_observability")
    assert (masked[:, removed] == 0).all()
    preserved = sorted(set(range(len(names))) - set(removed))
    np.testing.assert_allclose(masked[:, preserved], features[:, preserved].clip(-6, 6))
    empty = candidate_features(np.empty((0, 6)), np.empty(0, dtype=int), np.empty(0),
                               views, windows, record["size"], [])
    assert empty.shape == (0, len(names))


def test_sampler_is_deterministic_and_importance_weights_restore_state_counts():
    labels = np.array([0] * 200 + [1] * 100 + [2] * 4 + [3] * 2)
    indices, weights = sample_candidates(labels, "scene_frame")
    again, same_weights = sample_candidates(labels, "scene_frame")
    np.testing.assert_array_equal(indices, again)
    np.testing.assert_array_equal(weights, same_weights)
    assert len(indices) == 32 + 32 + 4 + 2
    for state in range(4):
        assert weights[labels[indices] == state].sum() == pytest.approx((labels == state).sum())
    assert scene_holdout("0000020_00001") == scene_holdout("0000020_00999")


def test_variant_targets_and_score_correction_preserve_boxes():
    labels = np.arange(4)
    assert variant_labels(labels, "calibration").tolist() == [0, 1, 1, 1]
    assert variant_labels(labels, "no_observability").tolist() == [0, 1, 2, 2]
    np.testing.assert_array_equal(labels, np.arange(4))
    candidates = np.array([[0, 0, 5, 5, 0.5, 0], [10, 10, 15, 15, 0.5, 0]])
    result = score_correction(candidates, np.array([[0, 0, 1, 0], [1, 0, 0, 0]]), "rescue")
    assert result[0, 4] > 0.5 > result[1, 4]
    np.testing.assert_array_equal(result[:, :4], candidates[:, :4])
    assert candidates[:, 4].tolist() == [0.5, 0.5]


def test_recipe_and_finite_backward_for_all_variants():
    assert asdict(load_recipe()) == asdict(RescueRecipe())
    torch.manual_seed(0)
    model = RescueScorer()
    logits = model(torch.randn(16, len(feature_layout()[0])))
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(16) % 4)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def passing_reports():
    def comparison(delta, low=0.1):
        return {"label": "fppi_10", "delta_recall_pp": {"small_all": delta},
                "paired_image_bootstrap_delta_recall_95ci_pp": {"small_all": [low, 1.2]}}
    report = {
        "matched_fp_budgets": [comparison(0.7), {
            "label": "fppi_5", "delta_recall_pp": {"small_all": 0.0}}],
        "coco_reference": {
            "baseline": {"tiny": {"ap50_95": 0.25}, "all": {"ap50_95": 0.32}},
            "sdcl": {"tiny": {"ap50_95": 0.255}, "all": {"ap50_95": 0.33}},
        },
    }
    return report, {"calibration": [comparison(0.5)], "no_observability": [comparison(0.3)]}


def test_screen_requires_strong_baseline_and_both_learned_controls():
    report, controls = passing_reports()
    assert decide_rescue(report, controls, True)["decision"] == "CONTINUE"
    controls["calibration"][0]["delta_recall_pp"]["small_all"] = 0.29
    assert decide_rescue(report, controls, True)["decision"] == "STOP"
    assert decide_rescue(report, controls, False)["decision"] == "SMOKE_ONLY"


def test_screen_rejects_zero_interval_and_stricter_budget_regression():
    report, controls = passing_reports()
    controls["no_observability"][0]["paired_image_bootstrap_delta_recall_95ci_pp"]["small_all"][0] = 0
    assert decide_rescue(report, controls, True)["decision"] == "STOP"
    report, controls = passing_reports()
    report["matched_fp_budgets"][1]["delta_recall_pp"]["small_all"] = -0.11
    assert decide_rescue(report, controls, True)["decision"] == "STOP"
