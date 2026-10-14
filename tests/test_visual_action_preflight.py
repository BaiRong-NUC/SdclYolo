import numpy as np

from sdcl.postprocess.localization import action_feature_names
from sdcl.postprocess.visual_action_features import recover_cached_boxes, visual_features
from sdcl.training.visual_action_preflight import pilot_decision


def test_coordinate_inversion_retains_rectangular_scale_and_offsets():
    names = {n: i for i, n in enumerate(action_feature_names())}
    x = np.zeros((1, 6, len(names)), dtype=np.float32)
    x[:, :, names["log_reference_side"]] = np.log1p(np.sqrt(20 * 40) / 16)
    x[:, :, names["log_aspect"]] = np.log(.5)
    x[:, :, names["center_x"]] = .5
    x[:, :, names["center_y"]] = .25
    x[:, :, names["iou_anchor"]] = 1
    recovered = recover_cached_boxes(x, [640, 640])
    np.testing.assert_allclose(recovered[0, 0], [310, 140, 330, 180], atol=1e-4)
    np.testing.assert_allclose(recovered[0, 0], recovered[0, 5])


def test_boundary_features_distinguish_actual_edge_from_interior_without_gt():
    image = np.zeros((80, 80, 3), np.uint8)
    image[20:60, 20:60] = 255
    boxes = np.array([[20, 20, 60, 60], [30, 30, 50, 50]], float)
    features = visual_features(image, boxes)
    assert features.shape == (2, 24)
    assert np.isfinite(features).all()
    assert features[0, [0, 4, 8, 12]].sum() > features[1, [0, 4, 8, 12]].sum()


def test_visual_effect_is_compared_to_geometry_ranking_not_weaker_regression():
    def values(iou, hit, harm):
        return {"owned_mean_iou_percent": iou, "owned_ten_threshold_percent": hit,
                "owned_harm_rate_percent": harm}
    metrics = {
        "geometry_absolute": values(50, 10, 5), "visual_absolute": values(52, 11, 5),
        "geometry_ranking": values(55, 12, 3), "visual_ranking": values(54, 11.5, 3),
        "frozen_uniform": values(56, 13, 3),
    }
    policy = {"visual_ranking_vs_geometry_ranking_owned_ten_threshold_min_pp": .05,
              "visual_ranking_vs_geometry_ranking_owned_mean_iou_min_exclusive_pp": 0.,
              "visual_ranking_vs_geometry_ranking_harm_rate_max_increase_pp": .5}
    assert pilot_decision(metrics, policy)["decision"] == "STOP_THIS_VISUAL_FEATURE_PILOT"
