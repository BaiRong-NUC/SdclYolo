import numpy as np

from sdcl.postprocess.native_spatial_field import control_features, field_features, neighboring_indices


def test_grid_edges_cannot_wrap_or_cross_pyramid_levels():
    neighbors = neighboring_indices(np.array([0, 119, 14399, 14400, 17999, 18000, 18899]))
    np.testing.assert_array_equal(neighbors[0], [-1, -1, -1, -1, 1, -1, 120, 121])
    for row, (low, high) in zip(neighbors, [(0, 14400)]*3+[(14400, 18000)]*2+[(18000, 18900)]*2):
        assert np.all((row == -1) | ((row >= low) & (row < high)))
    assert neighbors[1, 4] == -1
    assert neighbors[2, 4] == -1


def test_directional_coordinate_difference_uses_original_scale_and_no_labels():
    boxes = np.tile([10., 20, 18, 28], (18900, 1))
    prob = np.zeros((18900, 10))
    prob[:, 3] = .8
    boxes[121] += [1, -2, 1, -2]
    slots, neighbors = field_features(boxes, prob, np.array([0]), np.array([3]), np.array([[8., 8, 8, 8]]))
    np.testing.assert_allclose(slots[0, 7, :4], [.125, -.25, .125, -.25])
    np.testing.assert_allclose(slots[0, 7, 4:6], [.8, 1])
    assert slots[0, 7, 7] == 1
    np.testing.assert_array_equal(slots[0, neighbors[0] < 0], 0)
    assert slots.shape == (1, 8, 8)


def test_shuffle_preserves_slot_values_but_not_direction_and_mean_excludes_missing():
    slots = np.arange(128, dtype=np.float32).reshape(2, 8, 8)
    slots[:, :, 7] = 1
    slots[0, 0] = 0
    features, order = control_features(slots, "fixed")
    np.testing.assert_array_equal(control_features(slots, "fixed")[1], order)
    np.testing.assert_array_equal(np.sort(order, axis=1), np.tile(np.arange(8), (2, 1)))
    np.testing.assert_array_equal(features["shuffled"].reshape(2, 8, 8),
                                  np.take_along_axis(slots, order[:, :, None], axis=1))
    assert not np.array_equal(features["ordered"], features["shuffled"])
    np.testing.assert_allclose(features["mean"][0].reshape(8, 8)[0], slots[0].sum(0)/7)


def test_gate_cannot_pass_on_regret_gain_without_spatial_order_contribution():
    import json

    from sdcl.config import project_path
    from sdcl.training.native_spatial_field import REGISTRATION, decide

    policy = json.loads(project_path(REGISTRATION).read_text())["policy"]
    scores = {"regret": .4, "geometry": .401, "mean": .401, "shuffled": .402, "ordered": .4015}
    metrics = {a: {g: {"ap50_95": value} for g in ("tiny", "small_all", "all")} for a, value in scores.items()}
    decision = decide(metrics, [{"delta_recall_pp": {"small_all": 0.}}], policy)
    assert decision["checks"]["tiny_vs_regret"]
    assert not decision["checks"]["small_all_vs_shuffled"]
    assert not decision["passed"]
