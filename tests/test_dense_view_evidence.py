import numpy as np

from sdcl.postprocess.dense_view_evidence import dense_features, map_anchor_centers


def test_dense_summary_keeps_weak_nonargmax_evidence_and_excludes_padding():
    probability = np.zeros((2, 10), np.float32)
    probability[0, 0], probability[0, 1] = .2, .01
    probability[1, 1] = .99
    centers = np.array([[10, 10], [-1, 10]])
    windows = [[0, 0, 100, 100]] * 5
    features = dense_features(probability, centers, np.array([0, 0]), windows, [100, 100])
    # Four stats per class; class1 has weak probability without argmax support.
    np.testing.assert_allclose(features[:, 4], .01)
    assert not (features == .99).any()


def test_anchor_inverse_letterbox_mapping_restores_original_point():
    np.testing.assert_allclose(map_anchor_centers(np.array([[48., 288.]]), [400, 200]), [[20, 20]])
