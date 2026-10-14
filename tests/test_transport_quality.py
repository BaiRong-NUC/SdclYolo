import numpy as np
from scipy.stats import wasserstein_distance
import torch

from sdcl.postprocess.transport_quality import (
    QualityHead, WIDTH, active_indices, adjusted_decoded, boundary_support, quality_features, wasserstein1,
)


def test_discrete_transport_matches_scipy_with_duplicate_descending_support():
    rng = np.random.default_rng(5)
    x, y = rng.integers(-2, 5, (2, 10, 16))
    p, q = rng.random((2, 10, 16))
    actual = wasserstein1(x, p, y, q)
    expected = [wasserstein_distance(a, b, c, d) for a, b, c, d in zip(x, y, p, q)]
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_physical_transport_detects_anchor_translation_removed_by_centering():
    p = np.array([[0., .25, .75]])
    x = np.array([[0., 1., 2.]])
    y = x+3
    np.testing.assert_allclose(wasserstein1(x, p, y, p), [3.])
    np.testing.assert_allclose(wasserstein1(x-(x*p).sum(-1, keepdims=True), p,
                                        y-(y*p).sum(-1, keepdims=True), p), [0.])


def test_common_support_respects_crop_clipping_and_offset():
    x = boundary_support(np.array([[.5, .5]]), np.array([8.]), [10, 20, 970, 980])
    np.testing.assert_array_equal(x[0, 0, :2], [14, 10])
    np.testing.assert_array_equal(x[0, 2, :2], [14, 22])
    assert x.min() >= 10 and x.max() <= 980


def test_zero_shift_replays_half_tensor_and_bounded_shift_keeps_coordinates():
    values = torch.zeros((1, 14, 18900), dtype=torch.float16)
    values[:, :4] = 3
    values[:, 4:] = .1
    assert torch.equal(values, adjusted_decoded(values, [0, 3], np.zeros(2)))
    changed = adjusted_decoded(values, [0, 3], np.array([1., -1.]))
    assert torch.equal(changed[:, :4], values[:, :4])
    assert changed[0, 4, 0] > values[0, 4, 0] > changed[0, 4, 3]
    assert torch.equal(changed[:, :, 5:], values[:, :, 5:])


def test_equal_capacity_zero_initial_head_and_stable_candidate_cap():
    head = QualityHead()
    assert sum(p.numel() for p in head.parameters()) == 5313
    assert torch.count_nonzero(head(torch.randn(10, WIDTH))) == 0
    decoded = np.full((14, 18900), .1, np.float16)
    np.testing.assert_array_equal(active_indices(decoded), np.arange(4096))


def test_fixed_features_share_geometry_and_mask_grid_boundary_neighbors():
    decoded = np.zeros((14, 18900), np.float16)
    decoded[4] = .5
    boxes = np.tile([4., 4., 12., 12.], (18900, 1)).astype(np.float32)
    result = quality_features(decoded, np.zeros((18900, 4, 16), np.float16),
                              np.zeros((18900, 10), np.float16), boxes,
                              np.ones((18900, 2)), np.full(18900, 8.),
                              [0, 0, 960, 960], [960, 960], [0, 121])
    assert all(x.shape == (2, 164) for x in result.values())
    np.testing.assert_array_equal(result["geometry"][:, :92], result["transport"][:, :92])
    np.testing.assert_array_equal(result["dgqp"][:, :132], result["transport"][:, :132])
    assert np.count_nonzero(result["geometry"][:, 92:]) == 0
    assert result["transport"][0, 28:60].sum() == 0
