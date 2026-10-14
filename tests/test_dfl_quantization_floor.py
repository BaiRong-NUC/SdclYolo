import numpy as np

from sdcl.postprocess.dfl_quantization_floor import distribution_reliability, original_coordinates


def test_adjacent_bin_ideal_has_nonzero_variance_but_zero_excess_at_any_phase():
    p = np.zeros((3, 4, 16))
    for i, fraction in enumerate((.1, .5, .9)):
        p[i, :, 3], p[i, :, 4] = 1-fraction, fraction
    result = distribution_reliability(p)
    assert np.all(result["variance"] > 0)
    np.testing.assert_allclose(result["excess"], 0, atol=1e-14)
    np.testing.assert_allclose(result["variance"], result["floor"], atol=1e-14)
    np.testing.assert_allclose(result["mean"][:, 0], (3.1, 3.5, 3.9))


def test_nonlocal_spread_survives_subtraction_and_integer_shift():
    p = np.zeros((2, 4, 16))
    p[0, :, 2], p[0, :, 4] = .5, .5
    p[1, :, 5], p[1, :, 7] = .5, .5
    result = distribution_reliability(p)
    np.testing.assert_allclose(result["excess"], 1)
    np.testing.assert_allclose(result["floor"], 0)
    assert result["excess_variance"][0] < result["excess_variance"][1]


def test_original_coordinates_undoes_letterbox_then_clips_before_tile_offset():
    mean = np.array([[1., 2, 3, 4]])
    anchors, stride = np.array([[10., 20]]), np.array([8.])
    boxes = original_coordinates(mean, anchors, stride, (100, 200, 740, 520))
    # gain1.5, y padding240; clipped local ylow/yhigh both zero.
    np.testing.assert_allclose(boxes, [[148, 200, 508/3, 200]])
