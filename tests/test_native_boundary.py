import numpy as np
import torch

from sdcl.postprocess.native_boundary import action_differences, PixelResidual, shuffle_differences
from sdcl.postprocess.localization import select_actions


def test_native_pixel_differences_reference_zero_and_color_moments_survive_shuffle():
    image = np.arange(40 * 40 * 3, dtype=np.uint8).reshape(40, 40, 3)
    boxes = np.tile(np.array([[[4, 5, 14, 15]]], np.float32), (1, 6, 1))
    boxes[0, 1, 0] += 1
    differences = action_differences(image, boxes, np.array([0]), "boundary")
    assert not differences[0, 0].any()
    assert differences[0, 1].any()
    shuffled = shuffle_differences(differences, "image", np.array([42]))
    np.testing.assert_allclose(differences.reshape(1, 6, 160, 3).mean(2),
                               shuffled.reshape(1, 6, 160, 3).mean(2), atol=1e-6)
    np.testing.assert_array_equal(np.sort(differences, -1), np.sort(shuffled, -1))
    assert not np.array_equal(differences, shuffled)


def test_zero_residual_preserves_exact_utility_and_frozen_actions():
    model = PixelResidual()
    geometry, pixels = torch.randn(2, 6, 68), torch.randn(2, 6, 480)
    base = np.array([[0, .2, .1, .3, .2, .1], [0, 0, 0, 0, 0, 0]])
    output = model(geometry, pixels).detach().numpy()
    np.testing.assert_array_equal(output, np.zeros_like(output))
    valid = np.ones((2, 6), bool)
    np.testing.assert_array_equal(select_actions(base, valid, "absolute"),
                                  select_actions(base + output, valid, "absolute"))
