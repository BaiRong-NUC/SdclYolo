import numpy as np
import torch

from sdcl.postprocess.frozen_roi_features import letterbox_geometry, sample_context


def test_letterbox_uses_actual_integer_padding_and_original_scale():
    ratio, left, top, width, height = letterbox_geometry([101, 200])
    assert ratio == 4.8
    assert (left, top, width, height) == (237, 0, 485, 960)


def test_sampling_keeps_box_context_axes_channels_and_action_alignment():
    feature = torch.stack([torch.full((120, 120), float(i)) for i in range(16)])[None]
    boxes = np.array([[[10, 20, 30, 40], [12, 22, 32, 42]]], dtype=np.float32)
    result = sample_context(feature, boxes, [960, 960])
    assert result.shape == (1, 2, 288)
    np.testing.assert_allclose(result[0, 0].reshape(18, 16),
                               np.broadcast_to(np.arange(16), (18, 16)), atol=1e-5)
    np.testing.assert_allclose(result[0, 0], result[0, 1], atol=1e-5)
