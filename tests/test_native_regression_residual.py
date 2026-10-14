import numpy as np
import torch

from sdcl.postprocess.native_regression_residual import NativeResidual, correct_coordinates
from sdcl.training.native_regression_residual import associate_native


def test_initial_network_exactly_preserves_regret_and_has_matched_parameters():
    model = NativeResidual()
    assert sum(p.numel() for p in model.parameters()) == 4388
    delta = model(torch.randn(3, 132)).detach().numpy()
    reference = np.array([[1, 2, 5, 6, .7, 1]]*3, np.float32)
    actual = correct_coordinates(reference, delta, np.ones((3, 4)), np.ones(3, bool), (100, 80))
    np.testing.assert_array_equal(actual, reference)


def test_bounded_readout_preserves_score_class_inactive_and_invalid_rows():
    reference = np.array([[1, 2, 5, 6, .7, 1], [1, 2, 5, 6, .8, 2],
                          [1, 2, 1.01, 6, .9, 3]], np.float32)
    delta = np.array([[.1, -.1, .1, -.1]]*3)
    delta[2, 2] = -.1
    actual = correct_coordinates(reference, delta, np.ones((3, 4)), np.array([True, False, True]), (100, 80))
    np.testing.assert_array_equal(actual[:, 4:], reference[:, 4:])
    np.testing.assert_array_equal(actual[1:], reference[1:])
    np.testing.assert_allclose(actual[0, :4], [1.1, 1.9, 5.1, 5.9])


def test_native_association_keeps_class_threshold_and_cross_level_feature_index():
    from types import SimpleNamespace

    classification = {k: torch.full((1, 128, h, h), 10.+k)
                      for k, h in enumerate((120, 60, 30))}
    regression = {k: torch.full((1, 64, h, h), float(k))
                  for k, h in enumerate((120, 60, 30))}
    decoded = torch.zeros(1, 14, 18900)
    for row, cls, score in ((14399, 2, .9), (14400, 1, .7), (18000, 1, .8)):
        decoded[0, :4, row] = torch.tensor([10., 10, 8, 8])
        decoded[0, 4+cls, row] = score
    extractor = SimpleNamespace(
        decoded=decoded, features=classification,
        raw={"feats": [torch.zeros(1, 1, 120, 120)]},
        head=SimpleNamespace(stride=torch.tensor([8, 16, 32])),
        predict=lambda image: ({"control": np.zeros((0, 6))}, {}))
    reference = np.array([[6, 6, 14, 14, .9, 1], [6, 6, 14, 14, .04, 1]])
    _, quality, chosen, reg, cls = associate_native(
        extractor, regression, np.zeros((960, 960, 3), np.uint8), reference, (0, 0, 960, 960))
    np.testing.assert_array_equal(chosen, [18000, -1])
    np.testing.assert_allclose(quality, [.8, 0])
    np.testing.assert_array_equal(reg[0], np.full(64, 2))
    np.testing.assert_array_equal(cls[0], np.full(64, 12))
    np.testing.assert_array_equal(reg[1], np.zeros(64))
