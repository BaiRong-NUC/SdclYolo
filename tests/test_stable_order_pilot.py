import numpy as np
import torch

from sdcl.training.stable_order_pilot import objective, pair_masks


def test_random_thinning_matches_each_anchor_count_and_stable_rejects_reversal():
    ious = np.full((3, 17, 6), .5)
    ious[:, :, 1] = .8
    ious[0, 1, 1] = .4
    valid = np.zeros((3, 6), bool)
    valid[:, :2] = True
    signs, masks = pair_masks(ious, valid, np.array([True, True, False]), 20)
    assert signs.shape == (3, 15)
    assert masks["nominal_order"].sum() == 2
    assert masks["stable_order"].sum() == 1
    np.testing.assert_array_equal(masks["stable_order"].sum(1),
                                  masks["random_thinned_order"].sum(1))
    assert not masks["stable_order"][0].any()
    assert not masks["nominal_order"][2].any()


def test_pair_direction_gradient_and_common_unowned_keep_pressure():
    outputs = torch.zeros((2, 6), requires_grad=True)
    signs = torch.zeros((2, 15))
    signs[0, 0] = -1
    masks = torch.zeros((2, 15), dtype=torch.bool)
    masks[0, 0] = True
    valid = torch.zeros((2, 6), dtype=torch.bool)
    valid[:, :2] = True
    loss = objective(outputs, signs, masks, valid, torch.tensor([True, False]),
                     torch.ones(2, dtype=torch.bool))
    loss.backward()
    assert outputs.grad[0, 1] < 0
    assert outputs.grad[0, 0] > 0
    assert outputs.grad[1, 1] > 0
    assert outputs.grad[1, 0] < 0
    assert torch.all(outputs.grad[:, 2:] == 0)
