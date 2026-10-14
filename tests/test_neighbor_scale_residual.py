import numpy as np
import torch

from sdcl.postprocess.neighbor_scale_residual import ScaleResidual, derange_blocks, objective


def test_zero_residual_and_context_derangement_are_exact():
    values = np.arange(5*6*4).reshape(5, 6, 4)
    shuffled, partner = derange_blocks(values, 1331)
    assert (partner != np.arange(5)).all()
    np.testing.assert_array_equal(shuffled, values[partner])
    model = ScaleResidual()
    assert torch.equal(model(torch.randn(10, 6, 72)), torch.zeros(10, 6))


def test_unowned_keep_reference_can_be_nonzero_action():
    utility = torch.tensor([[0., 0., 0.]], requires_grad=True)
    loss = objective(utility, torch.zeros_like(utility), torch.ones_like(utility, dtype=torch.bool),
                     torch.tensor([False]), torch.tensor([2]), torch.tensor([[0, 1], [0, 2], [1, 2]]))
    loss.backward()
    assert utility.grad[0, 2] < 0
    assert (utility.grad[0, :2] > 0).all()
