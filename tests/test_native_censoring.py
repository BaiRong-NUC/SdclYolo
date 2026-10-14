import numpy as np
import torch

from sdcl.evaluation.headroom_screen import max_pool_matches
from sdcl.evaluation.native_censoring import native_inventory, pool_edges


def test_decoded_inventory_keeps_suppressed_anchor_and_exact_native_geometry():
    from ultralytics.utils.nms import non_max_suppression

    decoded = torch.zeros((1, 14, 3))
    decoded[0, :4] = torch.tensor([[100, 100, 300], [100, 100, 300],
                                  [20, 20, 30], [20, 20, 30]])
    decoded[0, 4] = torch.tensor([.9, .8, .7])
    control = non_max_suppression(decoded.clone(), .001, .7, max_det=500)[0].numpy()
    before, after = native_inventory(decoded, [960, 960], [960, 960], control)
    assert len(before) == 3
    assert len(after) == 2
    np.testing.assert_array_equal(after, control)


def test_inventory_edges_respect_class_and_score_and_graph_cannot_reuse_one_box():
    gt = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
    classes = np.array([0, 0])
    pools = np.array([[0, 0, 10, 10, .8, 0], [0, 0, 10, 10, .8, 1],
                      [0, 0, 10, 10, .01, 0]])
    edges = pool_edges(gt, classes, pools, .05, .5)
    assert edges.any(1).sum() == 2
    assert max_pool_matches(edges) == 1
    assert not edges[:, 1:].any()


def test_empty_inventory_preserves_empty_shapes():
    decoded = torch.zeros((1, 14, 1))
    before, after = native_inventory(decoded, [960, 960], [960, 960],
                                    np.empty((0, 6), np.float32))
    assert before.shape == after.shape == (0, 6)
