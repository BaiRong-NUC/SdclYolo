import numpy as np

from sdcl.evaluation.exclusive_peer_support import allocate, peer_graph, shared_pairs


def test_global_allocation_preserves_alternative_support_that_greedy_wastes():
    weights = np.array([[.9, .8], [.85, 0], [0, 0]])
    best, optimal, greedy = allocate(weights)
    np.testing.assert_array_equal(best, [0, 0, -1])
    np.testing.assert_array_equal(optimal, [1, 0, -1])
    np.testing.assert_array_equal(greedy, [0, -1, -1])
    assert weights[np.arange(2), optimal[:2]].sum() > weights[0, greedy[0]]


def test_allocation_supports_empty_rectangular_and_deterministic_ties():
    assert all(np.array_equal(x, [-1, -1]) for x in allocate(np.zeros((2, 0))))
    assert all(len(x) == 0 for x in allocate(np.zeros((0, 3))))
    _, optimal, greedy = allocate(np.ones((3, 1)))
    assert (optimal >= 0).sum() == (greedy >= 0).sum() == 1
    np.testing.assert_array_equal(greedy, [0, -1, -1])
    np.testing.assert_array_equal(allocate(np.ones((3, 1)))[1], optimal)


def test_peer_inventory_preserves_ids_and_excludes_self_view_and_wrong_class():
    prediction = np.array([[0, 0, 10, 10, .8, 0], [0, 0, 10, 10, .7, 0],
                           [0, 0, 10, 10, .2, 0]])
    raw = np.array([[0, 0, 10, 10, .01, 0], [0, 0, 10, 10, .9, 1],
                    [0, 0, 10, 10, .8, 0]])
    graph = peer_graph(prediction, np.array([0, 1, 0]), [raw, raw, [], [], []])
    np.testing.assert_array_equal(graph["rows"], [0, 1])
    np.testing.assert_array_equal(graph["best_ids"][:, :2], [[-1, 2], [2, -1]])
    np.testing.assert_array_equal(graph["exclusive_retention"], [1, 1])


def test_pairing_requires_same_observation_same_view_and_gives_query_weight_one():
    ids = np.array([[3, -1], [3, -1], [-1, 3], [3, -1], [3, -1]])
    positive, negative, weight, shared = shared_pairs(
        ids, np.array([0, 0, 0, 1, 0]), np.array([True, False, False, False, False]),
        np.ones(5, bool), np.ones((5, 5)))
    np.testing.assert_array_equal(positive, [0, 0])
    np.testing.assert_array_equal(negative, [1, 4])
    assert weight.sum() == 1
    assert not shared[0, 2] and not shared[0, 3]
