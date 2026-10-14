import numpy as np

from sdcl.postprocess.neighbor_scale import scale_context


def test_neighbors_exclude_target_aliases_and_other_classes_and_favor_consistent_shape():
    candidate = np.array([[40, 40, 50, 50, .8, 0]], float)
    actions = np.array([[[40, 40, 50, 50], [38, 38, 52, 52]]], float)
    peers = np.array([
        [40, 40, 50, 50, .9, 0], [39, 39, 51, 51, .9, 0],
        [10, 40, 20, 50, .8, 0], [70, 40, 80, 50, .8, 0],
        [40, 10, 50, 20, .8, 0], [40, 70, 50, 80, .8, 0],
        [55, 55, 56, 56, .9, 1],
    ])
    context = scale_context(candidate, actions, peers, np.array([0]), (200, 200))
    assert context["supported"][0]
    assert set(context["neighbors"][0][context["neighbors"][0] >= 0]) == {2, 3, 4, 5}
    assert context["local_scale"][0, 0] > context["local_scale"][0, 1]


def test_no_nearby_same_class_instances_means_no_context_instead_of_fake_evidence():
    candidate = np.array([[0, 0, 10, 10, .8, 0]], float)
    actions = candidate[:, None, :4]
    context = scale_context(candidate, actions, candidate, np.array([0]), (100, 100))
    assert not context["supported"].any()
    assert (context["neighbors"] == -1).all()
    assert not context["local_scale"].any()
