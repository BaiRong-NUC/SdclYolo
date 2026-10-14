import numpy as np
import torch

from sdcl.training.crossview_quality_rank import Projection, make_pairs, precision_audit


def example():
    # Two objects, two views, two different quality/score representatives per view.
    return dict(image=np.repeat([0, 1], 4), owner=np.zeros(8, int),
                view=np.tile([0, 0, 1, 1], 2), levels=np.zeros(8, int),
                classes=np.zeros(8, int), quality=np.tile([.9, .5, .85, .45], 2),
                true_score=np.tile([.2, .8, .25, .7], 2))


def test_pair_counts_identity_orientation_and_quality_matching():
    d = example()
    p = make_pairs(d)
    assert len(p["cross"]) > 0
    assert len(p["cross"]) == len(p["within"]) == len(p["unlinked"])
    for arm, pairs in p.items():
        hi, lo = pairs.T
        assert np.all(d["quality"][hi]-d["quality"][lo] >= .1)
        assert np.all(d["classes"][hi] == d["classes"][lo])
        if arm == "within":
            assert np.all(d["view"][hi] == d["view"][lo])
            assert np.all(d["image"][hi] == d["image"][lo])
        else:
            assert np.all(d["view"][hi] != d["view"][lo])
            assert np.all((d["image"][hi] == d["image"][lo]) == (arm == "cross"))
    np.testing.assert_array_equal(p["cross"][:, 0], p["unlinked"][:, 0])
    assert np.all(abs(d["quality"][p["cross"][:, 1]]-d["quality"][p["unlinked"][:, 1]]) <= .025)


def test_no_other_object_drops_all_arms_without_fabricating_control():
    d = {k: v[:4] for k, v in example().items()}
    assert all(v.shape == (0, 2) for v in make_pairs(d).values())


def test_no_within_view_quality_gap_drops_cross_comparisons_for_count_match():
    d = example()
    d["quality"] = np.tile([.9, .9, .45, .45], 2)
    assert all(len(v) == 0 for v in make_pairs(d).values())


def test_pair_sampling_is_deterministic_and_not_driven_by_row_order_rng_state():
    d = example()
    first = make_pairs(d)
    np.random.seed(193)
    np.random.random(100)
    second = make_pairs(d)
    for arm in first:
        np.testing.assert_array_equal(first[arm], second[arm])


def test_actual_projection_forward_and_gradients_include_level_biases():
    m = Projection(np.zeros((3, 10, 128)), np.zeros((3, 10)))
    x = torch.ones((3, 128))
    levels = torch.arange(3)
    z = m(x, levels)
    (z[0, 0]-z[2, 0]).backward()
    assert m.bias.grad[0, 0] == 1 and m.bias.grad[2, 0] == -1
    assert m.weight.grad[1].abs().sum() == 0


def test_prospective_precision_audit_with_nonzero_features_all_levels():
    rng = np.random.default_rng(17)
    m = Projection(rng.normal(0, .05, (3, 10, 128)), rng.normal(0, .1, (3, 10)))
    d = dict(features=rng.normal(size=(24, 128)).astype(np.float16),
             levels=np.tile(np.arange(3), 8), ordinary=rng.uniform(size=(24, 10)).astype(np.float32))
    audit = precision_audit(m, d, "cpu")
    assert audit["passed"] and not audit["cudnn_allow_tf32"] and not audit["matmul_allow_tf32"]
