import numpy as np

from sdcl.evaluation.scale_gradient_conflict import decide, gradient_statistics, shuffled_membership, summarize


def test_shuffle_preserves_class_counts_without_mutating_original():
    classes = np.repeat([0, 1], 20)
    tiny = np.tile(np.arange(20) % 2 == 0, 2)
    original = tiny.copy()
    control = shuffled_membership("fixed", classes, tiny)
    np.testing.assert_array_equal(original, tiny)
    np.testing.assert_array_equal(control, shuffled_membership("fixed", classes, tiny))
    for cls in (0, 1):
        assert control[classes == cls].sum() == tiny[classes == cls].sum()
    assert np.any(control != tiny)


def test_blocking_is_total_descent_direction_not_merely_negative_cosine():
    a = gradient_statistics([.8, 0], [1, 0], [.5, .1])
    assert a["actual"]["cosine"] == -1 and not a["actual"]["blocked"]
    b = gradient_statistics([-1, 0], [1, 0], [.5, .1])
    assert b["actual"]["blocked"] and b["actual"]["opposition_ratio"] == 2
    assert not gradient_statistics([1, 0], [0, 0], [.5, .1])["valid"]


def test_class_preserving_null_cannot_pass_by_generic_blocking():
    traces = [dict(changed_productive_gt=2, gradients=gradient_statistics([-1, 0], [1, 0], [1, 0]))
              for _ in range(30)]
    metrics = summarize(traces)
    assert metrics["actual_blocked_fraction"] == 1
    assert metrics["mean_control_minus_actual_cosine"] == 0
    policy = dict(valid_images_min=24, changed_control_images_min=20, changed_productive_gt_min=50,
                  actual_blocked_fraction_min=.1, mean_control_minus_actual_cosine_min=.02,
                  bootstrap_lower_control_minus_actual_cosine_min=0.)
    assert decide(metrics, policy)["decision"] == "STOP_THIS_SCALE_SPECIFIC_P3_GRADIENT_ROUTE"


def test_tiny_and_remainder_gradients_reconstruct_total_with_frozen_denominator():
    import torch
    x = torch.tensor([1., -2.], requires_grad=True)
    tiny, rest = (x[0]-3).square()/7, 2*(x[0]+x[1]).square()/7
    total_gradient = torch.autograd.grad(tiny+rest, x, retain_graph=True)[0]
    tiny_gradient = torch.autograd.grad(tiny, x, retain_graph=True)[0]
    rest_gradient = torch.autograd.grad(rest, x)[0]
    torch.testing.assert_close(total_gradient, tiny_gradient+rest_gradient)
