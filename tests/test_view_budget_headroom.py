import numpy as np

from sdcl.evaluation.view_budget_headroom import subset_inventory


def test_safe_subset_preserves_both_identity_support_and_unique_capacity():
    # Global's single box covers both GT but can match only one.
    # Local1 provides the second unique match; local2 is redundant.
    edges = np.array([[True, True, True], [True, False, False]])
    chosen, inventory = subset_inventory(
        {.5: edges, .75: edges}, np.array([0, 1, 2]), np.array([100, 100]))
    assert inventory[0]["metrics"]["0.5"]["lost_supported_gt"]["all"] == 0
    assert inventory[0]["metrics"]["0.5"]["all_max_matches"] == 1
    assert not inventory[0]["safe"]
    assert inventory[chosen]["safe"]
    assert inventory[chosen]["views"] == [0, 1]
