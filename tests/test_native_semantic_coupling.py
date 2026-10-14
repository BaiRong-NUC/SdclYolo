import numpy as np

from sdcl.evaluation.native_semantic_coupling import conditional_pairs, features, pair_totals, SCORES


def test_coupling_changes_with_pairing_while_marginal_statistics_stay_fixed():
    slots = np.zeros((2, 8, 8))
    slots[:, :4, 7] = 1
    slots[:, :4, 4] = [.1, .3, .6, .9]
    slots[0, :4, 6] = [.1, .3, .6, .9]
    slots[1, :4, 6] = [.9, .6, .3, .1]
    values, count, permutations = features(slots, np.array([.7, .7]), "fixed")
    np.testing.assert_array_equal(count, [4, 4])
    assert values["coupling"][0] > 0 > values["coupling"][1]
    np.testing.assert_allclose(values["marginal_product"][0], values["marginal_product"][1])
    np.testing.assert_array_equal(permutations[:, 4:], np.tile(np.arange(4, 8), (2, 1)))
    for row in range(2):
        np.testing.assert_array_equal(np.sort(permutations[row, :4]), np.arange(4))
    np.testing.assert_array_equal(features(slots, np.array([.7, .7]), "fixed")[2], permutations)


def test_pair_selection_uses_only_fixed_controls_and_has_one_weight_per_TP():
    values = {a: np.zeros(7) for a in SCORES}
    values["confidence"][:] = [.5, .51, .49, .52, .48, .5, .8]
    values["semantic_mean"][:] = .3
    values["geometry_mean"][:] = .7
    classes = np.array([1, 1, 1, 1, 1, 2, 1])
    labels = np.array([True, False, False, False, False, False, False])
    eligible = np.ones(7, bool)
    positive, negative, weight = conditional_pairs(classes, labels, eligible, values)
    assert len(positive) == 4
    assert set(negative) == {1, 2, 3, 4}
    assert weight.sum() == 1
    values["coupling"] = np.arange(7)*100.
    values["shuffled"] = -values["coupling"]
    changed = conditional_pairs(classes, labels, eligible, values)
    for old, new in zip((positive, negative, weight), changed):
        np.testing.assert_array_equal(old, new)


def test_pair_concordance_gives_ties_half_and_no_pairs_zero_weight():
    values = {a: np.array([1., 1., 0.]) for a in SCORES}
    result = pair_totals(np.array([0, 0]), np.array([1, 2]), np.array([.5, .5]), values)
    assert all(x == .75 for x in result.values())
    result = pair_totals(np.array([], int), np.array([], int), np.array([]), values)
    assert all(x == 0 for x in result.values())


def test_constant_native_semantic_field_has_no_coupling():
    slots = np.zeros((1, 8, 8))
    slots[0, :7, 7] = 1
    slots[0, :7, 4] = .125
    slots[0, :7, 6] = np.arange(7)/8
    values, _, _ = features(slots, np.array([.7]), "constant")
    assert values["coupling"][0] == values["shuffled"][0] == 0
