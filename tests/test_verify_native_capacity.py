import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("capacity_audit", Path(__file__).resolve().parents[1]/
                                           "scripts/evaluation/verify_native_capacity.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_certificate_verifier_rejects_two_matches_using_one_position():
    with pytest.raises(AssertionError):
        audit.verify_certificate(2, np.array([[0, 1], [8, 8]]), np.array([8, 8]),
                                 np.array([], int), np.array([8]))


def test_certificate_verifier_rejects_uncovered_edges():
    with pytest.raises(AssertionError):
        audit.verify_certificate(2, np.array([[0, 1], [8, 9]]), np.array([8, -1]),
                                 np.array([0]), np.array([], int))


def test_certificate_proves_exact_optimum_with_isolated_GT():
    support, statistics = audit.verify_certificate(3, np.array([[0, 1], [8, 8]]), np.array([8, -1, -1]),
                                                   np.array([], int), np.array([8]))
    np.testing.assert_array_equal(support, [True, True, False])
    assert statistics["matching_capacity"] == 1 and statistics["capacity_deficit"] == 1
