import numpy as np
import torch
from types import SimpleNamespace

from sdcl.evaluation.detection_soft_evidence import native_view, posterior_evidence


def test_native_soft_menu_exposes_only_top_three_supported_classes():
    baseline = np.array([[0, 0, 10, 10, .8, 0]], np.float32)
    probabilities = np.array([[.8, .4, .2, .1, .01, 0, 0, 0, 0, 0]], np.float32)
    hard_valid = np.zeros((1, 10), bool)
    hard_valid[0, 0] = True
    views = [baseline] + [np.empty((0, 6))] * 4
    scores = [probabilities] + [np.empty((0, 10))] * 4
    valid, evidence = posterior_evidence(baseline, views, scores, hard_valid)
    assert np.flatnonzero(valid[0]).tolist() == [0, 1, 2]
    np.testing.assert_allclose(evidence[0, :, 0], probabilities[0])


def test_no_supported_view_preserves_hard_menu_and_no_duplicate_votes():
    baseline = np.array([[0, 0, 10, 10, .8, 0]], np.float32)
    raw = np.array([[0, 0, 10, 10, .8, 0], [0, 0, 10, 10, .7, 1]], np.float32)
    probabilities = np.zeros((2, 10), np.float32)
    probabilities[:, 0] = [.8, .7]
    probabilities[1, 1] = .6
    hard = np.eye(10, dtype=bool)[[0]]
    valid, evidence = posterior_evidence(
        baseline, [raw] + [np.empty((0, 6))] * 4,
        [probabilities] + [np.empty((0, 10))] * 4, hard)
    assert not valid[0, 1]
    assert evidence[0, 0, 0] == .8
    empty_valid, empty = posterior_evidence(
        baseline, [np.empty((0, 6))] * 5, [np.empty((0, 10))] * 5, hard)
    np.testing.assert_array_equal(empty_valid, hard)
    assert not empty.any()


def test_empty_native_nms_indices_preserve_empty_class_score_rows():
    extractor = SimpleNamespace(
        predict=lambda image: ({"control": np.empty((0, 6), np.float32)}, {}),
        decoded=torch.zeros((1, 14, 1)),
        head=SimpleNamespace(stride=torch.tensor([8.])),
        raw={"feats": [torch.zeros((1, 1, 120, 120))]},
    )
    boxes, scores = native_view(extractor, np.zeros((960, 960, 3), np.uint8))
    assert boxes.shape == (0, 6)
    assert scores.shape == (0, 10)
