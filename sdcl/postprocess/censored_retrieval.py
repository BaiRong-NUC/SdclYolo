"""Prediction-only fixed suppression-witness retrieval for a diagnostic screen."""

import numpy as np

from sdcl.evaluation.size_analysis import box_iou_matrix


def row_membership(values, origins, reference, reference_origins):
    keys = {tuple(row) + (int(view),) for row, view in zip(reference, reference_origins)}
    return np.asarray([tuple(row) + (int(view),) in keys
                       for row, view in zip(values, origins)], dtype=bool)


def residual_mask(candidates, baseline):
    relevant = baseline[baseline[:, 4] >= .25]
    iou = box_iou_matrix(candidates[:, :4], relevant[:, :4])
    same = candidates[:, None, 5] == relevant[None, :, 5]
    return ~((iou >= .7) & same).any(1)


def witnesses(dense, dense_view, kept, kept_view):
    """Find a higher-score retained witness; labels are never an input."""
    iou = box_iou_matrix(dense[:, :4], kept[:, :4])
    valid = ((dense[:, None, 5] == kept[None, :, 5]) &
             (dense_view[:, None] == kept_view[None, :]) &
             (dense[:, None, 4] < kept[None, :, 4]) & (iou > .7))
    if not len(kept):
        return np.full(len(dense), -1, dtype=int)
    ranked = np.where(valid, kept[None, :, 4], -1)
    return np.where(valid.any(1), ranked.argmax(1), -1)


def peer_scores(candidates, origins, suppressors, dense, dense_view):
    agreement, contrast = np.zeros(len(candidates)), np.zeros(len(candidates))
    for cls in range(10):
        query = np.flatnonzero(candidates[:, 5] == cls)
        peer = np.flatnonzero(dense[:, 5] == cls)
        if not len(query) or not len(peer):
            continue
        iou = box_iou_matrix(candidates[query, :4], dense[peer, :4])
        other_iou = box_iou_matrix(suppressors[query, :4], dense[peer, :4])
        available = (origins[query, None] != dense_view[None, peer]) & (iou >= .5)
        agreement[query] = np.max(np.where(
            available, iou * dense[None, peer, 4], 0), axis=1)
        difference = iou - other_iou
        contrast[query] = np.max(np.where(
            available & (difference >= .10), difference * dense[None, peer, 4], 0), axis=1)
    return candidates[:, 4] * agreement, candidates[:, 4] * contrast


def select(candidates, ranking, count, require_positive=False):
    selected = []
    for index in np.argsort(-ranking, kind="stable"):
        if len(selected) >= count:
            break
        if require_positive and ranking[index] <= 0:
            break
        if selected:
            same = candidates[selected, 5] == candidates[index, 5]
            overlap = box_iou_matrix(candidates[index:index+1, :4],
                                     candidates[selected, :4])[0]
            if np.any(same & (overlap >= .7)):
                continue
        selected.append(index)
    return np.asarray(selected, dtype=int)


def append_selected(baseline, candidates, indices):
    extra = candidates[indices].copy()
    extra[:, 4] = .25
    merged = np.concatenate((baseline, extra))
    is_extra = np.r_[np.zeros(len(baseline), bool), np.ones(len(extra), bool)]
    order = np.argsort(-merged[:, 4], kind="stable")[:500]
    return merged[order], is_extra[order]


def retrieve(baseline, dense, dense_view, uncapped, uncapped_view):
    eligible = (~row_membership(dense, dense_view, uncapped, uncapped_view) &
                residual_mask(dense, baseline))
    source = np.flatnonzero(eligible)
    witness = witnesses(dense[source], dense_view[source], uncapped, uncapped_view)
    source, witness = source[witness >= 0], witness[witness >= 0]
    candidates, origins = dense[source], dense_view[source]
    agreement, contrast = peer_scores(candidates, origins, uncapped[witness], dense, dense_view)
    selected = select(candidates, contrast, 4, require_positive=True)
    count = len(selected)
    uncapped_residual = uncapped[residual_mask(uncapped, baseline)]
    arms = {
        "contrast": append_selected(baseline, candidates, selected),
        "native_rank": append_selected(
            baseline, candidates, select(candidates, candidates[:, 4], count)),
        "ordinary_agreement": append_selected(
            baseline, candidates, select(candidates, agreement, count)),
        "uncapped_native_rank": append_selected(
            baseline, uncapped_residual,
            select(uncapped_residual, uncapped_residual[:, 4], count)),
    }
    return arms, {
        "suppressed_residual_candidates": len(candidates),
        "contrast_positive_candidates": int((contrast > 0).sum()),
        "requested_additions": count,
        "additions_after_cap": {arm: int(extra.sum()) for arm, (_, extra) in arms.items()},
    }
