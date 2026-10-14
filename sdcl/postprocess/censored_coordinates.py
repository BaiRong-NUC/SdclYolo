"""Fixed coordinate exchanges using censored evidence, preserving prediction identity."""

import numpy as np

from sdcl.evaluation.size_analysis import box_iou_matrix
from sdcl.postprocess.censored_retrieval import peer_scores, row_membership, witnesses


def assign_anchors(candidates, baseline):
    iou = box_iou_matrix(candidates[:, :4], baseline[:, :4])
    valid = ((candidates[:, None, 5] == baseline[None, :, 5]) &
             (baseline[None, :, 4] >= .25) & (iou >= .7))
    if not len(baseline):
        return np.full(len(candidates), -1, int)
    return np.where(valid.any(1), np.where(valid, iou, -1).argmax(1), -1)


def exchange(baseline, dense, dense_view, uncapped, uncapped_view):
    suppressed = ~row_membership(dense, dense_view, uncapped, uncapped_view)
    source = np.flatnonzero(suppressed)
    witness = witnesses(dense[source], dense_view[source], uncapped, uncapped_view)
    source = source[witness >= 0]
    owner = assign_anchors(dense[source], baseline)
    source, owner = source[owner >= 0], owner[owner >= 0]
    candidates, origins = dense[source], dense_view[source]
    agreement, contrast = peer_scores(candidates, origins, baseline[owner], dense, dense_view)
    chosen = {}
    for row in np.argsort(-contrast, kind="stable"):
        if len(chosen) == 4 or contrast[row] <= 0:
            break
        chosen.setdefault(int(owner[row]), int(row))
    arms = {a: baseline.copy() for a in (
        "contrast_coordinates", "native_coordinates", "ordinary_agreement_coordinates",
        "kept_agreement_coordinates")}
    for anchor, row in chosen.items():
        alternatives = np.flatnonzero(owner == anchor)
        selections = {
            "contrast_coordinates": row,
            "native_coordinates": alternatives[np.argmax(candidates[alternatives, 4])],
            "ordinary_agreement_coordinates": alternatives[np.argmax(agreement[alternatives])],
        }
        for arm, choice in selections.items():
            arms[arm][anchor, :4] = candidates[choice, :4]
        valid = ((uncapped[:, 5] == baseline[anchor, 5]) &
                 (box_iou_matrix(baseline[anchor:anchor+1, :4], uncapped[:, :4])[0] >= .7))
        if valid.any():
            pool, view = uncapped[valid], uncapped_view[valid]
            support, _ = peer_scores(pool, view, np.repeat(baseline[anchor:anchor+1], len(pool), 0),
                                     dense, dense_view)
            arms["kept_agreement_coordinates"][anchor, :4] = pool[np.argmax(support), :4]
    return arms, {
        "suppressed_anchored_candidates": len(candidates),
        "positive_contrast_candidates": int((contrast > 0).sum()),
        "selected_anchor_rows": list(chosen),
        "selected_native_rows": [int(source[i]) for i in chosen.values()],
    }
