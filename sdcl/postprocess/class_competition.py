"""Frozen cross-class score suppression using common or independent native sites."""

import numpy as np

from sdcl.evaluation.size_analysis import box_iou_matrix


ARMS = ("shared", "marginal", "affinity", "geometry")


def candidate_pairs(predictions):
    overlap = box_iou_matrix(predictions[:, :4], predictions[:, :4])
    query, peer = np.nonzero(np.tril(np.ones(overlap.shape, bool), -1) &
        (predictions[:, 5, None] != predictions[None, :, 5]) & (overlap >= .7))
    return np.column_stack((query, peer)), overlap[query, peer]


def competition_evidence(predictions, views):
    values = np.asarray(predictions, float).reshape(-1, 6)
    if len(values) > 500 or np.any(np.diff(values[:, 4]) > 0):
        raise ValueError("Expected frozen descending-score strong output.")
    pairs, overlap = candidate_pairs(values)
    contrasts = {arm: np.zeros(len(pairs)) for arm in ARMS}
    contrasts["geometry"][:] = 1.
    witnesses = np.full((len(pairs), 5), -1, int)
    shared_weights, shared_directions = np.zeros((len(pairs), 5)), np.zeros((len(pairs), 5))
    marginal_i, marginal_j = np.zeros((len(pairs), 5)), np.zeros((len(pairs), 5))
    semantic_sum, semantic_weight = np.zeros((len(values), 10)), np.zeros(len(values))
    collided = np.unique(pairs) if len(pairs) else np.array([], int)
    if len(views) != 5:
        raise ValueError("Exactly five native views required.")
    for view, (raw_boxes, raw_probability) in enumerate(views):
        boxes, probability = np.asarray(raw_boxes, float), np.asarray(raw_probability, float)
        if boxes.shape != (len(probability), 4) or probability.ndim != 2 or probability.shape[1] != 10:
            raise ValueError("Expected aligned native xyxy and ten class probabilities.")
        valid = (np.isfinite(boxes).all(1) & np.isfinite(probability).all(1) &
                 (boxes[:, 2:] > boxes[:, :2]).all(1) & (probability.max(1) >= .05))
        native = np.flatnonzero(valid)
        if not len(native) or not len(collided):
            continue
        p = probability[native]
        ious = box_iou_matrix(values[collided, :4], boxes[native])
        row_lookup = {int(row): k for k, row in enumerate(collided)}
        for rank, row in enumerate(collided):
            mask = ious[rank] >= .5
            if not mask.any():
                continue
            weights = np.where(mask, ious[rank]*p[:, int(values[row, 5])], -1.)
            chosen = int(np.argmax(weights))
            weight = weights[chosen]
            semantic_sum[row] += weight*p[chosen]/max(p[chosen].sum(), 1e-12)
            semantic_weight[row] += weight
        for rank, (i, j) in enumerate(pairs):
            g = np.minimum(ious[row_lookup[int(i)]], ious[row_lookup[int(j)]])
            pool = np.flatnonzero(g >= .5)
            if not len(pool):
                continue
            pi, pj = p[pool, int(values[i, 5])], p[pool, int(values[j, 5])]
            weight = g[pool]*np.maximum(pi, pj)
            chosen = int(np.argmax(weight))
            witnesses[rank, view] = native[pool[chosen]]
            shared_weights[rank, view] = weight[chosen]
            shared_directions[rank, view] = (pj[chosen]-pi[chosen])/(pj[chosen]+pi[chosen]+1e-12)
            marginal_i[rank, view] = (g[pool]*pi).max()
            marginal_j[rank, view] = (g[pool]*pj).max()
    contrasts["shared"] = np.maximum(0., (shared_weights*shared_directions).sum(1)/
                                            np.maximum(shared_weights.sum(1), 1e-12))
    mi, mj = marginal_i.sum(1), marginal_j.sum(1)
    contrasts["marginal"] = np.maximum(0., (mj-mi)/(mj+mi+1e-12))
    semantic = semantic_sum/np.maximum(semantic_weight[:, None], 1e-12)
    if len(pairs):
        contrasts["affinity"] = np.sqrt(semantic[pairs[:, 0]]*semantic[pairs[:, 1]]).sum(1).clip(0, 1)
    penalties = {}
    for arm, contrast in contrasts.items():
        penalty = np.zeros(len(values))
        if len(pairs):
            np.maximum.at(penalty, pairs[:, 0], overlap**2*contrast/.5)
        penalties[arm] = penalty
    return dict(pairs=pairs, pair_iou=overlap, witnesses=witnesses, shared_weights=shared_weights,
                shared_directions=shared_directions, marginal_i=marginal_i, marginal_j=marginal_j,
                semantic=semantic, **{"contrast_"+a: v for a, v in contrasts.items()},
                **{"penalty_"+a: v for a, v in penalties.items()})


def suppress_scores(predictions, penalty):
    values = np.asarray(predictions, float).reshape(-1, 6).copy()
    penalty = np.asarray(penalty, float)
    if penalty.shape != (len(values),) or not np.isfinite(penalty).all() or np.any(penalty < 0):
        raise ValueError("Expected finite nonnegative per-row penalties.")
    changed = penalty > 0
    values[changed, 4] *= np.exp(-penalty[changed])
    keep = np.flatnonzero(values[:, 4] >= .001)
    order = keep[np.argsort(-values[keep, 4], kind="stable")][:500]
    return values[order], order
