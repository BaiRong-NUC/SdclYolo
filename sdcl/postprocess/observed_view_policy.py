"""Features from predictions in already-computed diagonal views only."""

import numpy as np

from sdcl.evaluation.size_analysis import box_areas, box_iou_matrix
from sdcl.postprocess.view_value import global_window_features


OBSERVED = (1, 4)
OMITTED = (2, 3)
GLOBAL_DIM = 61


def _inside(values, window):
    centers = (values[:, :2] + values[:, 2:4]) / 2
    return ((centers >= window[:2]) & (centers < window[2:])).all(1)


def observed_features(global_predictions, computed, windows, size):
    if set(computed) != set(OBSERVED) or len(windows) != 5:
        raise ValueError("Only the two fixed observed local views may enter the policy.")
    global_predictions = np.asarray(global_predictions, float).reshape(-1, 6)
    base = global_window_features(global_predictions, windows, size)[np.array(OMITTED)-1]
    rows = []
    for omitted in OMITTED:
        extra = []
        for view in OBSERVED:
            local = np.asarray(computed[view], float).reshape(-1, 6)
            observed_window = np.asarray(windows[view], float)
            candidate_window = np.asarray(windows[omitted], float)
            intersection = np.r_[np.maximum(observed_window[:2], candidate_window[:2]),
                                  np.minimum(observed_window[2:], candidate_window[2:])]
            overlap = box_iou_matrix(local[:, :4], global_predictions[:, :4])
            same_class = local[:, 5, None] == global_predictions[None, :, 5]
            for region in (observed_window, intersection):
                for score in (.05, .25):
                    lg = local[_inside(local, region) & (local[:, 4] >= score)]
                    gg = global_predictions[_inside(global_predictions, region) &
                                            (global_predictions[:, 4] >= score)]
                    # Counterpart search is unrestricted spatially so crop-edge
                    # centers cannot create artificial unmatched detections.
                    paired = ((overlap >= .5) & same_class &
                              (global_predictions[None, :, 4] >= score))
                    local_novel = ~paired.any(1)
                    global_novel = ~((overlap >= .5) & same_class &
                                     (local[:, 4, None] >= score)).any(0)
                    lmask = _inside(local, region) & (local[:, 4] >= score)
                    gmask = _inside(global_predictions, region) & (global_predictions[:, 4] >= score)
                    la = box_areas(lg[:, :4]) * (640 / max(size)) ** 2
                    ga = box_areas(gg[:, :4]) * (640 / max(size)) ** 2
                    for bound in (np.inf, 16**2, 32**2):
                        extra.extend(np.log1p([
                            (la < bound).sum(),
                            ((la < bound) & local_novel[lmask]).sum(),
                            ((ga < bound) & global_novel[gmask]).sum(),
                        ]))
        rows.append(extra)
    result = np.concatenate((base, np.asarray(rows, np.float32)), axis=1).astype(np.float32)
    if result.shape != (2, 133) or not np.isfinite(result).all():
        raise ValueError("Invalid observed-view input.")
    return result


def derange_images(features, seed):
    values = np.asarray(features)
    if len(values) < 2:
        raise ValueError("At least two images needed for an input-shuffle control.")
    order = np.random.default_rng(seed).permutation(len(values))
    partner = np.empty(len(order), int)
    partner[order] = np.roll(order, 1)
    result = values.copy()
    result[:, :, GLOBAL_DIM:] = values[partner, :, GLOBAL_DIM:]
    return result, partner
