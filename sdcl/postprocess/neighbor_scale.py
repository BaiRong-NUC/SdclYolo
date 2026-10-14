"""Prediction-only scale consistency against other nonoverlapping instances."""

import numpy as np

from sdcl.evaluation.size_analysis import box_iou_matrix


def scale_context(candidates, actions, reference_predictions, reference_choice, size):
    n = len(candidates)
    count = actions.shape[1]
    centers = (candidates[:, :2]+candidates[:, 2:4])/2
    peer_centers = (reference_predictions[:, :2]+reference_predictions[:, 2:4])/2
    peer_wh = reference_predictions[:, 2:4]-reference_predictions[:, :2]
    action_wh = actions[:, :, 2:]-actions[:, :, :2]
    if np.any(peer_wh <= 0) or np.any(action_wh <= 0):
        raise ValueError("Positive candidate and peer extents required.")
    overlap = box_iou_matrix(candidates[:, :4], reference_predictions[:, :4])
    distance = np.linalg.norm(centers[:, None]-peer_centers[None], axis=-1)
    eligible = ((candidates[:, 5, None] == reference_predictions[None, :, 5]) &
                (reference_predictions[None, :, 4] >= .25) & (overlap < .1))
    local_score, global_score = np.zeros((n, count)), np.zeros((n, count))
    supported = np.zeros(n, bool)
    neighbors = np.full((n, 8), -1, int)
    global_count = np.zeros(n, int)
    for row in range(n):
        global_peers = np.flatnonzero(eligible[row])
        local_peers = global_peers[distance[row, global_peers] <= .20*np.linalg.norm(size)]
        order = np.argsort(distance[row, local_peers], kind="stable")[:8]
        local_peers = local_peers[order]
        global_count[row] = len(global_peers)
        neighbors[row, :len(local_peers)] = local_peers
        if len(local_peers) < 4 or len(global_peers) < 4:
            continue
        supported[row] = True
        log_actions = np.log(action_wh[row])
        for dest, peer in ((local_score, local_peers), (global_score, global_peers)):
            center = np.median(np.log(peer_wh[peer]), axis=0)
            dest[row] = -np.abs(log_actions-center).mean(-1)
    row = np.arange(n)
    reference_boxes = actions[row, reference_choice]
    scale = np.tile(candidates[:, 2:4]-candidates[:, :2], (1, 2))
    reference_distance = -np.abs((actions-reference_boxes[:, None])/scale[:, None]).mean(-1)
    return {"local_scale": local_score, "global_scale": global_score,
            "reference_distance": reference_distance, "supported": supported,
            "neighbors": neighbors, "global_peer_count": global_count}
