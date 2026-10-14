"""Screen native DFL likelihood for fixed coordinate actions before any fitting."""

import json
from time import perf_counter

import cv2
import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.class_choice_signal import weighted_auc
from sdcl.evaluation.size_analysis import box_iou_matrix, file_hash, group_mask
from sdcl.evaluation.slicing_screen import map_tile_predictions
from sdcl.postprocess.dfl_action_evidence import action_distances, evidence
from sdcl.postprocess.rescue import save_json
from sdcl.training.class_margin_probe import FrozenViews
from sdcl.training.marginal_class_intervention import verify_record
from sdcl.training.rescue import archive_source


REGISTRATION = "docs/experiments/candidate24_dfl_action_signal_registration_20261014.json"
OUTPUT = "output/analysis/dfl-action-signal-fixed-20261014"
BASE = "output/analysis/class-choice-pilot-fixed-20261014"
MENU = "output/analysis/action-order-sensitivity-fixed-20261014"
SCORES = ("distribution", "mean_distance", "gaussian")
GROUPS = ("all", "tiny", "small_all")


def native_evidence(extractor, image, baseline, boxes, window):
    from ultralytics.utils.ops import scale_boxes, xywh2xyxy

    prediction, _ = extractor.predict(image)
    head, raw, decoded = extractor.head, extractor.raw, extractor.decoded
    expected = head.decode_bboxes(head.dfl(raw["boxes"]), head.anchors.unsqueeze(0)) * head.strides
    if not torch.equal(expected, decoded[:, :4]):
        raise ValueError("Native DFL box decode differs.")
    probability = raw["boxes"].view(1, 4, 16, -1).transpose(2, 1).softmax(1)
    probability = probability[0].permute(2, 1, 0)
    native = decoded[0].T
    score, cls = native[:, 4:14].max(1)
    active = torch.nonzero(score >= .05).reshape(-1)
    input_shape = tuple(int(x * head.stride[0]) for x in raw["feats"][0].shape[2:])
    xyxy = xywh2xyxy(native[active, :4])
    xyxy = scale_boxes(input_shape, xyxy, image.shape[:2])
    values = torch.cat((xyxy, score[active, None], cls[active, None].float()), 1).float().cpu().numpy()
    values = map_tile_predictions(values, window)
    scores = {s: np.zeros(boxes.shape[:2], np.float32) for s in SCORES}
    available = np.zeros(len(baseline), bool)
    indices = np.full(len(baseline), -1, int)
    selected_probability = np.zeros((len(baseline), 4, 16), np.float32)
    selected_anchors = np.zeros((len(baseline), 2), np.float32)
    selected_strides = np.zeros(len(baseline), np.float32)
    if len(values):
        iou = box_iou_matrix(baseline[:, :4], values[:, :4])
        valid = (baseline[:, None, 5] == values[None, :, 5]) & (iou >= .5)
        quality = np.where(valid, iou * values[None, :, 4], 0)
        best = quality.argmax(1)
        available = valid.any(1)
        native_indices = active[torch.from_numpy(best).to(active.device)]
        rows = np.flatnonzero(available)
        chosen = native_indices[torch.from_numpy(rows).to(active.device)]
        p = probability[chosen].float().cpu().numpy()
        anchors = head.anchors[:, chosen].T.float().cpu().numpy()
        strides = head.strides[:, chosen].reshape(-1).float().cpu().numpy()
        distance = action_distances(boxes[rows], anchors, strides, window, input_shape)
        calculated = evidence(p, distance)
        for name, value in calculated.items():
            scores[name][rows] = value
        indices[rows] = chosen.cpu().numpy()
        selected_probability[rows], selected_anchors[rows], selected_strides[rows] = p, anchors, strides
    return prediction["control"], scores, {
        "available": available, "native_anchor": indices, "probability": selected_probability,
        "anchors": selected_anchors, "strides": selected_strides,
    }


def summarize(frames):
    accumulated = {g: {"labels": [], "weights": [], **{s: [] for s in SCORES}} for g in GROUPS}
    for frame in frames:
        reference = frame["uniform"]
        nominal = frame["ious"][:, 0]
        delta = nominal - nominal[np.arange(len(reference)), reference, None]
        eligible = (frame["owner"] >= 0) & (frame["candidates"][:, 4] >= .25) & (frame["views"] >= 2)
        valid = frame["valid"] & eligible[:, None] & (np.abs(delta) > .01)
        valid[np.arange(len(reference)), reference] = False
        weight = np.broadcast_to(1 / np.maximum(valid.sum(1), 1)[:, None], valid.shape)
        for group in GROUPS:
            mask = valid & group_mask(frame["areas"], group)[:, None]
            accumulated[group]["labels"].append((delta > .01)[mask])
            accumulated[group]["weights"].append(weight[mask])
            for name in SCORES:
                score = frame[name]
                relative = score - score[np.arange(len(reference)), reference, None]
                accumulated[group][name].append(relative[mask])
    result = {}
    for group, values in accumulated.items():
        joined = {k: np.concatenate(v) for k, v in values.items()}
        labels, weight = joined["labels"], joined["weights"]
        result[group] = {
            "positive_actions": int(labels.sum()), "negative_actions": int((~labels).sum()),
            **{s + "_weighted_auc": weighted_auc(labels, joined[s], weight) for s in SCORES},
        }
    return result


def decide(metrics, policy):
    tiny = metrics["tiny"]
    auc = tiny["distribution_weighted_auc"]
    gains = {s: None if auc is None or tiny[s + "_weighted_auc"] is None else
             auc - tiny[s + "_weighted_auc"] for s in ("mean_distance", "gaussian")}
    checks = {
        "positive_actions": tiny["positive_actions"] >= policy["tiny_positive_actions_min"],
        "negative_actions": tiny["negative_actions"] >= policy["tiny_negative_actions_min"],
        "distribution_discrimination": auc is not None and auc >= policy["tiny_distribution_auc_min"],
        "beyond_mean": gains["mean_distance"] is not None and
        gains["mean_distance"] >= policy["tiny_distribution_auc_vs_mean_min"],
        "beyond_gaussian": gains["gaussian"] is not None and
        gains["gaussian"] >= policy["tiny_distribution_auc_vs_gaussian_min"],
    }
    return {"decision": "MATCHED_DFL_LEARNABILITY_PILOT_ALLOWED" if all(checks.values()) else
            "STOP_THIS_INTERPOLATED_DFL_EVIDENCE_RECIPE", "checks": checks,
            "tiny_distribution_auc": auc, "tiny_auc_gains": gains,
            "scope": "Conditional same-owned action discrimination only; no AP or deployment result."}


def evaluate(output=OUTPUT, device="0"):
    started = perf_counter()
    if not str(device).isdigit() or not torch.cuda.is_available():
        raise ValueError("An available CUDA index is required.")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    base, menu = project_path(BASE), project_path(MENU)
    _, plan = verify_record(base, "docs/experiments/candidate12_class_choice_result_20261014.json")
    _, action_plan = verify_record(menu, "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json")
    stems = plan["image_selection"]["holdout"][:64]
    if stems != action_plan["image_selection"]["holdout"]:
        raise ValueError("Fixed head-held-out order differs.")
    records = {r["stem"]: r for r in plan["records"]["holdout"]}
    raw_path = base / "raw_views.jsonl"
    if file_hash(raw_path) != action_plan["raw_views_sha256"]:
        raise ValueError("Raw frozen views changed.")
    raw = {r["image"]: r for r in (json.loads(line) for line in raw_path.read_text().splitlines())}
    manifest = json.loads((menu / "frames.json").read_text())
    registration = json.loads(project_path(REGISTRATION).read_text())
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    paths = (REGISTRATION, "sdcl/postprocess/dfl_action_evidence.py",
             "sdcl/evaluation/dfl_action_signal.py", "scripts/evaluation/diagnose_dfl_action_signal.py",
             "sdcl/training/class_margin_probe.py", "sdcl/evaluation/class_choice_signal.py",
             "sdcl/evaluation/size_analysis.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    archive_source(output, hashes)
    import ultralytics.nn.modules.block as installed_block
    import ultralytics.nn.modules.head as installed_head
    import ultralytics.utils.ops as installed_ops
    installed = {m.__file__: file_hash(m.__file__) for m in
                 (installed_block, installed_head, installed_ops)}
    save_json(output / "plan.json", {
        "registration": registration, "source_hashes": hashes, "images": stems,
        "source_archive_sha256": file_hash(output / "source.zip"), "installed_sources": installed,
        "raw_views_sha256": file_hash(raw_path),
        "prior_result_sha256": file_hash(project_path(
            "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json")),
    })
    extractor = FrozenViews(device)
    summaries, frames, forwards = [], {}, 0
    try:
        with torch.inference_mode():
            for index, stem in enumerate(stems, 1):
                snapshot = manifest[stem]
                frame_path = menu / snapshot["file"]
                image_path = project_path(f"data/visdrone/images/train/{stem}.jpg")
                if (file_hash(frame_path) != snapshot["sha256"] or
                        file_hash(image_path) != records[stem]["image_sha256"]):
                    raise ValueError("Fixed action frame or input image changed.")
                with np.load(frame_path, allow_pickle=False) as c:
                    # Labels are used only in summarize, after evidence is complete.
                    cached = {k: c[k].copy() for k in c.files}
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError("Frozen image unreadable.")
                original = raw[stem]
                per_view = {s: [] for s in SCORES}
                details = []
                for view, window in enumerate(original["windows"]):
                    x1, y1, x2, y2 = window
                    prediction, scores, trace = native_evidence(
                        extractor, image[y1:y2, x1:x2].copy(), cached["baseline"], cached["boxes"], window)
                    mapped = map_tile_predictions(prediction, window) if view else prediction
                    if not np.array_equal(mapped, np.asarray(original["views"][view], np.float32).reshape(-1, 6)):
                        raise ValueError("Frozen raw detector replay differs.")
                    for name in SCORES:
                        per_view[name].append(scores[name])
                    details.append(trace)
                    forwards += 1
                available = np.stack([v["available"] for v in details])
                view_count = available.sum(0)
                evidence_scores = {s: np.stack(v).sum(0) / np.maximum(view_count[:, None], 1)
                                   for s, v in per_view.items()}
                summaries.append({k: cached[k] for k in
                                  ("uniform", "ious", "owner", "candidates", "valid", "areas")})
                summaries[-1].update(**evidence_scores, views=view_count)
                target = output / "frames" / f"{stem}.npz"
                np.savez_compressed(target, **evidence_scores, views=view_count,
                                    **{k: np.stack([d[k] for d in details]) for k in details[0]})
                frames[stem] = {"file": target.relative_to(output).as_posix(),
                                "sha256": file_hash(target), "menu_frame_sha256": snapshot["sha256"]}
                if index == 1 or index % 16 == 0:
                    print(f"Native DFL action signal: {index}/64, {perf_counter()-started:.1f}s", flush=True)
    finally:
        extractor.close()
    metrics = summarize(summaries)
    verdict = decide(metrics, registration["policy"])
    save_json(output / "frames.json", frames)
    save_json(output / "metrics.json", metrics)
    save_json(output / "decision.json", verdict)
    save_json(output / "verification.json", {
        "images": len(stems), "detector_forwards_excluding_warmup": forwards, "training_steps": 0,
        "raw_kept_replay_exact": True, "native_dfl_decode_exact": True,
        "prediction_only_association_fixed_across_actions": True,
        "no_val_no_testdev_no_threshold_search": True,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in hashes.items()),
        "installed_sources_unchanged": all(file_hash(p) == h for p, h in installed.items()),
        "elapsed_seconds": perf_counter() - started,
    })
    return {"output": str(output), "decision": verdict, "metrics": metrics}
