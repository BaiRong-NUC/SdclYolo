"""Fixed scene response information screen on ordinary calibrated detections."""

import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter
import zipfile

import cv2
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import torch
from ultralytics.data import augment as augment_module
from ultralytics.nn.modules import head as head_module
from ultralytics.utils import nms as nms_module
from ultralytics.utils import ops as ops_module

from sdcl.config import project_path
from sdcl.data.geometry import prediction_ignore_mask
from sdcl.evaluation.classifier_nullspace import fit
from sdcl.evaluation.size_analysis import box_areas, box_iou_matrix, file_hash, match_at_threshold
from sdcl.evaluation.slicing_screen import map_tile_predictions
from sdcl.postprocess.localization import calibrated_candidates, frozen_control, load_calibrator
from sdcl.postprocess.rescue import save_json
from sdcl.training.class_margin_probe import FrozenViews
from sdcl.training.marginal_class_intervention import verify_record


REGISTRATION = "docs/experiments/candidate55_scene_background_registration_20261015.json"
OUTPUT = "output/analysis/scene-background-fixed-20261015"
PRIOR = "output/analysis/class-choice-pilot-fixed-20261014"
SELECTION = "output/analysis/action-order-sensitivity-fixed-20261014"
ARMS = ("row", "whole", "masked", "shuffled")
GROUPS = dict(all=float("inf"), tiny=256., small_all=1024.)


def view_statistics(logits, predictions, width, height):
    ratio = min(960/width, 960/height)
    rw, rh = round(width*ratio), round(height*ratio)
    pad = [round((960-rw)/2-.1), round((960-rh)/2-.1)]
    whole, masked, fractions, masks, fallback = [], [], [], [], []
    start = 0
    for stride in (8, 16, 32):
        side = 960//stride
        yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
        point = np.column_stack(((xx.ravel()+.5)*stride, (yy.ravel()+.5)*stride))
        valid = ((point >= pad) & (point < np.array(pad)+[rw, rh])).all(1)
        excluded = np.zeros(len(point), bool)
        for box in np.asarray(predictions).reshape(-1, 6):
            if box[4] < .25:
                continue
            low = box[:2]*ratio+pad
            high = box[2:4]*ratio+pad
            excluded |= ((point >= low) & (point <= high)).all(1)
        pool = valid & ~excluded
        fractions.append(float(pool.sum()/valid.sum()))
        fallback.append(bool(pool.sum() < 32))
        selected = valid if fallback[-1] else pool
        values = np.asarray(logits[start:start+side*side], float)
        for output, mask in ((whole, valid), (masked, selected)):
            sample = values[mask]
            quantiles = np.quantile(sample, [.5, .9, .99], axis=0).T
            output.append(np.column_stack((quantiles, (sample >= np.log(.05/.95)).mean(0))))
        masks.append(np.column_stack((valid, pool, selected)))
        start += side*side
    assert start == 18900
    return np.array(whole), np.array(masked), np.array(fractions), np.concatenate(masks), np.array(fallback)


def scene_permutation(stems):
    order = sorted(range(len(stems)), key=lambda i: hashlib.sha256(("candidate55-scene-shuffle:"+stems[i]).encode()).hexdigest())
    result = np.empty(len(stems), int)
    for i, row in enumerate(order):
        result[row] = order[(i+1) % len(order)]
    if len(stems) < 2 or np.any(result == np.arange(len(stems))):
        raise ValueError("Scene control needs a permutation without fixed points.")
    return result


def labels_for(reference, metadata, confidence):
    gt = np.array([o["box_xyxy"] for o in metadata["valid_objects"]], float).reshape(-1, 4)
    classes = np.array([o["class_id"] for o in metadata["valid_objects"]], int)
    ignored = prediction_ignore_mask(torch.tensor(reference[:, :4]),
                                    torch.tensor(metadata["ignore_boxes_xyxy"], dtype=torch.float64).reshape(-1, 4)).numpy()
    temp = reference.copy()
    temp[ignored, 4] = -1
    iou = box_iou_matrix(gt, reference[:, :4])
    _, tp = match_at_threshold(iou, classes, temp, confidence)
    duplicate = ~tp & ((iou >= .5) & (classes[:, None] == reference[None, :, 5])).any(0)
    return tp, duplicate, ignored


def normalize(frames):
    means, scales = {}, {}
    for name in ("base", "whole", "masked", "fraction"):
        values = np.concatenate([f[name][f["fit_eligible"]] for f in frames["train"]])
        means[name], scales[name] = values.mean(0), np.maximum(values.std(0), 1e-6)
    # Both arms use common training statistics for comparable scene coordinates.
    mixed = np.concatenate([f[k][f["fit_eligible"]] for f in frames["train"] for k in ("whole", "masked")])
    means["whole"] = means["masked"] = mixed.mean(0)
    scales["whole"] = scales["masked"] = np.maximum(mixed.std(0), 1e-6)
    for split, fs in frames.items():
        permutation = scene_permutation([f["stem"] for f in fs])
        for i, f in enumerate(fs):
            classes = f["reference"][:, 5].astype(int)
            other = fs[permutation[i]]
            shuffled = other["scene_masked"][:, :, classes, :].transpose(2, 0, 1, 3).reshape(len(classes), 60)
            base = (f["base"]-means["base"])/scales["base"]
            whole = (f["whole"]-means["whole"])/scales["whole"]
            masked = (f["masked"]-means["masked"])/scales["masked"]
            shuffled = (shuffled-means["masked"])/scales["masked"]
            fraction = (f["fraction"]-means["fraction"])/scales["fraction"]
            zero = np.zeros_like(masked)
            f["shuffled_stem"] = other["stem"]
            f["inputs"] = dict(row=np.column_stack((base, zero, np.zeros_like(fraction), zero)),
                               whole=np.column_stack((base, whole, fraction, zero)),
                               masked=np.column_stack((base, whole, fraction, masked)),
                               shuffled=np.column_stack((base, whole, fraction, shuffled)))
    return dict(mean={k: v.tolist() for k, v in means.items()}, std={k: v.tolist() for k, v in scales.items()})


def summarize(frames, coefficients):
    traces = []
    labels, values = {g: [] for g in GROUPS}, {g: {a: [] for a in ARMS} for g in GROUPS}
    for f in frames:
        reference = f["reference"]
        tp, duplicate, ignored = labels_for(reference, f["metadata"], .25)
        areas = box_areas(reference[:, :4])*(640/max(f["metadata"]["original_size"]))**2
        offset = np.log(np.clip(reference[:, 4], 1e-6, 1-1e-6)/(1-np.clip(reference[:, 4], 1e-6, 1-1e-6)))
        logits = {a: offset+f["inputs"][a]@t[:-1]+t[-1] for a, t in coefficients.items()}
        trace = {}
        f.update(eval_tp=tp, duplicate=duplicate, ignored=ignored, areas=areas, output_logits=logits)
        for group, limit in GROUPS.items():
            population = (reference[:, 4] >= .25) & ~ignored & (areas < limit)
            selected = population & ~duplicate
            labels[group].extend(tp[selected].tolist())
            for a in ARMS:
                values[group][a].extend(logits[a][selected].tolist())
            trace[group] = dict(rows=int(selected.sum()), tp=int((selected & tp).sum()),
                                fp=int((selected & ~tp).sum()), duplicate_excluded=int((population & duplicate).sum()),
                                images=int(selected.any()),
                                loss_sums={a: float((np.logaddexp(0, z[selected])-tp[selected]*z[selected]).sum())
                                           for a, z in logits.items()})
        traces.append(trace)
    draws = np.random.default_rng(0).multinomial(64, [1/64]*64, size=400)
    metrics = {}
    for group in GROUPS:
        total = {k: sum(t[group][k] for t in traces) for k in ("rows", "tp", "fp", "duplicate_excluded", "images")}
        n = total["rows"]
        total["NLL"] = {a: sum(t[group]["loss_sums"][a] for t in traces)/n for a in ARMS}
        total["AUROC"] = {a: float(roc_auc_score(labels[group], v)) for a, v in values[group].items()}
        mass = np.array([t[group]["rows"] for t in traces])
        gain = np.array([t[group]["loss_sums"]["whole"]-t[group]["loss_sums"]["masked"] for t in traces])
        denominator = draws@mass
        total["whole_minus_masked_NLL_bootstrap95ci"] = np.percentile((draws@gain)/denominator, [2.5, 97.5]).tolist() if (denominator > 0).all() else None
        metrics[group] = total
    return metrics, traces


def decide(metrics, policy, numerical):
    tiny = metrics["tiny"]
    interval = tiny["whole_minus_masked_NLL_bootstrap95ci"]
    checks = dict(numerical=numerical, tp=tiny["tp"] >= policy["tiny_tp_min"],
                  fp=tiny["fp"] >= policy["tiny_nonduplicate_fp_min"], images=tiny["images"] >= policy["tiny_images_min"],
                  vs_row=tiny["NLL"]["row"]-tiny["NLL"]["masked"] >= policy["tiny_NLL_gain_vs_row_min"],
                  vs_whole=tiny["NLL"]["whole"]-tiny["NLL"]["masked"] >= policy["tiny_NLL_gain_vs_whole_min"],
                  vs_shuffle=tiny["NLL"]["shuffled"]-tiny["NLL"]["masked"] >= policy["tiny_NLL_gain_vs_shuffle_min"],
                  AUROC=tiny["AUROC"]["masked"]-tiny["AUROC"]["whole"] >= policy["tiny_AUROC_gain_vs_whole_min"],
                  interval=interval is not None and interval[0] > policy["tiny_NLL_gain_vs_whole_bootstrap_lower_min_exclusive"],
                  all_NLL=metrics["all"]["NLL"]["whole"]-metrics["all"]["NLL"]["masked"] >= policy["all_NLL_gain_vs_whole_min"])
    return dict(decision="SCENE_BACKGROUND_INFORMATION_SUPPORTS_SEPARATE_FRESH_SCENE_DEPLOYMENT_PILOT" if all(checks.values())
                else "STOP_THIS_SCENE_BACKGROUND_QUANTILE_LINEAR_RECIPE", checks=checks,
                scope="Repeated-train-scene information screen only, no deployed scores/AP or innovation proof.")


def run(output=OUTPUT, device="0"):
    started = perf_counter()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    output = project_path(output)
    if output.exists():
        raise FileExistsError(output)
    registration = json.loads(project_path(REGISTRATION).read_text())
    prior = project_path(PRIOR)
    _, previous = verify_record(prior, "docs/experiments/candidate12_class_choice_result_20261014.json")
    _, selection = verify_record(project_path(SELECTION), "docs/experiments/candidate18_action_order_sensitivity_result_20261014.json")
    assert len(selection["image_selection"]["train"]) == 128 and len(selection["image_selection"]["holdout"]) == 64
    control = frozen_control()
    calibrator = load_calibrator(control)
    raw_path = prior/"raw_views.jsonl"
    raw = {r["image"]: r for r in map(json.loads, raw_path.read_text().splitlines())}
    paths = (REGISTRATION, "sdcl/evaluation/scene_background.py", "scripts/evaluation/diagnose_scene_background.py",
             "tests/test_scene_background.py", "sdcl/evaluation/classifier_nullspace.py",
             "sdcl/training/class_margin_probe.py", "sdcl/training/marginal_class_intervention.py",
             "sdcl/postprocess/localization.py", "sdcl/postprocess/rescue.py", "sdcl/data/geometry.py",
             "sdcl/evaluation/size_analysis.py", "sdcl/evaluation/slicing_screen.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    installed = {n: Path(inspect.getfile(m)) for n, m in
                 (("augment.py", augment_module), ("head.py", head_module), ("nms.py", nms_module), ("ops.py", ops_module))}
    output.mkdir(parents=True)
    (output/"frames").mkdir()
    with zipfile.ZipFile(output/"source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(project_path(p), p)
        for n, path in installed.items():
            z.write(path, "installed/"+n)
    plan = dict(registration=registration, source_hashes=hashes, source_archive_sha256=file_hash(output/"source.zip"),
                installed_sources={n: dict(path=str(p), sha256=file_hash(p)) for n, p in installed.items()},
                prior_plan_sha256=file_hash(prior/"plan.json"), raw_views_sha256=file_hash(raw_path),
                selection_plan_sha256=file_hash(project_path(SELECTION)/"plan.json"),
                image_selection=selection["image_selection"], records=previous["records"],
                strong_control_manifest_sha256=file_hash(project_path("output/analysis/calibration-control-supplement-20261013/control_manifest.json")))
    assert len(plan["image_selection"]["train"]) == 128 and len(plan["image_selection"]["holdout"]) == 64
    save_json(output/"plan.json", plan)
    record_lookup = {r["stem"]: r for rs in previous["records"].values() for r in rs}
    frames, manifest, forwards = dict(train=[], holdout=[]), {}, 0
    extractor = FrozenViews(device)
    try:
        with torch.inference_mode():
            for split, stems in plan["image_selection"].items():
                for stem in stems:
                    image_path = project_path(f"data/visdrone/images/train/{stem}.jpg")
                    metadata_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
                    record = record_lookup[stem]
                    assert file_hash(image_path) == record["image_sha256"] and file_hash(metadata_path) == record["metadata_sha256"]
                    metadata = json.loads(metadata_path.read_text())
                    item = raw[stem]
                    views = [np.array(v, float).reshape(-1, 6) for v in item["views"]]
                    reference, _, _, base = calibrated_candidates(views, item["windows"], metadata["original_size"], [], calibrator)
                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise ValueError("Unreadable fixed image.")
                    whole, masked, fraction, logits, masks, fallback = [], [], [], [], [], []
                    for view, window in enumerate(item["windows"]):
                        x1, y1, x2, y2 = window
                        prediction, _ = extractor.predict(image[y1:y2, x1:x2].copy())
                        kept = map_tile_predictions(prediction["control"], window) if view else prediction["control"]
                        if not np.array_equal(kept, np.array(views[view], np.float32)):
                            raise ValueError("Frozen five-view raw replay differs; no control change allowed.")
                        native = extractor.raw["scores"][0].T.detach().cpu().numpy().copy()
                        if native.shape != (18900, 10) or not np.isfinite(native).all():
                            raise ValueError("Unexpected or nonfinite native score inventory.")
                        w, m, f, mask, back = view_statistics(native, prediction["control"], x2-x1, y2-y1)
                        whole.append(w); masked.append(m); fraction.append(f)
                        logits.append(native); masks.append(mask); fallback.append(back)
                        forwards += 1
                    whole, masked, fraction = np.array(whole), np.array(masked), np.array(fraction)
                    classes = reference[:, 5].astype(int)
                    tp, _, ignored = labels_for(reference, metadata, .05)
                    base = np.column_stack((base, np.eye(10)[classes])).astype(float)
                    f = dict(stem=stem, reference=reference, metadata=metadata, base=base,
                             scene_whole=whole, scene_masked=masked,
                             whole=whole[:, :, classes, :].transpose(2, 0, 1, 3).reshape(len(reference), 60),
                             masked=masked[:, :, classes, :].transpose(2, 0, 1, 3).reshape(len(reference), 60),
                             fraction=np.broadcast_to(fraction.ravel(), (len(reference), 15)).copy(),
                             fit_labels=tp, fit_eligible=(reference[:, 4] >= .05) & ~ignored)
                    frames[split].append(f)
                    path = output/"frames"/(stem+".npz")
                    np.savez_compressed(path, reference=reference, base=base, scene_whole=whole,
                                        scene_masked=masked, fraction=fraction, native_logits=np.array(logits),
                                        masks=np.array(masks), fallback=np.array(fallback),
                                        fit_labels=tp, fit_eligible=f["fit_eligible"])
                    manifest[stem] = dict(file=path.relative_to(output).as_posix(), sha256=file_hash(path),
                                          split=split, image_sha256=record["image_sha256"], metadata_sha256=record["metadata_sha256"])
                    if len(manifest) % 32 == 0:
                        print(f"Scene background response: {len(manifest)}/192", flush=True)
    finally:
        extractor.close()
    stats = normalize(frames)
    coefficients, training = {}, {}
    eligible = [f["fit_eligible"] for f in frames["train"]]
    confidence = np.concatenate([f["reference"][m, 4] for f, m in zip(frames["train"], eligible)])
    offset = np.log(np.clip(confidence, 1e-6, 1-1e-6)/(1-np.clip(confidence, 1e-6, 1-1e-6)))
    labels = np.concatenate([f["fit_labels"][m] for f, m in zip(frames["train"], eligible)])
    for arm in ARMS:
        x = np.concatenate([f["inputs"][arm][m] for f, m in zip(frames["train"], eligible)])
        coefficients[arm], training[arm] = fit(x, offset, labels)
        print(f"Scene linear {arm}: {training[arm]}", flush=True)
    metrics, traces = summarize(frames["holdout"], coefficients)
    (output/"inputs").mkdir()
    for split, fs in frames.items():
        for f in fs:
            path = output/"inputs"/(f["stem"]+".npz")
            np.savez_compressed(path, **f["inputs"], fit_labels=f["fit_labels"], fit_eligible=f["fit_eligible"],
                                **({a+"_logits": f["output_logits"][a] for a in ARMS} if split == "holdout" else {}))
            manifest[f["stem"]].update(inputs_file=path.relative_to(output).as_posix(), inputs_sha256=file_hash(path),
                                       shuffled_stem=f["shuffled_stem"])
    np.savez_compressed(output/"coefficients.npz", **coefficients)
    decision = decide(metrics, registration["policy"], all(t["success"] for t in training.values()))
    verification = dict(fit_images=128, holdout_images=64, JPEG_decodes=192, frozen_forwards_excluding_warmup=forwards,
                        all960_raw_kept_replays_exact=True, detector_updates=0, linear_models_fit=4, AP_evaluations=0,
                        reference_has_no_GT_ignore_filter=True, no_GT_in_scene_inputs=True,
                        source_unchanged=all(file_hash(project_path(p)) == h for p, h in hashes.items()),
                        installed_sources_unchanged=all(file_hash(p) == plan["installed_sources"][n]["sha256"] for n, p in installed.items()),
                        elapsed_seconds=perf_counter()-started)
    for name, value in (("frames.json", manifest), ("normalization.json", stats), ("training.json", training),
                        ("per_image.json", traces), ("metrics.json", metrics), ("decision.json", decision), ("verification.json", verification)):
        save_json(output/name, value)
    result = dict(experiment=registration["experiment"], date="2026-10-15", output=str(output),
                  metrics=metrics, training=training, decision=decision, verification=verification,
                  artifacts_sha256={p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()})
    save_json(output/"result.json", result)
    return result
