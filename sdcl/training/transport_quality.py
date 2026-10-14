"""Preregistered small pre-NMS quality pilot; detector and calibrator stay frozen."""

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import re
from time import perf_counter
import zipfile

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from sdcl.config import project_path
from sdcl.data.geometry import prediction_ignore_mask
from sdcl.evaluation.official_details import NAMES
from sdcl.evaluation.size_analysis import box_iou_matrix, coco_size_metrics, file_hash, recall_at_fp_budgets
from sdcl.evaluation.slicing_screen import map_tile_predictions, tile_windows
from sdcl.evaluation.texture_deletion import original_inventory
from sdcl.postprocess.localization import calibrated_candidates, frozen_control, load_calibrator
from sdcl.postprocess.transport_quality import (
    ARMS, QualityHead, active_indices, adjusted_decoded, normalized, quality_features,
)
from sdcl.training.class_margin_probe import FrozenViews


REGISTRATION = "docs/experiments/candidate62_transport_quality_registration_20261015.json"
OUTPUT = "output/analysis/transport-quality-fixed-20261015"
STEMS = r"\d{7}_\d{5}_[a-z]_\d{7}"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def select_images():
    excluded, archives = set(), {}
    for path in sorted(project_path("output/analysis").glob("*/plan.json")):
        archives[path.relative_to(project_path(".")).as_posix()] = file_hash(path)
        excluded.update(re.findall(STEMS, path.read_text(encoding="utf-8")))
    groups = {}
    for path in project_path("data/visdrone/images/train").glob("*.jpg"):
        if path.stem not in excluded:
            groups.setdefault(path.stem[:7], []).append(path.stem)
    seen_prefix = {s[:7] for s in excluded}
    fresh = sorted(set(groups)-seen_prefix)
    if len(fresh) != 5:
        raise ValueError("Preregistered five confirmation sequences unavailable.")
    other = sorted(set(groups)-set(fresh), key=lambda s: digest("candidate62-scene-v1:"+s))
    if len(other) < 96:
        raise ValueError("Not enough fixed fit/screen sequences.")
    single = [min(groups[s], key=lambda t: digest("candidate62-image-v1:"+t)) for s in other[:96]]
    confirmation = [stem for s in fresh for stem in
                    sorted(groups[s], key=lambda t: digest("candidate62-confirm-image-v1:"+t))[:4]]
    selection = dict(fit=single[:64], screen=single[64:], confirmation=confirmation)
    assert len({s[:7] for stems in selection.values() for s in stems}) == 101
    return selection, archives, fresh


def native_nms(decoded, shape):
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    prediction = non_max_suppression(decoded.clone(), .001, .7, max_det=500)[0]
    prediction[:, :4] = scale_boxes((960, 960), prediction[:, :4], shape)
    return prediction.float().cpu().numpy()


def supervised_targets(boxes, classes, metadata):
    objects = metadata["valid_objects"]
    gt = np.array([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
    labels = np.array([o["class_id"] for o in objects], int)
    overlap = box_iou_matrix(gt, boxes)
    maximum = np.where(labels[:, None] == classes[None], overlap, 0).max(0) if len(gt) else np.zeros(len(boxes))
    target = np.where(maximum >= .5, maximum, 0).astype(np.float32)
    ignored = prediction_ignore_mask(torch.from_numpy(boxes.copy()),
        torch.tensor(metadata["ignore_boxes_xyxy"], dtype=torch.float32).reshape(-1, 4)).numpy()
    return target, ~ignored


def sample_rows(target, eligible, stem, view):
    rng = np.random.default_rng(int(digest(f"candidate62-sample-v1:{stem}:{view}")[:16], 16))
    rows, weights = [], []
    for state in (False, True):
        pool = np.flatnonzero(eligible & ((target > 0) == state))
        count = min(len(pool), 128)
        if count:
            rows.extend(np.sort(rng.choice(pool, count, replace=False)))
            weights.extend([len(pool)/count]*count)
    return np.asarray(rows, int), np.asarray(weights, np.float32)


def extract(root, selection, device, calibrator):
    extractor = FrozenViews(device)
    manifest, count = {}, 0
    image_count = sum(map(len, selection.values()))
    (root/"cache").mkdir()
    try:
        with torch.inference_mode():
            for split, stems in selection.items():
                for stem in stems:
                    image_path = project_path(f"data/visdrone/images/train/{stem}.jpg")
                    meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise ValueError("Unreadable preregistered image.")
                    height, width = image.shape[:2]
                    windows = [(0, 0, width, height), *tile_windows(width, height)]
                    pending, views = [], []
                    for view, window in enumerate(windows):
                        x1, y1, x2, y2 = window
                        pixels = image[y1:y2, x1:x2].copy()
                        outputs, _ = extractor.predict(pixels)
                        decoded = extractor.decoded[0].cpu().numpy().copy()
                        dfl = extractor.raw["boxes"][0].T.reshape(18900, 4, 16).cpu().numpy().copy()
                        classes = extractor.raw["scores"][0].T.cpu().numpy().copy()
                        anchors = extractor.head.anchors.T.float().cpu().numpy().copy()
                        strides = extractor.head.strides.reshape(-1).float().cpu().numpy().copy()
                        np.testing.assert_array_equal(native_nms(extractor.decoded, pixels.shape[:2]), outputs["control"])
                        local_boxes, _ = original_inventory(extractor, x2-x1, y2-y1)
                        boxes = local_boxes+np.array([x1, y1, x1, y1], np.float32)
                        indices = active_indices(decoded)
                        inputs = quality_features(decoded, dfl, classes, boxes, anchors, strides,
                                                  window, [width, height], indices)
                        pending.append(dict(decoded=decoded, raw_dfl=dfl, raw_class=classes,
                            boxes=boxes, anchors=anchors, strides=strides, indices=indices,
                            original_local=outputs["control"], **{"input_"+a: inputs[a] for a in ARMS}))
                        views.append(map_tile_predictions(outputs["control"], window) if view else outputs["control"])
                    # Labels are consulted only after every prediction and input is fixed.
                    metadata = read(meta_path)
                    assert metadata["original_size"] == [width, height]
                    reference = calibrated_candidates(views, windows, [width, height], [], calibrator)[0]
                    entries = []
                    for view, frame in enumerate(pending):
                        indices = frame["indices"]
                        predicted_class = frame["decoded"][4:14, indices].argmax(0)
                        target, eligible = supervised_targets(frame["boxes"][indices], predicted_class, metadata)
                        rows, weights = sample_rows(target, eligible, stem, view)
                        probability = frame["decoded"][4:14, indices].max(0).astype(np.float32)
                        probability = np.clip(probability, 1e-6, 1-1e-6)
                        frame.update(target=target, eligible=eligible, sample_rows=rows, sample_weights=weights,
                                     offset=np.log(probability/(1-probability)))
                        path = root/"cache"/f"{stem}-v{view}.npz"
                        np.savez_compressed(path, **frame)
                        entries.append(dict(file=path.relative_to(root).as_posix(), sha256=file_hash(path)))
                    path = root/"cache"/f"{stem}-reference.npy"
                    np.save(path, reference)
                    manifest[stem] = dict(split=split, windows=windows, size=[width, height],
                        views=entries, reference=dict(file=path.relative_to(root).as_posix(), sha256=file_hash(path)),
                        image_sha256=file_hash(image_path), metadata_sha256=file_hash(meta_path))
                    count += 1
                    if count == 1 or count % 8 == 0 or count == image_count:
                        print(f"Transport quality extraction: {count}/{image_count}", flush=True)
    finally:
        extractor.close()
    save(root/"frames.json", manifest)
    return manifest


def load_cache(root, entry):
    path = root/entry["file"]
    if file_hash(path) != entry["sha256"]:
        raise ValueError("Frozen pilot cache changed.")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def fit_seed(root, manifest, selection, device, seed, normal=None):
    collected = {a: [] for a in ARMS}
    targets, offsets, weights = [], [], []
    for stem in selection["fit"]:
        for entry in manifest[stem]["views"]:
            frame = load_cache(root, entry)
            rows = frame["sample_rows"]
            for arm in ARMS:
                collected[arm].append(frame["input_"+arm][rows])
            targets.append(frame["target"][rows])
            offsets.append(frame["offset"][rows])
            weights.append(frame["sample_weights"])
    arrays = {a: np.concatenate(v) for a, v in collected.items()}
    if normal is None:
        normal = {a: dict(mean=x.mean(0).tolist(), std=np.maximum(x.std(0), 1e-6).tolist())
                  for a, x in arrays.items()}
        save(root/"normalization.json", normal)
    inputs = {a: torch.tensor(normalized(x, normal[a]["mean"], normal[a]["std"]), device=device)
              for a, x in arrays.items()}
    y, offset, weight = [torch.tensor(np.concatenate(v), device=device) for v in (targets, offsets, weights)]
    weight /= weight.mean()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initial = QualityHead().to(device)
    models = {a: deepcopy(initial) for a in ARMS}
    optimizers = {a: torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.0001) for a, m in models.items()}
    history = []
    for epoch in range(8):
        order = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed*100+epoch)).to(device)
        losses = dict.fromkeys(ARMS, 0.)
        for start in range(0, len(order), 2048):
            rows = order[start:start+2048]
            for arm in ARMS:
                prediction = offset[rows]+models[arm](inputs[arm][rows])
                loss = (F.binary_cross_entropy_with_logits(prediction, y[rows], reduction="none")*weight[rows]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite quality fit; preserve fixed recipe.")
                optimizers[arm].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[arm].step()
                losses[arm] += float(loss.detach())*len(rows)
        history.append(dict(epoch=epoch+1, losses={a: v/len(y) for a, v in losses.items()}))
        print(f"Transport quality seed{seed} epoch{epoch+1}/8", flush=True)
    torch.save(dict(seed=seed, models={a: {k: v.detach().cpu() for k, v in m.state_dict().items()}
                                     for a, m in models.items()}, fit_rows=len(y),
                    registration_sha256=file_hash(project_path(REGISTRATION)),
                    normalization_sha256=file_hash(root/"normalization.json")), root/f"heads-seed{seed}.pt")
    save(root/f"fit-seed{seed}.json", history)
    return models, normal


def diagnostic_filter(values, metadata):
    ignored = prediction_ignore_mask(torch.tensor(values[:, :4].copy()),
        torch.tensor(metadata["ignore_boxes_xyxy"], dtype=torch.float64).reshape(-1, 4)).numpy()
    return values[~ignored]


def mechanism_summary(rows):
    prefixes = sorted({r["stem"][:7] for r in rows})
    counts = np.array([sum(r["rows"] for r in rows if r["stem"][:7] == s) for s in prefixes])
    delta = np.array([sum(r["loss_sums"]["centered"]-r["loss_sums"]["transport"]
                         for r in rows if r["stem"][:7] == s) for s in prefixes])
    draws = np.random.default_rng(0).multinomial(len(prefixes), [1/len(prefixes)]*len(prefixes), size=400)
    denominator = draws@counts
    ci = np.percentile((draws@delta)/denominator, [2.5, 97.5]).tolist() if (denominator > 0).all() else None
    mass = int(counts.sum())
    return dict(rows=mass, sequence_units=len(prefixes),
                NLL={a: sum(r["loss_sums"][a] for r in rows)/mass if mass else None for a in ARMS},
                centered_minus_transport_NLL_sequence_bootstrap95ci=ci)


def evaluate_seed(root, manifest, selection, device, seed, models, normal, calibrator):
    prediction_dir = root/f"predictions-seed{seed}"
    prediction_dir.mkdir()
    reports, prediction_manifest = {}, {}
    with torch.inference_mode():
        for split in ("screen", "confirmation"):
            records, losses = [], []
            maps = {a: {} for a in ("strong", *ARMS)}
            for count, stem in enumerate(selection[split], 1):
                entry = manifest[stem]
                metadata = read(project_path(f"data/visdrone/metadata/train/{stem}.json"))
                if file_hash(project_path(f"data/visdrone/metadata/train/{stem}.json")) != entry["metadata_sha256"]:
                    raise ValueError("Pilot metadata changed.")
                records.append(dict(id=count, stem=stem, size=entry["size"], objects=metadata["valid_objects"]))
                views = {a: [] for a in ("strong", *ARMS)}
                totals, population = dict.fromkeys(ARMS, 0.), 0
                frame_shifts = []
                for view, snapshot in enumerate(entry["views"]):
                    frame = load_cache(root, snapshot)
                    decoded = torch.tensor(frame["decoded"][None], device=device)
                    window = entry["windows"][view]
                    shape = (window[3]-window[1], window[2]-window[0])
                    original = native_nms(decoded, shape)
                    np.testing.assert_array_equal(original, frame["original_local"])
                    views["strong"].append(map_tile_predictions(original, window) if view else original)
                    shifts = {}
                    population += int(frame["eligible"].sum())
                    for arm in ARMS:
                        x = normalized(frame["input_"+arm], normal[arm]["mean"], normal[arm]["std"])
                        shift = models[arm](torch.tensor(x, device=device)).cpu().numpy()
                        shifts[arm] = shift
                        z = frame["offset"].astype(float)+shift
                        totals[arm] += float((np.logaddexp(0, z)-frame["target"]*z)[frame["eligible"]].sum())
                        changed = adjusted_decoded(decoded, frame["indices"], shift)
                        local = native_nms(changed, shape)
                        views[arm].append(map_tile_predictions(local, window) if view else local)
                    frame_shifts.extend({"shifts_v"+str(view)+"_"+a: z for a, z in shifts.items()}.items())
                arrays = dict(frame_shifts)
                for arm in views:
                    output = calibrated_candidates(views[arm], entry["windows"], entry["size"], [], calibrator)[0]
                    if arm == "strong":
                        reference = root/entry["reference"]["file"]
                        assert file_hash(reference) == entry["reference"]["sha256"]
                        np.testing.assert_array_equal(output, np.load(reference, allow_pickle=False))
                    arrays[arm] = output
                    maps[arm][stem] = diagnostic_filter(output, metadata)
                    arrays["diagnostic_"+arm] = maps[arm][stem]
                path = prediction_dir/(stem+".npz")
                np.savez_compressed(path, **arrays)
                prediction_manifest[stem] = dict(split=split, file=path.relative_to(root).as_posix(), sha256=file_hash(path))
                losses.append(dict(stem=stem, rows=population, loss_sums=totals))
                if count == 1 or count % 8 == 0:
                    print(f"Transport quality evaluation seed{seed} {split}: {count}/{len(selection[split])}", flush=True)
            print(f"Transport quality COCO diagnostics seed{seed} {split}", flush=True)
            ap = {a: coco_size_metrics(records, mapping, dict(enumerate(NAMES)), reference_size=640)
                  for a, mapping in maps.items()}
            fixed_fp = {}
            for arm in ARMS:
                comparison, _ = recall_at_fp_budgets(records, dict(baseline=maps["strong"], sdcl=maps[arm]),
                                                     budgets=(10,), bootstrap_samples=0)
                fixed_fp[arm] = comparison
            reports[split] = dict(AP=ap, fixed_fp=fixed_fp, mechanism=mechanism_summary(losses), per_image_losses=losses)
    save(root/f"predictions-seed{seed}.json", prediction_manifest)
    save(root/f"metrics-seed{seed}.json", reports)
    return reports


def decide(reports, policy):
    checks = {}
    for split, metrics in reports.items():
        ap, mechanism = metrics["AP"], metrics["mechanism"]
        def gain(group, control):
            x, y = ap["transport"][group]["ap50_95"], ap[control][group]["ap50_95"]
            return None if x is None or y is None else (x-y)*100
        checks[split] = dict(tiny_GT=ap["strong"]["tiny"]["gt"] >= policy["tiny_GT_min"])
        comparisons = (("tiny", "strong"), ("small_all", "strong"), ("all", "strong"),
                       ("tiny", "geometry"), ("tiny", "dgqp"), ("tiny", "centered"))
        for group, control in comparisons:
            name = f"{group}_ap_vs_{control}"
            difference = gain(group, control)
            checks[split][name] = difference is not None and difference >= policy[name+"_min_pp"]
        delta = metrics["fixed_fp"]["transport"][0]["delta_recall_pp"]["small_all"]
        checks[split]["fixed_FP"] = delta is not None and delta >= policy["small_all_recall_fppi10_vs_strong_min_pp"]
        nll, ci = mechanism["NLL"], mechanism["centered_minus_transport_NLL_sequence_bootstrap95ci"]
        checks[split]["NLL"] = nll["centered"] is not None and nll["transport"] is not None and (
            nll["centered"]-nll["transport"] >= policy["NLL_gain_vs_centered_min"])
        checks[split]["sequence_interval"] = ci is not None and ci[0] > policy[
            "NLL_gain_vs_centered_sequence_bootstrap_lower_min_exclusive"]
    return dict(checks=checks, passed=all(all(v.values()) for v in checks.values()))


def run(device="0"):
    started = perf_counter()
    root = project_path(OUTPUT)
    if root.exists():
        raise FileExistsError(root)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    gpu = "cuda:"+str(device)
    selection, exclusions, fresh = select_images()
    registration = read(project_path(REGISTRATION))
    control, calibrator = frozen_control(), load_calibrator(frozen_control())
    sources = (REGISTRATION, "docs/experiments/candidate62_transport_quality_preflight_original_20261015.json",
        "docs/literature/transport_quality_prior_audit_20261015.json",
        "sdcl/postprocess/transport_quality.py", "sdcl/training/transport_quality.py",
        "scripts/training/run_transport_quality.py", "tests/test_transport_quality.py",
        "sdcl/postprocess/native_spatial_field.py", "sdcl/data/geometry.py", "sdcl/evaluation/official_details.py",
        "sdcl/evaluation/size_analysis.py", "sdcl/evaluation/slicing_screen.py", "sdcl/evaluation/texture_deletion.py",
        "sdcl/postprocess/localization.py", "sdcl/postprocess/rescue.py", "sdcl/training/class_margin_probe.py",
        "sdcl/config.py")
    hashes = {s: file_hash(project_path(s)) for s in sources}
    from ultralytics.nn.modules import head
    from ultralytics.utils import ops, nms
    installed = {inspect.getfile(m): file_hash(inspect.getfile(m)) for m in (head, ops, nms)}
    root.mkdir(parents=True)
    with zipfile.ZipFile(root/"source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in sources:
            z.write(project_path(name), name)
    plan = dict(registration=registration, image_selection=selection, exclusion_plan_hashes=exclusions,
        confirmation_fresh_prefixes=fresh, source_hashes=hashes, installed_source_hashes=installed,
        source_archive_sha256=file_hash(root/"source.zip"), device=gpu,
        prior61_closeout_sha256=file_hash(project_path(
            "docs/experiments/candidate61_peak_gap_survival_result_20261015.json")),
        detector_sha256=control["detector"]["sha256"], calibration_sha256=control["scorer"]["sha256"])
    save(root/"plan.json", plan)
    manifest = extract(root, selection, device, calibrator)
    decisions, normal = {}, None
    for seed in (0, 1, 2):
        models, normal = fit_seed(root, manifest, selection, gpu, seed, normal)
        reports = evaluate_seed(root, manifest, selection, gpu, seed, models, normal, calibrator)
        decisions[str(seed)] = decide(reports, registration["policy_per_half"])
        save(root/f"decision-seed{seed}.json", decisions[str(seed)])
        if seed == 0 and not decisions["0"]["passed"]:
            break
    full_pass = sum(d["passed"] for d in decisions.values()) >= 2 and all(
        v["all_ap_vs_strong"] and v["fixed_FP"] for d in decisions.values() for v in d["checks"].values())
    verdict = "REPLICATED_DEPLOYABLE_TRANSPORT_QUALITY_PILOT_SUPPORTS_OFFICIAL_AND_EXTERNAL_AUDIT" if full_pass else (
        "STOP_THIS_COMMON_COORDINATE_W1_PRENMS_QUALITY_RECIPE")
    assert all(file_hash(project_path(s)) == d for s, d in hashes.items())
    assert all(file_hash(s) == d for s, d in installed.items())
    frozen_control()
    result = dict(experiment=registration["experiment"], decision=verdict, seeds=decisions,
        output=str(root), costs=dict(JPEG_decodes=sum(map(len, selection.values())),
        detector_forwards=5*sum(map(len, selection.values())), predictor_warmup_forwards=1,
        detector_updates=0, tiny_head_fits=4*len(decisions), epochs_each=8, official_AP_evaluations=0,
        elapsed_seconds=perf_counter()-started),
        limits="COCO-style individual-ignore diagnostics, not officialVisDroneAP. Detector saw train data; five plan-unlisted confirmation sequences only. No innovation claim,external replication or full training recommendation.",
        artifacts_sha256={p.name: file_hash(p) for p in sorted(root.iterdir()) if p.is_file()})
    save(root/"result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result
