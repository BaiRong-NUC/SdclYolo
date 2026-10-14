"""Preregistered paired texture-deletion information screen on fresh train images."""

import hashlib
import inspect
import json
from pathlib import Path
import re
from time import perf_counter
import zipfile

import cv2
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import torch
from ultralytics.data import augment as augment_module
from ultralytics.nn.modules import head as head_module
from ultralytics.utils import nms as nms_module, ops as ops_module

from sdcl.config import project_path
from sdcl.evaluation.classifier_nullspace import fit
from sdcl.evaluation.scene_background import labels_for
from sdcl.evaluation.size_analysis import box_iou_matrix, file_hash
from sdcl.evaluation.slicing_screen import map_tile_predictions, tile_windows
from sdcl.postprocess.localization import calibrated_candidates, frozen_control, load_calibrator
from sdcl.postprocess.rescue import save_json
from sdcl.training.class_margin_probe import FrozenViews


REGISTRATION = "docs/experiments/candidate56_texture_deletion_registration_20261015.json"
OUTPUT = "output/analysis/texture-deletion-fixed-20261015"
ARMS = ("base", "sham", "own", "shuffled")
SPLITS = ("fit", "screen", "confirmation")
STEMS = r"\d{7}_\d{5}_[a-z]_\d{7}"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def patch_rectangles(box, width, height):
    own = np.r_[np.floor(box[:2]), np.ceil(box[2:])].astype(int)
    own[[0, 2]] = own[[0, 2]].clip(0, width)
    own[[1, 3]] = own[[1, 3]].clip(0, height)
    w, h = own[2:] - own[:2]
    if w <= 0 or h <= 0:
        return None
    for dx, dy in ((2*w, 0), (-2*w, 0), (0, 2*h), (0, -2*h)):
        sham = own + [dx, dy, dx, dy]
        if (sham[:2] >= 0).all() and (sham[2:] <= [width, height]).all():
            return own, sham
    return None


def erase(image, rectangle):
    x1, y1, x2, y2 = rectangle
    pixels = image[y1:y2, x1:x2].astype(float)
    mean = pixels.mean((0, 1))
    texture = np.r_[mean, pixels.std((0, 1))]
    fill = np.rint(mean).astype(np.uint8)
    altered = image.copy()
    altered[y1:y2, x1:x2] = fill
    return altered, texture, fill


def original_inventory(extractor, width, height):
    from ultralytics.utils.ops import scale_boxes, xywh2xyxy
    boxes = xywh2xyxy(extractor.decoded[0, :4].T).float()
    boxes = scale_boxes((960, 960), boxes, (height, width)).cpu().numpy()
    logits = extractor.raw["scores"][0].T.detach().cpu().numpy().copy()
    if boxes.shape != (18900, 4) or logits.shape != (18900, 10):
        raise ValueError("Unexpected native inventory.")
    return boxes, logits


def donors(frames):
    entries = [(i, j) for i, f in enumerate(frames) for j in range(len(f["selected"]))]
    mapping = {}
    for cls in range(10):
        rows = [(i, j) for i, j in entries if int(frames[i]["reference"][frames[i]["selected"][j], 5]) == cls]
        rows.sort(key=lambda ij: digest("candidate56-donor-v1:"+frames[ij[0]]["stem"]+":"+
                                       str(int(frames[ij[0]]["selected"][ij[1]]))))
        for k, ij in enumerate(rows):
            mapping[ij] = next((rows[(k+d) % len(rows)] for d in range(1, len(rows))
                                if rows[(k+d) % len(rows)][0] != ij[0]), None)
    return mapping


def build_inputs(frames):
    normal = {}
    for key in ("base", "sham", "own"):
        values = np.concatenate([f[key][f["eligible"]] for f in frames["fit"]])
        if not len(values):
            raise ValueError("No fit population; do not change the sample.")
        normal[key] = dict(mean=values.mean(0).tolist(), std=np.maximum(values.std(0), 1e-6).tolist())
    for split, fs in frames.items():
        mapping = donors(fs)
        for i, f in enumerate(fs):
            base = (f["base"]-normal["base"]["mean"])/normal["base"]["std"]
            sham = (f["sham"]-normal["sham"]["mean"])/normal["sham"]["std"]
            own = (f["own"]-normal["own"]["mean"])/normal["own"]["std"]
            shuffled = np.zeros_like(own)
            donor_rows = np.full((len(own), 2), -1, int)
            for j in range(len(own)):
                donor = mapping[(i, j)]
                if donor is None:
                    own[j] = 0
                else:
                    di, dj = donor
                    shuffled[j] = (fs[di]["own"][dj]-normal["own"]["mean"])/normal["own"]["std"]
                    donor_rows[j] = [di, dj]
            zero = np.zeros_like(own)
            f["donors"] = donor_rows
            f["inputs"] = dict(base=np.column_stack((base, zero, zero)),
                               sham=np.column_stack((base, sham, zero)),
                               own=np.column_stack((base, sham, own)),
                               shuffled=np.column_stack((base, sham, shuffled)))
    return normal


def summarize(frames, coefficients):
    traces, labels, outputs = [], [], {a: [] for a in ARMS}
    for f in frames:
        selected, keep = f["selected"], f["eligible"]
        p = np.clip(f["reference"][selected, 4], 1e-6, 1-1e-6)
        offset = np.log(p/(1-p))
        y = f["labels"][selected]
        values = {a: offset+f["inputs"][a]@theta[:-1]+theta[-1] for a, theta in coefficients.items()}
        f["output_logits"] = values
        labels.extend(y[keep])
        for a in ARMS:
            outputs[a].extend(values[a][keep])
        traces.append(dict(rows=int(keep.sum()), tp=int(y[keep].sum()), fp=int((~y[keep]).sum()),
                           images=int(keep.any()),
                           loss_sums={a: float((np.logaddexp(0, z[keep])-y[keep]*z[keep]).sum())
                                      for a, z in values.items()}))
    total = {k: sum(t[k] for t in traces) for k in ("rows", "tp", "fp", "images")}
    total["NLL"] = {a: sum(t["loss_sums"][a] for t in traces)/total["rows"] if total["rows"] else None for a in ARMS}
    total["AUROC"] = {a: float(roc_auc_score(labels, z)) if total["tp"] and total["fp"] else None
                      for a, z in outputs.items()}
    draws = np.random.default_rng(0).multinomial(64, [1/64]*64, size=400)
    mass = draws@np.array([t["rows"] for t in traces])
    difference = draws@np.array([t["loss_sums"]["sham"]-t["loss_sums"]["own"] for t in traces])
    total["sham_minus_own_NLL_bootstrap95ci"] = np.percentile(difference/mass, [2.5, 97.5]).tolist() if (mass > 0).all() else None
    return total, traces


def decide(metrics, policy, numerical):
    checks = {}
    for half, m in metrics.items():
        nll, auc, ci = m["NLL"], m["AUROC"], m["sham_minus_own_NLL_bootstrap95ci"]
        populated = bool(m["tp"] and m["fp"])
        checks[half] = dict(tp=m["tp"] >= policy["tp_min"],
                            fp=m["fp"] >= policy["nonduplicate_fp_min"],
                            images=m["images"] >= policy["images_min"],
                            vs_base=populated and nll["base"]-nll["own"] >= policy["NLL_gain_vs_base_min"],
                            vs_sham=populated and nll["sham"]-nll["own"] >= policy["NLL_gain_vs_sham_min"],
                            vs_shuffled=populated and nll["shuffled"]-nll["own"] >= policy["NLL_gain_vs_shuffled_min"],
                            AUROC=populated and auc["own"]-auc["sham"] >= policy["AUROC_gain_vs_sham_min"],
                            interval=ci is not None and ci[0] > policy["NLL_gain_vs_sham_bootstrap_lower_min_exclusive"])
    passed = numerical and all(all(v.values()) for v in checks.values())
    return dict(numerical=numerical, checks=checks,
                decision="REPLICATED_TEXTURE_DELETION_INFORMATION_SUPPORTS_SEPARATE_DEPLOYMENT_AND_COST_PILOT" if passed else
                         "STOP_THIS_PAIRED_MEAN_FILL_TEXTURE_DELETION_RECIPE",
                scope="Fixed-query diagnostic on fresh images from detector-training data; no AP, deployed gain, efficient amortization or novelty proof.")


def run(output=OUTPUT, device="0"):
    tick = perf_counter()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    root = project_path(output)
    if root.exists():
        raise FileExistsError(root)
    registration = json.loads(project_path(REGISTRATION).read_text())
    control = frozen_control()
    calibrator = load_calibrator(control)
    excluded, exclusions = set(), {}
    for p in sorted(project_path("output/analysis").glob("*/plan.json")):
        exclusions[p.relative_to(project_path(".")).as_posix()] = file_hash(p)
        excluded.update(re.findall(STEMS, p.read_text(encoding="utf-8")))
    groups = {}
    for p in project_path("data/visdrone/images/train").glob("*.jpg"):
        if p.stem not in excluded:
            groups.setdefault(p.stem[:7], []).append(p)
    scenes = sorted(groups, key=lambda s: digest("candidate56-scene-v1:"+s))[:192]
    assert len(scenes) == 192
    images = [min(groups[s], key=lambda p: digest("candidate56-image-v1:"+p.stem)) for s in scenes]
    selection = {s: [p.stem for p in images[i*64:(i+1)*64]] for i, s in enumerate(SPLITS)}
    paths = (REGISTRATION, "sdcl/evaluation/texture_deletion.py", "scripts/evaluation/diagnose_texture_deletion.py",
             "tests/test_texture_deletion.py", "sdcl/config.py", "sdcl/evaluation/classifier_nullspace.py",
             "sdcl/evaluation/scene_background.py", "sdcl/evaluation/size_analysis.py",
             "sdcl/evaluation/slicing_screen.py", "sdcl/postprocess/localization.py",
             "sdcl/postprocess/rescue.py", "sdcl/training/class_margin_probe.py")
    sources = {p: file_hash(project_path(p)) for p in paths}
    installed = {n: Path(inspect.getfile(m)) for n, m in
                 (("augment.py", augment_module), ("head.py", head_module), ("nms.py", nms_module), ("ops.py", ops_module))}
    root.mkdir(parents=True)
    (root/"frames").mkdir()
    with zipfile.ZipFile(root/"source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(project_path(p), p)
        for n, p in installed.items():
            z.write(p, "installed/"+n)
    plan = dict(registration=registration, source_hashes=sources, source_archive_sha256=file_hash(root/"source.zip"),
                installed_sources={n: dict(path=str(p), sha256=file_hash(p)) for n, p in installed.items()},
                exclusions=exclusions, excluded_unique_images=len(excluded), image_selection=selection,
                inputs={p.stem: dict(image_sha256=file_hash(p),
                                    metadata_sha256=file_hash(project_path(f"data/visdrone/metadata/train/{p.stem}.json")))
                        for p in images},
                control_manifest_sha256=file_hash(project_path("output/analysis/calibration-control-supplement-20261013/control_manifest.json")),
                checkpoint_sha256=control["detector"]["sha256"], selection_saved_before_decode=True)
    save_json(root/"plan.json", plan)
    frames, manifest, native_calls, perturbed_calls = {s: [] for s in SPLITS}, {}, 0, 0
    extractor = FrozenViews(device)
    try:
        with torch.inference_mode():
            for split, stems in selection.items():
                for stem in stems:
                    image = cv2.imread(str(project_path(f"data/visdrone/images/train/{stem}.jpg")))
                    metadata = json.loads(project_path(f"data/visdrone/metadata/train/{stem}.json").read_text())
                    width, height = image.shape[1], image.shape[0]
                    assert [width, height] == metadata["original_size"]
                    windows = [[0, 0, width, height], *tile_windows(width, height)]
                    views, locals_, boxes, logits = [], [], [], []
                    for view, window in enumerate(windows):
                        x1, y1, x2, y2 = window
                        prediction, _ = extractor.predict(image[y1:y2, x1:x2].copy())
                        local = prediction["control"].copy()
                        b, l = original_inventory(extractor, x2-x1, y2-y1)
                        views.append(map_tile_predictions(local, window) if view else local.copy())
                        locals_.append(local); boxes.append(b); logits.append(l)
                        native_calls += 1
                    replay, _ = extractor.predict(image.copy())
                    rb, rl = original_inventory(extractor, width, height)
                    np.testing.assert_array_equal(replay["control"], locals_[0])
                    np.testing.assert_array_equal(rb, boxes[0])
                    np.testing.assert_array_equal(rl, logits[0])
                    native_calls += 1
                    reference, origins, _, geometry = calibrated_candidates(views, windows, [width, height], [], calibrator)
                    areas = np.maximum(reference[:, 2:4]-reference[:, :2], 0).prod(1)*(640/max(width, height))**2
                    candidates = np.flatnonzero((reference[:, 4] >= .25) & (areas < 256))
                    queries = sorted(candidates, key=lambda j: digest(f"candidate56-query-v1:{stem}:{j}"))[:12]
                    selected, features, own, sham, probes = [], [], [], [], []
                    for j in queries:
                        view, cls = int(origins[j]), int(reference[j, 5])
                        x1, y1, x2, y2 = windows[view]
                        crop = image[y1:y2, x1:x2].copy()
                        local_box = reference[j, :4]-[x1, y1, x1, y1]
                        rectangles = patch_rectangles(local_box, x2-x1, y2-y1)
                        overlap = box_iou_matrix(local_box[None], boxes[view])[0]
                        eligible = overlap >= .5
                        if rectangles is None or not eligible.any():
                            probes.append(dict(reference_index=int(j), available=False))
                            continue
                        anchor = int(np.argmax(np.where(eligible, overlap*expit(logits[view][:, cls].astype(float)), -1)))
                        original = logits[view][anchor].astype(float)
                        row = dict(reference_index=int(j), available=True, view=view, anchor=anchor,
                                   rectangles=[r.tolist() for r in rectangles], perturbations=[])
                        texture, deltas = [], []
                        for rectangle in rectangles:
                            altered, descriptor, fill = erase(crop, rectangle)
                            extractor.predict(altered)
                            z = extractor.raw["scores"][0, :, anchor].float().cpu().numpy().astype(float)
                            deltas.append(original-z)
                            texture.extend(descriptor)
                            row["perturbations"].append(dict(fill=fill.tolist(),
                                image_sha256=hashlib.sha256(altered.tobytes()).hexdigest(), logits=z.tolist()))
                            perturbed_calls += 1
                        base = np.r_[geometry[j], np.eye(10)[cls], np.eye(5)[view], original, overlap[anchor], texture]
                        selected.append(j); features.append(base); own.append(deltas[0]); sham.append(deltas[1])
                        probes.append(row)
                    n = len(selected)
                    selected = np.array(selected, int)
                    tp, duplicate, ignored = labels_for(reference, metadata, .25)
                    f = dict(stem=stem, reference=reference, origins=origins, selected=selected, queries=np.array(queries, int),
                             base=np.array(features, float).reshape(n, geometry.shape[1]+38),
                             own=np.array(own, float).reshape(n, 10), sham=np.array(sham, float).reshape(n, 10),
                             labels=tp, duplicate=duplicate, ignored=ignored,
                             eligible=~(duplicate | ignored)[selected])
                    frames[split].append(f)
                    path = root/"frames"/(stem+".npz")
                    np.savez_compressed(path, **{k: v for k, v in f.items() if k != "stem"},
                                        native_boxes=np.array(boxes), native_logits=np.array(logits),
                                        **{"local_"+str(i): v for i, v in enumerate(locals_)})
                    manifest[stem] = dict(split=split, windows=windows, file=path.relative_to(root).as_posix(),
                                          sha256=file_hash(path), probes=probes)
                    if len(manifest) % 16 == 0:
                        print(f"Paired texture deletion: {len(manifest)}/192; perturbed forwards={perturbed_calls}", flush=True)
    finally:
        extractor.close()
    normal = build_inputs(frames)
    fit_frames = frames["fit"]
    y = np.concatenate([f["labels"][f["selected"]][f["eligible"]] for f in fit_frames])
    p = np.concatenate([f["reference"][f["selected"], 4][f["eligible"]] for f in fit_frames])
    offset = np.log(np.clip(p, 1e-6, 1-1e-6)/(1-np.clip(p, 1e-6, 1-1e-6)))
    coefficients, training = {}, {}
    for arm in ARMS:
        x = np.concatenate([f["inputs"][arm][f["eligible"]] for f in fit_frames])
        coefficients[arm], training[arm] = fit(x, offset, y)
        print(f"Texture-deletion readout {arm}: {training[arm]}", flush=True)
    metrics, traces = {}, {}
    for split in ("screen", "confirmation"):
        metrics[split], traces[split] = summarize(frames[split], coefficients)
    (root/"inputs").mkdir()
    for fs in frames.values():
        for f in fs:
            path = root/"inputs"/(f["stem"]+".npz")
            np.savez_compressed(path, donors=f["donors"], **f["inputs"],
                                **{a+"_logits": z for a, z in f.get("output_logits", {}).items()})
            manifest[f["stem"]].update(inputs_file=path.relative_to(root).as_posix(), inputs_sha256=file_hash(path))
    np.savez_compressed(root/"coefficients.npz", **coefficients)
    decision = decide(metrics, registration["policy_per_half"], all(t["success"] for t in training.values()))
    verification = dict(JPEG_decodes=192, original_and_identity_forwards=native_calls,
                        perturbed_forwards=perturbed_calls, existing_predictor_warmup=1, exact_global_identity_replays=192,
                        detector_updates=0, linear_models_fit=4, AP_evaluations=0,
                        sequence_prefixes_disjoint_across_all192_images=True,
                        fresh_exact_images_against_recorded_prior_plans=True,
                        source_unchanged=all(file_hash(project_path(p)) == h for p, h in sources.items()),
                        installed_sources_unchanged=all(file_hash(p) == plan["installed_sources"][n]["sha256"] for n, p in installed.items()),
                        elapsed_seconds=perf_counter()-tick)
    for name, value in (("frames.json", manifest), ("normalization.json", normal), ("training.json", training),
                        ("per_image.json", traces), ("metrics.json", metrics), ("decision.json", decision),
                        ("verification.json", verification)):
        save_json(root/name, value)
    result = dict(experiment=registration["experiment"], date="2026-10-15", output=str(root),
                  metrics=metrics, decision=decision, training=training, verification=verification,
                  artifacts_sha256={p.name: file_hash(p) for p in sorted(root.iterdir()) if p.is_file()})
    save_json(root/"result.json", result)
    return result
