"""Fixed, train-only instance moment update folded into one P3 class conv."""

import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter
import zipfile

import cv2
import numpy as np
import torch

from sdcl.config import project_path
from sdcl.evaluation.official_details import NAMES
from sdcl.evaluation.size_analysis import coco_size_metrics, file_hash, recall_at_fp_budgets
from sdcl.evaluation.slicing_screen import map_tile_predictions
from sdcl.postprocess.localization import calibrated_candidates, frozen_control, load_calibrator
from sdcl.postprocess.rescue import save_json
from sdcl.training.class_margin_probe import FrozenViews
from sdcl.training.crossview_quality_rank import precision, state_hash
from sdcl.training.transport_quality import diagnostic_filter, load_cache


REGISTRATION = "docs/experiments/candidate69_instance_moments_registration_20261015.json"
OUTPUT = "output/analysis/instance-moments-fixed-20261015"
PRIOR = "output/analysis/transport-quality-fixed-20261015"
ARMS = ("global", "tiny_pixel", "shuffle", "instance")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def coverage(box, side=120, stride=8):
    edge = np.arange(side, dtype=float)*stride
    x = np.maximum(np.minimum(edge+stride, box[2])-np.maximum(edge, box[0]), 0)/stride
    y = np.maximum(np.minimum(edge+stride, box[3])-np.maximum(edge, box[1]), 0)/stride
    return y[:, None]*x[None, :]


def weight_maps(metadata, window, stem, view):
    x1, y1, x2, y2 = window
    scale = 960/max(x2-x1, y2-y1)
    pad = np.array([round((960-round((x2-x1)*scale))/2-.1),
                    round((960-round((y2-y1)*scale))/2-.1)])
    origin = np.array([x1, y1, x1, y1])
    def transform(box):
        return (np.asarray(box)-origin)*scale+np.tile(pad, 2)
    global_map = coverage(transform(window))
    for box in metadata["ignore_boxes_xyxy"]:
        global_map[coverage(transform(box)) > 0] = 0
    pixel = np.zeros_like(global_map)
    instance = pixel.copy()
    count = 0
    for obj in metadata["valid_objects"]:
        box = np.asarray(obj["box_xyxy"], float)
        area = np.prod(np.maximum(box[2:]-box[:2], 0))
        if area*(640/max(metadata["original_size"]))**2 >= 256 or area <= 0:
            continue
        clipped = np.r_[np.maximum(box[:2], [x1, y1]), np.minimum(box[2:], [x2, y2])]
        if np.prod(np.maximum(clipped[2:]-clipped[:2], 0))/area < .9:
            continue
        weights = coverage(transform(clipped))*global_map
        if weights.sum() > 0:
            pixel += weights
            instance += weights/weights.sum()
            count += 1
    valid = np.flatnonzero(global_map.ravel() > 0)
    seed = int.from_bytes(hashlib.sha256(f"candidate69-shuffle-v1:{stem}:{view}".encode()).digest()[:8], "little")
    shuffled = np.zeros_like(instance)
    shuffled.ravel()[valid] = np.random.default_rng(seed).permutation(instance.ravel()[valid])
    return dict(global_=global_map, tiny_pixel=pixel, shuffle=shuffled,
                instance=instance), count


def summaries(z, maps):
    flat = np.asarray(z, np.float64).reshape(128, -1)
    weights = np.stack([maps["global_" if a == "global" else a].ravel() for a in ARMS])
    return dict(mass=weights.sum(1), sums=weights@flat.T, squares=weights@(flat*flat).T)


def coefficients(mean, variance, gamma, beta, old_var, eps, rho=.03):
    a, b = np.ones_like(gamma, dtype=float), np.zeros_like(gamma, dtype=float)
    active = np.abs(gamma) >= 1e-6
    denominator = (1-rho)+rho*(variance[active]/gamma[active]**2+eps/(old_var[active]+eps))
    a[active] = 1/np.sqrt(denominator)
    b[active] = (1-a[active])*beta[active]-a[active]*rho*(mean[active]-beta[active])
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a <= 0):
        raise ValueError("Invalid fixed moment update; no clipping or retuning.")
    return a, b


def folded(weight, bias, a, b):
    return ((np.asarray(weight, float)*a[:, None, None, None]).astype(np.float16),
            (np.asarray(bias, float)*a+b).astype(np.float16))


def metrics(root, selection, predictions, manifest):
    reports = {}
    for split in ("screen", "confirmation"):
        records, maps = [], {a: {} for a in ("strong", *ARMS)}
        for index, stem in enumerate(selection[split]):
            metadata = read(project_path(f"data/visdrone/metadata/train/{stem}.json"))
            records.append(dict(id=index+1, stem=stem, size=manifest[stem]["size"],
                                objects=metadata["valid_objects"]))
            with np.load(root/predictions[stem]["file"], allow_pickle=False) as z:
                for arm in maps:
                    maps[arm][stem] = diagnostic_filter(z[arm], metadata)
        ap = {a: coco_size_metrics(records, m, dict(enumerate(NAMES)), reference_size=640)
              for a, m in maps.items()}
        fp = {a: recall_at_fp_budgets(records, dict(baseline=maps["strong"], sdcl=maps[a]),
                                    budgets=(10,), bootstrap_samples=0)[0] for a in ARMS}
        reports[split] = dict(AP=ap, fixed_fp=fp)
    return reports


def decide(reports, policy):
    checks, differences = {}, {}
    for split, report in reports.items():
        ap = report["AP"]
        c = dict(tiny_GT=ap["strong"]["tiny"]["gt"] >= policy["tiny_GT_min"])
        d = {}
        for group, control in [("tiny", a) for a in ("strong", "global", "tiny_pixel", "shuffle")] + [
                ("small_all", "strong"), ("all", "strong")]:
            key = f"{group}_AP_vs_{control}"
            x, y = ap["instance"][group]["ap50_95"], ap[control][group]["ap50_95"]
            d[key] = None if x is None or y is None else (x-y)*100
            c[key] = d[key] is not None and d[key] >= policy[key+"_min_pp"]
        d["recall"] = report["fixed_fp"]["instance"][0]["delta_recall_pp"]["small_all"]
        c["fixed_FP"] = d["recall"] is not None and d["recall"] >= policy[
            "small_all_recall_FPPI10_vs_strong_min_pp"]
        checks[split], differences[split] = c, d
    return dict(passed=all(all(c.values()) for c in checks.values()),
                checks=checks, differences=differences)


def run(device="0"):
    from ultralytics import YOLO
    from ultralytics.nn.modules import head, conv as conv_module
    from ultralytics.utils import ops, nms
    from ultralytics.data import augment
    started = perf_counter()
    precision(False)
    root, prior = project_path(OUTPUT), project_path(PRIOR)
    if root.exists():
        raise FileExistsError(root)
    registration = read(project_path(REGISTRATION))
    previous = read(project_path("docs/experiments/candidate62_transport_quality_result_20261015.json"))
    for name in ("plan.json", "frames.json"):
        assert file_hash(prior/name) == previous["artifacts_sha256"][name]
    selection = read(prior/"plan.json")["image_selection"]
    manifest = read(prior/"frames.json")
    control = frozen_control()
    calibrator = load_calibrator(control)
    sources = [REGISTRATION, registration["literature"], "sdcl/evaluation/instance_moments.py",
               "scripts/evaluation/diagnose_instance_moments.py",
               "scripts/evaluation/verify_instance_moments.py", "tests/test_instance_moments.py",
               "sdcl/config.py", "sdcl/evaluation/size_analysis.py",
               "sdcl/evaluation/official_details.py", "sdcl/evaluation/slicing_screen.py",
               "sdcl/training/class_margin_probe.py", "sdcl/training/crossview_quality_rank.py",
               "sdcl/training/transport_quality.py", "sdcl/postprocess/localization.py",
               "sdcl/postprocess/class_margin_probe.py", "sdcl/postprocess/rescue.py",
               "sdcl/data/geometry.py"]
    hashes = {p: file_hash(project_path(p)) for p in sources}
    installed = {inspect.getfile(m): file_hash(inspect.getfile(m)) for m in (head, conv_module, ops, nms, augment)}
    root.mkdir(parents=True)
    (root/"fit").mkdir()
    (root/"predictions").mkdir()
    with zipfile.ZipFile(root/"source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(project_path(name), name)
    save_json(root/"plan.json", dict(registration=registration, selection=selection,
        prior_plan_sha256=file_hash(prior/"plan.json"), prior_frames_sha256=file_hash(prior/"frames.json"),
        source_hashes=hashes, installed_source_hashes=installed,
        source_archive_sha256=file_hash(root/"source.zip"), detector=control["detector"],
        scorer=control["scorer"], base_commit="3c4b1ff", device=device))
    unfused = YOLO(control["detector"]["path"]).model.eval()
    bn = unfused.model[-1].cv3[0][1][1].bn
    bn_values = dict(gamma=bn.weight.detach().numpy().astype(float),
                     beta=bn.bias.detach().numpy().astype(float),
                     old_var=bn.running_var.numpy().astype(float), eps=np.array(bn.eps))
    np.savez(root/"bn.npz", **bn_values)
    del unfused, bn
    extractor = FrozenViews(device)
    conv = extractor.head.cv3[0][1][1].conv
    original_weight, original_bias = conv.weight.detach().clone(), conv.bias.detach().clone()
    initial_hash = state_hash(extractor.model.predictor.model.model)
    z_capture = {}
    def hook(module, inputs, output):
        z_capture["z"] = output.detach().clone()
    # Ultralytics shares its activation singleton; the conv is unique to P3.
    handle = conv.register_forward_hook(hook)
    costs = dict(JPEG_decodes=0, detector_forwards=0, explicit_initialization_predictions=1,
                 closed_form_moment_fits=0, optimizer_steps=0, full_training_steps=0)
    fit_manifest, predictions, eval_cache = {}, {}, {}
    total = dict(mass=np.zeros(4), sums=np.zeros((4, 128)), squares=np.zeros((4, 128)))
    def install(w, b):
        conv.weight.copy_(torch.as_tensor(w, device=conv.weight.device, dtype=conv.weight.dtype))
        conv.bias.copy_(torch.as_tensor(b, device=conv.bias.device, dtype=conv.bias.dtype))
    def predict(image):
        costs["detector_forwards"] += 1
        return extractor.predict(image)[0]["control"]
    def pristine(image, frame):
        local = predict(image)
        np.testing.assert_array_equal(extractor.decoded[0].cpu().numpy(), frame["decoded"])
        np.testing.assert_array_equal(extractor.raw["scores"][0].T.cpu().numpy(), frame["raw_class"])
        np.testing.assert_array_equal(extractor.raw["boxes"][0].T.reshape(-1, 4, 16).cpu().numpy(), frame["raw_dfl"])
        np.testing.assert_array_equal(local, frame["original_local"])
        return local
    try:
        with torch.inference_mode():
            for i, stem in enumerate(selection["fit"]):
                entry = manifest[stem]
                image_path = project_path(f"data/visdrone/images/train/{stem}.jpg")
                meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
                assert file_hash(image_path) == entry["image_sha256"]
                assert file_hash(meta_path) == entry["metadata_sha256"]
                image, metadata = cv2.imread(str(image_path)), read(meta_path)
                costs["JPEG_decodes"] += 1
                assert image is not None and list(image.shape[1::-1]) == entry["size"]
                for view, window in enumerate(entry["windows"]):
                    frame = load_cache(prior, entry["views"][view])
                    x1, y1, x2, y2 = window
                    pristine(image[y1:y2, x1:x2].copy(), frame)
                    z = z_capture["z"][0].cpu().numpy()
                    assert z.shape == (128, 120, 120)
                    maps, count = weight_maps(metadata, window, stem, view)
                    summary = summaries(z, maps)
                    for key in total:
                        total[key] += summary[key]
                    path = root/"fit"/f"{stem}-v{view}.npz"
                    np.savez(path, z=z, **maps, **summary, tiny_count=np.array(count))
                    fit_manifest[f"{stem}:{view}"] = dict(file=path.relative_to(root).as_posix(),
                        sha256=file_hash(path), stem=stem, view=view)
                if i == 0 or (i+1) % 8 == 0:
                    print(f"Instance moment fit: {i+1}/64", flush=True)
            if np.any(total["mass"] <= 0):
                raise ValueError("No moment mass; preserve fixed data and fail engineering.")
            mean = total["sums"]/total["mass"][:, None]
            variance = np.maximum(total["squares"]/total["mass"][:, None]-mean**2, 0)
            np.savez(root/"moments.npz", **total, mean=mean, variance=variance)
            weights = {}
            for index, arm in enumerate(ARMS):
                a, b = coefficients(mean[index], variance[index], **bn_values)
                costs["closed_form_moment_fits"] += 1
                w, bias = folded(original_weight.cpu().numpy(), original_bias.cpu().numpy(), a, b)
                weights[arm] = (w, bias)
                np.savez(root/f"folded-{arm}.npz", weight=w, bias=bias, a=a, b=b)
            np.savez(root/"folded-strong.npz", weight=original_weight.cpu().numpy(),
                     bias=original_bias.cpu().numpy())
            for split in ("screen", "confirmation"):
                for i, stem in enumerate(selection[split]):
                    entry = manifest[stem]
                    image_path = project_path(f"data/visdrone/images/train/{stem}.jpg")
                    meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
                    assert file_hash(image_path) == entry["image_sha256"]
                    assert file_hash(meta_path) == entry["metadata_sha256"]
                    image = cv2.imread(str(image_path))
                    costs["JPEG_decodes"] += 1
                    assert image is not None and list(image.shape[1::-1]) == entry["size"]
                    views = {a: [] for a in ("strong", *ARMS)}
                    for view, window in enumerate(entry["windows"]):
                        frame = load_cache(prior, entry["views"][view])
                        x1, y1, x2, y2 = window
                        crop = image[y1:y2, x1:x2].copy()
                        install(original_weight, original_bias)
                        local = pristine(crop, frame)
                        views["strong"].append(map_tile_predictions(local, window) if view else local)
                        cache = dict(strong_local=local, strong_decoded=extractor.decoded[0].cpu().numpy())
                        for arm in ARMS:
                            install(*weights[arm])
                            local = predict(crop)
                            np.testing.assert_array_equal(extractor.decoded[0, :4].cpu().numpy(), frame["decoded"][:4])
                            views[arm].append(map_tile_predictions(local, window) if view else local)
                            cache[arm+"_local"] = local
                            cache[arm+"_decoded"] = extractor.decoded[0].cpu().numpy()
                            cache[arm+"_raw_class"] = extractor.raw["scores"][0].cpu().numpy()
                        path = root/"predictions"/f"{stem}-v{view}.npz"
                        np.savez(path, **cache)
                        eval_cache[f"{stem}:{view}"] = dict(file=path.relative_to(root).as_posix(),
                            sha256=file_hash(path), stem=stem, view=view)
                    install(original_weight, original_bias)
                    assert state_hash(extractor.model.predictor.model.model) == initial_hash
                    final = {a: calibrated_candidates(v, entry["windows"], entry["size"], [], calibrator)[0]
                             for a, v in views.items()}
                    ref = prior/entry["reference"]["file"]
                    assert file_hash(ref) == entry["reference"]["sha256"]
                    np.testing.assert_array_equal(final["strong"], np.load(ref, allow_pickle=False))
                    path = root/"predictions"/f"{stem}-final.npz"
                    np.savez(path, **final)
                    predictions[stem] = dict(file=path.relative_to(root).as_posix(), sha256=file_hash(path))
                    if i == 0 or (i+1) % 8 == 0:
                        print(f"Instance moment evaluation {split}: {i+1}/{len(selection[split])}", flush=True)
            reports = metrics(root, selection, predictions, manifest)
            decision = decide(reports, registration["policy_per_half"])
            save_json(root/"metrics.json", reports)
            save_json(root/"decision.json", decision)
            save_json(root/"fit_manifest.json", fit_manifest)
            save_json(root/"eval_manifest.json", eval_cache)
            save_json(root/"predictions.json", predictions)
            save_json(root/"state.json", dict(before=initial_hash,
                after=state_hash(extractor.model.predictor.model.model), weights_restored=True))
            assert all(file_hash(project_path(p)) == h for p, h in hashes.items())
            assert all(file_hash(p) == h for p, h in installed.items())
            frozen_control()
            costs["main_elapsed_seconds"] = perf_counter()-started
            result = dict(experiment=registration["experiment"], decision=registration[
                "pass" if decision["passed"] else "fail"], checks=decision, costs=costs, output=str(root),
                limits=registration["metrics"])
            save_json(root/"result.json", result)
            return result
    except Exception as error:
        save_json(root/"engineering_failure.json", dict(type=type(error).__name__, message=str(error),
            costs=costs, elapsed_seconds=perf_counter()-started))
        raise
    finally:
        with torch.inference_mode():
            install(original_weight, original_bias)
        handle.remove()
        extractor.close()
