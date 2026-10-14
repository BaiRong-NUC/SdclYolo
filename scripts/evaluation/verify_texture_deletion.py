"""Independent cached geometry, pixel, fit and decision audit for candidate56."""

import hashlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile

import cv2
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.localization import calibrated_candidates, frozen_control, load_calibrator


ROOT = "output/analysis/texture-deletion-fixed-20261015"
ARMS = ("base", "sham", "own", "shuffled")


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, (str, bool)) or expected is None:
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)


def overlaps(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    intersection = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:])-
                              np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(2)
    aa = np.maximum(a[:, 2:]-a[:, :2], 0).prod(1)
    bb = np.maximum(b[:, 2:]-b[:, :2], 0).prod(1)
    return intersection/np.maximum(aa[:, None]+bb[None]-intersection, 1e-12)


def supervision(reference, metadata):
    gt = np.array([o["box_xyxy"] for o in metadata["valid_objects"]], float).reshape(-1, 4)
    cls = np.array([o["class_id"] for o in metadata["valid_objects"]])
    area = np.maximum(reference[:, 2:4]-reference[:, :2], 0).prod(1)
    ignored = np.zeros(len(reference), bool)
    for box in metadata["ignore_boxes_xyxy"]:
        intersection = np.maximum(np.minimum(reference[:, 2:4], box[2:])-
                                  np.maximum(reference[:, :2], box[:2]), 0).prod(1)
        ignored |= intersection/np.maximum(area, 1e-12) > .5
    iou, tp, used = overlaps(gt, reference[:, :4]), np.zeros(len(reference), bool), set()
    for row in sorted(range(len(reference)), key=lambda j: (-reference[j, 4], j)):
        if reference[row, 4] < .25 or ignored[row]:
            continue
        eligible = [g for g in range(len(gt)) if g not in used and cls[g] == reference[row, 5] and iou[g, row] >= .5]
        if eligible:
            used.add(max(eligible, key=lambda g: (iou[g, row], -g)))
            tp[row] = True
    duplicate = ~tp & ((iou >= .5) & (cls[:, None] == reference[None, :, 5])).any(0)
    return tp, duplicate, ignored


def rectangles(box, width, height):
    x1 = max(0, min(width, math.floor(box[0])))
    y1 = max(0, min(height, math.floor(box[1])))
    x2 = max(0, min(width, math.ceil(box[2])))
    y2 = max(0, min(height, math.ceil(box[3])))
    w, h = x2-x1, y2-y1
    if w <= 0 or h <= 0:
        return None
    for dx, dy in ((2*w, 0), (-2*w, 0), (0, 2*h), (0, -2*h)):
        if x1+dx >= 0 and y1+dy >= 0 and x2+dx <= width and y2+dy <= height:
            return [[x1, y1, x2, y2], [x1+dx, y1+dy, x2+dx, y2+dy]]
    return None


def verify():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    root = project_path(ROOT)
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    result, plan, manifest = read(root/"result.json"), read(root/"plan.json"), read(root/"frames.json")
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(root/name) == digest
    with zipfile.ZipFile(root/"source.zip") as z:
        assert file_hash(root/"source.zip") == plan["source_archive_sha256"]
        for name, digest in plan["source_hashes"].items():
            assert file_hash(project_path(name)) == hashlib.sha256(z.read(name)).hexdigest() == digest
        for name, item in plan["installed_sources"].items():
            assert file_hash(Path(item["path"])) == hashlib.sha256(z.read("installed/"+name)).hexdigest() == item["sha256"]
    excluded = set()
    for name, digest in plan["exclusions"].items():
        p = project_path(name)
        assert file_hash(p) == digest
        excluded.update(re.findall(r"\d{7}_\d{5}_[a-z]_\d{7}", p.read_text(encoding="utf-8")))
    assert len(excluded) == plan["excluded_unique_images"]
    groups = {}
    for p in project_path("data/visdrone/images/train").glob("*.jpg"):
        if p.stem not in excluded:
            groups.setdefault(p.stem[:7], []).append(p.stem)
    scenes = sorted(groups, key=lambda s: sha("candidate56-scene-v1:"+s))[:192]
    images = [min(groups[s], key=lambda stem: sha("candidate56-image-v1:"+stem)) for s in scenes]
    selection = {s: images[i*64:(i+1)*64] for i, s in enumerate(("fit", "screen", "confirmation"))}
    assert selection == plan["image_selection"] and len(set(stem[:7] for stem in images)) == 192
    control = frozen_control()
    assert control["detector"]["sha256"] == plan["checkpoint_sha256"]
    assert file_hash(project_path("output/analysis/calibration-control-supplement-20261013/control_manifest.json")) == plan["control_manifest_sha256"]
    calibrator = load_calibrator(control)
    frames, pixel_variants, query_count, unavailable = {s: [] for s in selection}, 0, 0, 0
    for split, stems in selection.items():
        for stem in stems:
            entry = manifest[stem]
            assert file_hash(root/entry["file"]) == entry["sha256"]
            assert file_hash(root/entry["inputs_file"]) == entry["inputs_sha256"]
            image_path = project_path(f"data/visdrone/images/train/{stem}.jpg")
            meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            assert file_hash(image_path) == plan["inputs"][stem]["image_sha256"]
            assert file_hash(meta_path) == plan["inputs"][stem]["metadata_sha256"]
            image, metadata = cv2.imread(str(image_path)), read(meta_path)
            width, height = metadata["original_size"]
            assert image.shape == (height, width, 3)
            tw, th = min(width, math.ceil(width/1.8)), min(height, math.ceil(height/1.8))
            windows = [[0, 0, width, height], *[[x, y, x+tw, y+th] for y in (0, height-th) for x in (0, width-tw)]]
            assert windows == entry["windows"]
            with np.load(root/entry["file"], allow_pickle=False) as z:
                f = {k: z[k].copy() for k in z.files}
            views = []
            for view, (x1, y1, x2, y2) in enumerate(windows):
                v = f["local_"+str(view)].copy()
                if view:
                    v[:, [0, 2]] = v[:, [0, 2]].clip(0, x2-x1)+x1
                    v[:, [1, 3]] = v[:, [1, 3]].clip(0, y2-y1)+y1
                views.append(v)
            reference, origins, _, geometry = calibrated_candidates(views, windows, [width, height], [], calibrator)
            np.testing.assert_array_equal(reference, f["reference"])
            np.testing.assert_array_equal(origins, f["origins"])
            areas = np.maximum(reference[:, 2:4]-reference[:, :2], 0).prod(1)*(640/max(width, height))**2
            candidates = [j for j in range(len(reference)) if reference[j, 4] >= .25 and areas[j] < 256]
            queries = sorted(candidates, key=lambda j: sha(f"candidate56-query-v1:{stem}:{j}"))[:12]
            np.testing.assert_array_equal(queries, f["queries"])
            assert len(queries) == len(entry["probes"])
            selected, features, own, sham = [], [], [], []
            for j, probe in zip(queries, entry["probes"], strict=True):
                view, cls = int(origins[j]), int(reference[j, 5])
                x1, y1, x2, y2 = windows[view]
                local_box = reference[j, :4]-[x1, y1, x1, y1]
                rects = rectangles(local_box, x2-x1, y2-y1)
                iou = overlaps(local_box[None], f["native_boxes"][view])[0]
                valid = iou >= .5
                available = rects is not None and bool(valid.any())
                assert probe["reference_index"] == j and probe["available"] == available
                query_count += 1
                if not available:
                    unavailable += 1
                    continue
                values = f["native_logits"][view].astype(float)
                sigmoid = 1/(1+np.exp(-values[:, cls]))
                anchor = int(np.argmax(np.where(valid, iou*sigmoid, -1)))
                assert probe["anchor"] == anchor and probe["view"] == view and probe["rectangles"] == rects
                original = values[anchor]
                crop = image[y1:y2, x1:x2]
                texture, delta = [], []
                for rect, perturbation in zip(rects, probe["perturbations"], strict=True):
                    lx, ly, hx, hy = rect
                    patch = crop[ly:hy, lx:hx].reshape(-1, 3).astype(float)
                    mean, std = patch.mean(0), patch.std(0)
                    fill = np.rint(mean).astype(np.uint8)
                    altered = crop.copy()
                    altered[ly:hy, lx:hx] = fill
                    assert hashlib.sha256(altered.tobytes()).hexdigest() == perturbation["image_sha256"]
                    np.testing.assert_array_equal(fill, perturbation["fill"])
                    texture.extend(np.r_[mean, std])
                    delta.append(original-np.array(perturbation["logits"]))
                    pixel_variants += 1
                selected.append(j)
                features.append(np.r_[geometry[j], np.eye(10)[cls], np.eye(5)[view], original, iou[anchor], texture])
                own.append(delta[0]); sham.append(delta[1])
            np.testing.assert_array_equal(selected, f["selected"])
            equal(np.array(features).reshape(f["base"].shape), f["base"])
            equal(np.array(own).reshape(-1, 10), f["own"])
            equal(np.array(sham).reshape(-1, 10), f["sham"])
            tp, duplicate, ignored = supervision(reference, metadata)
            for name, rebuilt in (("labels", tp), ("duplicate", duplicate), ("ignored", ignored),
                                  ("eligible", ~(duplicate | ignored)[f["selected"]])):
                np.testing.assert_array_equal(rebuilt, f[name])
            f.update(stem=stem)
            for name in [n for n in f if n.startswith("native_") or n.startswith("local_")]:
                del f[name]
            frames[split].append(f)
    assert pixel_variants == result["verification"]["perturbed_forwards"]
    normal = read(root/"normalization.json")
    for key in ("base", "sham", "own"):
        values = np.concatenate([f[key][f["eligible"]] for f in frames["fit"]])
        equal(dict(mean=values.mean(0).tolist(), std=np.maximum(values.std(0), 1e-6).tolist()), normal[key])
    for split, fs in frames.items():
        order = {cls: [] for cls in range(10)}
        for i, f in enumerate(fs):
            for j, row in enumerate(f["selected"]):
                order[int(f["reference"][row, 5])].append((i, j))
        for cls, rows in order.items():
            rows.sort(key=lambda ij: sha(f"candidate56-donor-v1:{fs[ij[0]]['stem']}:{fs[ij[0]]['selected'][ij[1]]}"))
        for i, f in enumerate(fs):
            standardized = {k: (f[k]-normal[k]["mean"])/normal[k]["std"] for k in ("base", "sham", "own")}
            own, shuffled = standardized["own"].copy(), np.zeros_like(standardized["own"])
            donor_indices = np.full((len(own), 2), -1, int)
            for j, reference_index in enumerate(f["selected"]):
                cls = int(f["reference"][reference_index, 5])
                rows = order[cls]
                k = rows.index((i, j))
                donor = next((rows[(k+d) % len(rows)] for d in range(1, len(rows)) if rows[(k+d) % len(rows)][0] != i), None)
                if donor is None:
                    own[j] = 0
                else:
                    di, dj = donor
                    donor_indices[j] = donor
                    shuffled[j] = (fs[di]["own"][dj]-normal["own"]["mean"])/normal["own"]["std"]
            zero = np.zeros_like(own)
            f["inputs"] = dict(base=np.column_stack((standardized["base"], zero, zero)),
                               sham=np.column_stack((standardized["base"], standardized["sham"], zero)),
                               own=np.column_stack((standardized["base"], standardized["sham"], own)),
                               shuffled=np.column_stack((standardized["base"], standardized["sham"], shuffled)))
            with np.load(root/manifest[f["stem"]]["inputs_file"], allow_pickle=False) as z:
                np.testing.assert_array_equal(donor_indices, z["donors"])
                for arm in ARMS:
                    equal(f["inputs"][arm], z[arm])
    with np.load(root/"coefficients.npz", allow_pickle=False) as z:
        coefficients = {a: z[a].copy() for a in ARMS}
    fit_y = np.concatenate([f["labels"][f["selected"]][f["eligible"]] for f in frames["fit"]])
    fit_p = np.concatenate([f["reference"][f["selected"], 4][f["eligible"]] for f in frames["fit"]])
    fit_p = np.clip(fit_p, 1e-6, 1-1e-6)
    offset = np.log(fit_p/(1-fit_p))
    gradients = {}
    for arm, theta in coefficients.items():
        x = np.concatenate([f["inputs"][arm][f["eligible"]] for f in frames["fit"]])
        z = offset+x@theta[:-1]+theta[-1]
        residual = (expit(z)-fit_y)/len(fit_y)
        gradient = np.r_[x.T@residual+.01*theta[:-1], residual.sum()]
        objective = float(np.mean(np.logaddexp(0, z)-fit_y*z)+.005*np.dot(theta[:-1], theta[:-1]))
        equal(objective, result["training"][arm]["objective"])
        equal(float(abs(gradient).max()), result["training"][arm]["gradient_inf"])
        gradients[arm] = float(abs(gradient).max())
    traces, metrics = {}, {}
    for split in ("screen", "confirmation"):
        traces[split], labels, scores = [], [], {a: [] for a in ARMS}
        for f in frames[split]:
            p = np.clip(f["reference"][f["selected"], 4], 1e-6, 1-1e-6)
            offset = np.log(p/(1-p))
            keep, y = f["eligible"], f["labels"][f["selected"]]
            outputs = {a: offset+f["inputs"][a]@t[:-1]+t[-1] for a, t in coefficients.items()}
            with np.load(root/manifest[f["stem"]]["inputs_file"], allow_pickle=False) as z:
                for a in ARMS:
                    equal(outputs[a], z[a+"_logits"])
            labels.extend(y[keep])
            for a in ARMS:
                scores[a].extend(outputs[a][keep])
            traces[split].append(dict(rows=int(keep.sum()), tp=int(y[keep].sum()), fp=int((~y[keep]).sum()),
                images=int(keep.any()), loss_sums={a: math.fsum(np.logaddexp(0, v[keep])-y[keep]*v[keep]) for a, v in outputs.items()}))
        total = {k: sum(t[k] for t in traces[split]) for k in ("rows", "tp", "fp", "images")}
        total["NLL"] = {a: math.fsum(t["loss_sums"][a] for t in traces[split])/total["rows"] if total["rows"] else None for a in ARMS}
        y = np.array(labels, bool)
        total["AUROC"] = {a: float((rankdata(v)[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum())) if total["tp"] and total["fp"] else None for a, v in scores.items()}
        draws = np.random.default_rng(0).multinomial(64, [1/64]*64, size=400)
        gains = []
        for draw in draws:
            mass = math.fsum(w*t["rows"] for w, t in zip(draw, traces[split]))
            if not mass:
                gains = None
                break
            gains.append(math.fsum(w*(t["loss_sums"]["sham"]-t["loss_sums"]["own"]) for w, t in zip(draw, traces[split]))/mass)
        total["sham_minus_own_NLL_bootstrap95ci"] = np.percentile(gains, [2.5, 97.5]).tolist() if gains is not None else None
        metrics[split] = total
    for split, items in traces.items():
        for actual, expected in zip(items, read(root/"per_image.json")[split], strict=True):
            equal(actual, expected)
    equal(metrics, result["metrics"])
    policy, checks = plan["registration"]["policy_per_half"], {}
    for split, m in metrics.items():
        populated = bool(m["tp"] and m["fp"])
        ci = m["sham_minus_own_NLL_bootstrap95ci"]
        checks[split] = dict(tp=m["tp"] >= policy["tp_min"], fp=m["fp"] >= policy["nonduplicate_fp_min"],
            images=m["images"] >= policy["images_min"],
            vs_base=populated and m["NLL"]["base"]-m["NLL"]["own"] >= policy["NLL_gain_vs_base_min"],
            vs_sham=populated and m["NLL"]["sham"]-m["NLL"]["own"] >= policy["NLL_gain_vs_sham_min"],
            vs_shuffled=populated and m["NLL"]["shuffled"]-m["NLL"]["own"] >= policy["NLL_gain_vs_shuffled_min"],
            AUROC=populated and m["AUROC"]["own"]-m["AUROC"]["sham"] >= policy["AUROC_gain_vs_sham_min"],
            interval=ci is not None and ci[0] > policy["NLL_gain_vs_sham_bootstrap_lower_min_exclusive"])
    numerical = all(result["training"][a]["solver_success"] and gradients[a] <= 1e-5 for a in ARMS)
    assert checks == result["decision"]["checks"] and numerical == result["decision"]["numerical"]
    passed = numerical and all(all(c.values()) for c in checks.values())
    expected = "REPLICATED_TEXTURE_DELETION_INFORMATION_SUPPORTS_SEPARATE_DEPLOYMENT_AND_COST_PILOT" if passed else "STOP_THIS_PAIRED_MEAN_FILL_TEXTURE_DELETION_RECIPE"
    assert result["decision"]["decision"] == expected
    audit = dict(all_sources_archive_installed_code_frames_inputs192image_metadata_hashes_verified=True,
        all_exclusions_fresh_selection_disjoint_sequence_prefixes_and_windows_rebuilt=True,
        full_reference_and_features_rebuilt_with_shared_frozen_calibration=True,
        query_candidates_independent_IoU_association_and_patch_availability_rebuilt=query_count,
        unavailable_queries=unavailable, pixel_variants_recreated_and_hashed=pixel_variants,
        all_texture_deltas_labels_duplicates_ignores_inputs_normalization_and_donors_rebuilt=True,
        all_four_saved_fit_objectives_stationarity_and_both_halves_NLL_rank_AUROC_bootstrap_gates_rebuilt=True,
        four_fit_gradient_inf=gradients, additional_JPEG_decodes=192, additional_detector_forwards=0,
        verifier_sha256=file_hash(Path(__file__)),
        limits="Native original tensors and perturbed queried logits are hashed/stored, not independently re-forwarded. Full calibration shares frozen implementation. Diagnostic scenes are new exact images but detector/calibrator training data; prior diagnostics may contain other frames from these sequences. Mean fill deletes texture, not a plausible absent object. Sham can affect other objects. No detection AP, deployable gain, cost amortization or novelty proof.")
    destination = root/"independent_verification.json"
    assert not destination.exists()
    destination.write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    verify()
