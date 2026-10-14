"""Independent pixel/readout audit and fixed three-batch feature replay for58."""

import hashlib
import json
import math
import os
from pathlib import Path
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


ROOT = "output/analysis/external-content-fixed-20261015"
PRIOR = "output/analysis/texture-deletion-fixed-20261015"
ARMS = ("base", "own", "context", "shuffled")


def compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, list) and expected and isinstance(expected[0], dict):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            compare(a, b)
    elif isinstance(expected, (str, bool)) or expected is None:
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)


def independently_preprocessed(image, box, factor):
    center_x, center_y = (box[0]+box[2])/2, (box[1]+box[3])/2
    half_w, half_h = (box[2]-box[0])*factor/2, (box[3]-box[1])*factor/2
    rect = [math.floor(center_x-half_w), math.floor(center_y-half_h),
            math.ceil(center_x+half_w), math.ceil(center_y+half_h)]
    x1, y1, x2, y2 = rect
    crop = np.empty((y2-y1, x2-x1, 3), dtype=np.uint8)
    crop.fill(114)
    lx, hx, ly, hy = max(0, x1), min(image.shape[1], x2), max(0, y1), min(image.shape[0], y2)
    if lx < hx and ly < hy:
        crop[ly-y1:hy-y1, lx-x1:hx-x1] = image[ly:hy, lx:hx]
    ratio = 224/max(y2-y1, x2-x1)
    w, h = round((x2-x1)*ratio), round((y2-y1)*ratio)
    resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
    canvas = np.empty((224, 224, 3), np.uint8)
    canvas.fill(114)
    top, left = (224-h)//2, (224-w)//2
    canvas[top:top+h, left:left+w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)/np.float32(255)
    rgb -= np.array([.485, .456, .406], np.float32)
    rgb /= np.array([.229, .224, .225], np.float32)
    return np.moveaxis(rgb, -1, 0).copy(), rect, canvas


def independent_labels(reference, metadata):
    gt = np.array([o["box_xyxy"] for o in metadata["valid_objects"]], float).reshape(-1, 4)
    cls = np.array([o["class_id"] for o in metadata["valid_objects"]], int)
    areas = np.maximum(reference[:, 2:4]-reference[:, :2], 0).prod(1)
    ignored = np.zeros(len(reference), bool)
    for box in metadata["ignore_boxes_xyxy"]:
        intersect = np.maximum(np.minimum(reference[:, 2:4], box[2:])-
                               np.maximum(reference[:, :2], box[:2]), 0).prod(1)
        ignored |= intersect/np.maximum(areas, 1e-12) > .5
    intersection = np.maximum(np.minimum(gt[:, None, 2:], reference[None, :, 2:4])-
                              np.maximum(gt[:, None, :2], reference[None, :, :2]), 0).prod(2)
    ga = np.maximum(gt[:, 2:]-gt[:, :2], 0).prod(1)
    iou = intersection/np.maximum(ga[:, None]+areas[None]-intersection, 1e-12)
    tp, owned = np.zeros(len(reference), bool), np.zeros(len(gt), bool)
    for j in sorted(range(len(reference)), key=lambda i: (-reference[i, 4], i)):
        if reference[j, 4] < .25 or ignored[j]:
            continue
        eligible = np.flatnonzero(~owned & (cls == reference[j, 5]) & (iou[:, j] >= .5))
        if len(eligible):
            owned[eligible[np.argmax(iou[eligible, j])]] = True
            tp[j] = True
    duplicate = ~tp & ((iou >= .5) & (cls[:, None] == reference[None, :, 5])).any(0)
    return tp, ignored, duplicate


def verify():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    os.environ["XFORMERS_DISABLED"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, prior = project_path(ROOT), project_path(PRIOR)
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    result, plan, manifest = read(root/"result.json"), read(root/"plan.json"), read(root/"frames.json")
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(root/name) == digest
    encoder = plan["encoder"]
    source = Path(encoder["source_directory"])
    encoder_root = source.parent
    assert file_hash(encoder_root/"provenance.json") == plan["encoder_provenance_sha256"]
    for name, value in encoder["files"].items():
        assert file_hash(encoder_root/name) == value["sha256"]
    with zipfile.ZipFile(root/"source.zip") as z:
        for name, digest in plan["source_hashes"].items():
            assert file_hash(project_path(name)) == hashlib.sha256(z.read(name)).hexdigest() == digest
        for name, digest in encoder["source_python_hashes"].items():
            assert file_hash(source/name) == hashlib.sha256(z.read("encoder/"+name)).hexdigest() == digest
        assert z.read("encoder/LICENSE") == (source/"LICENSE").read_bytes()
    old_result_path = project_path("docs/experiments/candidate56_texture_deletion_result_20261015.json")
    assert file_hash(old_result_path) == plan["prior_result_sha256"]
    for name, digest in read(old_result_path)["artifacts_sha256"].items():
        assert file_hash(prior/name) == digest
    assert file_hash(prior/"plan.json") == plan["prior_plan_sha256"]
    assert file_hash(prior/"frames.json") == plan["prior_frames_manifest_sha256"]
    old, old_manifest = read(prior/"plan.json"), read(prior/"frames.json")
    for name, digest in old["source_hashes"].items():
        assert file_hash(project_path(name)) == digest
    for installed in old["installed_sources"].values():
        assert file_hash(Path(installed["path"])) == installed["sha256"]
    assert plan["image_selection"] == old["image_selection"]
    assert len({s[:7] for ss in plan["image_selection"].values() for s in ss}) == 192
    frames, replays, patches = {s: [] for s in plan["image_selection"]}, [], 0
    for split, stems in plan["image_selection"].items():
        for position, stem in enumerate(stems):
            entry, original = manifest[stem], old_manifest[stem]
            assert file_hash(root/entry["file"]) == entry["sha256"]
            assert file_hash(root/entry["inputs_file"]) == entry["inputs_sha256"]
            assert file_hash(prior/original["file"]) == entry["prior_frame_sha256"] == original["sha256"]
            image_path = project_path(f"data/visdrone/images/train/{stem}.jpg")
            meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            assert file_hash(image_path) == entry["image_sha256"] == old["inputs"][stem]["image_sha256"]
            assert file_hash(meta_path) == old["inputs"][stem]["metadata_sha256"]
            with np.load(root/entry["file"], allow_pickle=False) as z:
                f = {k: z[k].copy() for k in z.files}
            with np.load(prior/original["file"], allow_pickle=False) as z:
                for name in ("reference", "selected", "base", "labels", "eligible", "duplicate", "ignored"):
                    np.testing.assert_array_equal(f[name], z[name])
            tp, ignored, duplicate = independent_labels(f["reference"], read(meta_path))
            for name, rebuilt in (("labels", tp), ("ignored", ignored), ("duplicate", duplicate),
                                  ("eligible", ~(ignored | duplicate)[f["selected"]])):
                np.testing.assert_array_equal(f[name], rebuilt)
            image = cv2.imread(str(image_path))
            inputs = []
            assert len(entry["patches"]) == len(f["selected"])
            for row, pair in zip(f["selected"], entry["patches"], strict=True):
                for scale, record in zip((1., 2.), pair, strict=True):
                    values, rect, canvas = independently_preprocessed(image, f["reference"][row, :4], scale)
                    assert rect == record["rectangle"]
                    assert hashlib.sha256(canvas.tobytes()).hexdigest() == record["canvas_sha256"]
                    assert hashlib.sha256(values.tobytes()).hexdigest() == record["tensor_sha256"]
                    if position == 0:
                        inputs.append(values)
                    patches += 1
            assert f["own"].shape == f["context"].shape == (len(f["selected"]), 384)
            assert np.isfinite(f["own"]).all() and np.isfinite(f["context"]).all()
            if position == 0:
                expected = np.empty((len(inputs), 384), np.float32)
                expected[0::2], expected[1::2] = f["own"], f["context"]
                replays.append((np.array(inputs), expected))
            f["stem"] = stem
            frames[split].append(f)
    assert patches == 4014
    sys.path.insert(0, str(source))
    from dinov2.hub.backbones import dinov2_vits14
    model = dinov2_vits14(pretrained=False)
    model.load_state_dict(torch.load(encoder_root/"dinov2_vits14_pretrain.pth", map_location="cpu", weights_only=True), strict=True)
    model = model.float().cuda().eval().requires_grad_(False)
    replayed = 0
    with torch.inference_mode():
        for values, expected in replays:
            actual = model(torch.from_numpy(values).cuda()).cpu().numpy()
            np.testing.assert_array_equal(actual, expected)
            replayed += len(actual)
    stats = read(root/"normalization.json")
    base_values = np.concatenate([f["base"][f["eligible"]] for f in frames["fit"]])
    content = np.concatenate([f[name][f["eligible"]] for f in frames["fit"] for name in ("own", "context")])
    compare(dict(base=dict(mean=base_values.mean(0).tolist(), std=np.maximum(base_values.std(0), 1e-6).tolist()),
                 content=dict(mean=content.mean(0).tolist(), std=np.maximum(content.std(0), 1e-6).tolist())), stats)
    for fs in frames.values():
        by_class = {}
        for i, f in enumerate(fs):
            for j, row in enumerate(f["selected"]):
                by_class.setdefault(int(f["reference"][row, 5]), []).append((i, j))
        for rows in by_class.values():
            rows.sort(key=lambda ij: hashlib.sha256(f"candidate58-donor-v1:{fs[ij[0]]['stem']}:{fs[ij[0]]['selected'][ij[1]]}".encode()).hexdigest())
        for i, f in enumerate(fs):
            base = (f["base"]-stats["base"]["mean"])/stats["base"]["std"]
            own = (f["own"]-stats["content"]["mean"])/stats["content"]["std"]
            context = (f["context"]-stats["content"]["mean"])/stats["content"]["std"]
            donors, shuffled = np.full((len(own), 2), -1, int), np.zeros_like(own)
            for j, row in enumerate(f["selected"]):
                order = by_class[int(f["reference"][row, 5])]
                k = order.index((i, j))
                donor = next((order[(k+d) % len(order)] for d in range(1, len(order)) if order[(k+d) % len(order)][0] != i), None)
                if donor is None:
                    own[j] = context[j] = 0
                else:
                    di, dj = donor
                    donors[j] = donor
                    shuffled[j] = (fs[di]["own"][dj]-stats["content"]["mean"])/stats["content"]["std"]
            f["inputs"] = {a: np.column_stack((base, extra)) for a, extra in
                (("base", np.zeros_like(own)), ("own", own), ("context", context), ("shuffled", shuffled))}
            with np.load(root/manifest[f["stem"]]["inputs_file"], allow_pickle=False) as z:
                np.testing.assert_array_equal(donors, z["donors"])
                for a in ARMS:
                    compare(f["inputs"][a], z[a])
    with np.load(root/"coefficients.npz", allow_pickle=False) as z:
        coefficients = {a: z[a].copy() for a in ARMS}
    labels = np.concatenate([f["labels"][f["selected"]][f["eligible"]] for f in frames["fit"]])
    p = np.concatenate([f["reference"][f["selected"], 4][f["eligible"]] for f in frames["fit"]])
    p = np.clip(p, 1e-6, 1-1e-6)
    offset = np.log(p/(1-p))
    gradients = {}
    for a, theta in coefficients.items():
        x = np.concatenate([f["inputs"][a][f["eligible"]] for f in frames["fit"]])
        logits = offset+x@theta[:-1]+theta[-1]
        residual = (expit(logits)-labels)/len(labels)
        gradient = np.r_[x.T@residual+.01*theta[:-1], residual.sum()]
        objective = float(np.mean(np.logaddexp(0, logits)-labels*logits)+.005*np.dot(theta[:-1], theta[:-1]))
        compare(objective, result["training"][a]["objective"])
        gradients[a] = float(abs(gradient).max())
        compare(gradients[a], result["training"][a]["gradient_inf"])
    metrics, traces, checks = {}, {}, {}
    policy = plan["registration"]["policy_per_half"]
    for split in ("screen", "confirmation"):
        traces[split], labels, scores = [], [], {a: [] for a in ARMS}
        for f in frames[split]:
            keep, y = f["eligible"], f["labels"][f["selected"]]
            p = np.clip(f["reference"][f["selected"], 4], 1e-6, 1-1e-6)
            offset = np.log(p/(1-p))
            values = {a: offset+f["inputs"][a]@t[:-1]+t[-1] for a, t in coefficients.items()}
            with np.load(root/manifest[f["stem"]]["inputs_file"], allow_pickle=False) as z:
                for a in ARMS:
                    compare(values[a], z[a+"_logits"])
            labels.extend(y[keep])
            for a in ARMS:
                scores[a].extend(values[a][keep])
            traces[split].append(dict(rows=int(keep.sum()), tp=int(y[keep].sum()), fp=int((~y[keep]).sum()), images=int(keep.any()),
                loss_sums={a: math.fsum(np.logaddexp(0, v[keep])-y[keep]*v[keep]) for a, v in values.items()}))
        m = {k: sum(t[k] for t in traces[split]) for k in ("rows", "tp", "fp", "images")}
        m["NLL"] = {a: math.fsum(t["loss_sums"][a] for t in traces[split])/m["rows"] for a in ARMS}
        y = np.array(labels, bool)
        m["AUROC"] = {a: float((rankdata(v)[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum())) for a, v in scores.items()}
        draws, gains = np.random.default_rng(0).multinomial(64, [1/64]*64, size=400), []
        for draw in draws:
            mass = math.fsum(int(w)*t["rows"] for w, t in zip(draw, traces[split]))
            assert mass > 0
            gains.append(math.fsum(int(w)*(t["loss_sums"]["context"]-t["loss_sums"]["own"]) for w, t in zip(draw, traces[split]))/mass)
        m["context_minus_own_NLL_bootstrap95ci"] = np.percentile(gains, [2.5, 97.5]).tolist()
        metrics[split] = m
        ci, nll, auc = m["context_minus_own_NLL_bootstrap95ci"], m["NLL"], m["AUROC"]
        checks[split] = dict(tp=m["tp"] >= policy["tp_min"], fp=m["fp"] >= policy["nonduplicate_fp_min"],
            images=m["images"] >= policy["images_min"], vs_base=nll["base"]-nll["own"] >= policy["NLL_gain_vs_base_min"],
            vs_context=nll["context"]-nll["own"] >= policy["NLL_gain_vs_context_min"],
            vs_shuffled=nll["shuffled"]-nll["own"] >= policy["NLL_gain_vs_shuffled_min"],
            AUROC=auc["own"]-auc["context"] >= policy["AUROC_gain_vs_context_min"],
            interval=ci[0] > policy["NLL_gain_vs_context_bootstrap_lower_min_exclusive"])
    compare(metrics, result["metrics"])
    compare(traces, read(root/"per_image.json"))
    assert checks == result["decision"]["checks"]
    numerical = all(result["training"][a]["solver_success"] and gradients[a] <= 1e-5 for a in ARMS)
    assert numerical == result["decision"]["numerical"]
    passed = numerical and all(all(c.values()) for c in checks.values())
    expected = "REPLICATED_EXTERNAL_CONTENT_INFORMATION_SUPPORTS_SEPARATE_DEPLOYMENT_COST_AND_TRANSFER_PILOT" if passed else "STOP_THIS_FROZEN_DINO_OWN_BOX_LINEAR_RECIPE"
    assert expected == result["decision"]["decision"]
    timing = result["timing"]
    assert sum(timing["batches"]) == 4014 and len(timing["batches"]) == result["verification"]["encoder_normal_forward_batches"]
    audit = dict(all_source_archive_encoder_weight_source_prior_frame_image_metadata_inputs_hashes_verified=True,
        exact_original_queries_bases_labels_eligibility_and_sequence_disjoint_splits_verified=True,
        independently_recreated_crop_canvases_and_FP32normalized_inputs=patches,
        first_image_each_split_encoder_feature_batches_replayed_exact=3,
        independently_reextracted_feature_vectors=replayed,
        all_fit_only_normalization_donors_inputs_convex_objectives_and_stationarity_rebuilt=True,
        all_NLL_rank_AUROC_paired_bootstrap_and_seventeen_gates_rebuilt=True,
        four_convex_fit_gradient_inf=gradients,
        additional_JPEG_decodes=192, additional_encoder_forward_batches=3, additional_YOLO_forwards=0,
        added_model_updates=0, verifier_sha256=file_hash(Path(__file__)),
        limits="Only first3fixedimagebatches(56vectors) independently re-extracted;otherfeaturevectors are hashedcachedvalues. Cropresize sharescv2kernel,encoderreplay uses sameofficialsource/weights/runtime. Measurements are extraction-batch timings,not fullonline deploymentlatency. Previouslydiagnosed trainingdata,andfoundationpretrainingoverlap unknown;noAP,causality,generalization or novelty proof.")
    destination = root/"independent_verification.json"
    assert not destination.exists()
    destination.write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    verify()
