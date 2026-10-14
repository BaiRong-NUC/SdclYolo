"""Fixed independent-encoder content information screen on cached tiny queries."""

import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import zipfile

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from sdcl.config import project_path
from sdcl.evaluation.classifier_nullspace import fit
from sdcl.evaluation.native_capacity import checked_prior
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.rescue import save_json


REGISTRATION = "docs/experiments/candidate58_external_content_registration_20261015.json"
OUTPUT = "output/analysis/external-content-fixed-20261015"
ENCODER = "output/tools/dinov2-fixed-20261015"
WEIGHT_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
ARMS = ("base", "own", "context", "shuffled")


def crop_input(image, box, scale):
    box = np.asarray(box, float)
    center, half = (box[:2]+box[2:])/2, (box[2:]-box[:2])*scale/2
    rectangle = np.r_[np.floor(center-half), np.ceil(center+half)].astype(int)
    x1, y1, x2, y2 = rectangle
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Nonpositive registered crop; no sample replacement.")
    crop = np.full((y2-y1, x2-x1, 3), 114, np.uint8)
    lx, ly, hx, hy = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)
    if hx > lx and hy > ly:
        crop[ly-y1:hy-y1, lx-x1:hx-x1] = image[ly:hy, lx:hx]
    ratio = min(224/crop.shape[1], 224/crop.shape[0])
    rw, rh = round(crop.shape[1]*ratio), round(crop.shape[0]*ratio)
    if min(rw, rh) < 1:
        raise ValueError("Registered aspect ratio collapses crop; do not change interpolation or selection.")
    resized = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((224, 224, 3), 114, np.uint8)
    left, top = (224-rw)//2, (224-rh)//2
    canvas[top:top+rh, left:left+rw] = resized
    values = canvas[:, :, ::-1].copy().astype(np.float32)/np.float32(255)
    values = (values-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
    return values.transpose(2, 0, 1).copy(), rectangle, canvas


def encoder_state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checked_encoder():
    root = project_path(ENCODER)
    provenance = json.loads((root/"provenance.json").read_text())
    assert provenance["revision"] == REVISION
    for name, value in provenance["files"].items():
        assert file_hash(root/name) == value["sha256"]
    assert file_hash(root/"dinov2_vits14_pretrain.pth") == WEIGHT_SHA
    source = Path(provenance["source_directory"])
    for name, digest in provenance["source_python_hashes"].items():
        assert file_hash(source/name) == digest
    return root, source, provenance


def donor_indices(frames):
    mapping = {}
    for cls in range(10):
        rows = [(i, j) for i, f in enumerate(frames) for j, row in enumerate(f["selected"])
                if int(f["reference"][row, 5]) == cls]
        rows.sort(key=lambda ij: hashlib.sha256(
            f"candidate58-donor-v1:{frames[ij[0]]['stem']}:{frames[ij[0]]['selected'][ij[1]]}".encode()).hexdigest())
        for k, ij in enumerate(rows):
            mapping[ij] = next((rows[(k+d) % len(rows)] for d in range(1, len(rows))
                                if rows[(k+d) % len(rows)][0] != ij[0]), None)
    return mapping


def inputs_for(frames):
    base_values = np.concatenate([f["base"][f["eligible"]] for f in frames["fit"]])
    content_values = np.concatenate([f[name][f["eligible"]] for f in frames["fit"] for name in ("own", "context")])
    stats = dict(base=dict(mean=base_values.mean(0).tolist(), std=np.maximum(base_values.std(0), 1e-6).tolist()),
                 content=dict(mean=content_values.mean(0).tolist(), std=np.maximum(content_values.std(0), 1e-6).tolist()))
    for fs in frames.values():
        mapping = donor_indices(fs)
        for i, f in enumerate(fs):
            base = (f["base"]-stats["base"]["mean"])/stats["base"]["std"]
            own = (f["own"]-stats["content"]["mean"])/stats["content"]["std"]
            context = (f["context"]-stats["content"]["mean"])/stats["content"]["std"]
            shuffled = np.zeros_like(own)
            donors = np.full((len(own), 2), -1, int)
            for j in range(len(own)):
                donor = mapping[(i, j)]
                if donor is None:
                    own[j] = context[j] = 0
                else:
                    di, dj = donor
                    shuffled[j] = (fs[di]["own"][dj]-stats["content"]["mean"])/stats["content"]["std"]
                    donors[j] = donor
            f["donors"] = donors
            f["inputs"] = {a: np.column_stack((base, extra)) for a, extra in
                           (("base", np.zeros_like(own)), ("own", own), ("context", context), ("shuffled", shuffled))}
    return stats


def summarize(frames, coefficients):
    traces, ys, scores = [], [], {a: [] for a in ARMS}
    for f in frames:
        keep, y = f["eligible"], f["labels"][f["selected"]]
        p = np.clip(f["reference"][f["selected"], 4], 1e-6, 1-1e-6)
        offset = np.log(p/(1-p))
        outputs = {a: offset+f["inputs"][a]@t[:-1]+t[-1] for a, t in coefficients.items()}
        f["output_logits"] = outputs
        ys.extend(y[keep])
        for a in ARMS:
            scores[a].extend(outputs[a][keep])
        traces.append(dict(rows=int(keep.sum()), tp=int(y[keep].sum()), fp=int((~y[keep]).sum()), images=int(keep.any()),
            loss_sums={a: float((np.logaddexp(0, z[keep])-y[keep]*z[keep]).sum()) for a, z in outputs.items()}))
    total = {k: sum(t[k] for t in traces) for k in ("rows", "tp", "fp", "images")}
    total["NLL"] = {a: sum(t["loss_sums"][a] for t in traces)/total["rows"] for a in ARMS}
    total["AUROC"] = {a: float(roc_auc_score(ys, values)) if total["tp"] and total["fp"] else None for a, values in scores.items()}
    draws = np.random.default_rng(0).multinomial(64, [1/64]*64, size=400)
    mass = draws@np.array([t["rows"] for t in traces])
    gain = draws@np.array([t["loss_sums"]["context"]-t["loss_sums"]["own"] for t in traces])
    total["context_minus_own_NLL_bootstrap95ci"] = np.percentile(gain/mass, [2.5, 97.5]).tolist() if (mass > 0).all() else None
    return total, traces


def decide(metrics, policy, numerical):
    checks = {}
    for half, m in metrics.items():
        nll, auc, ci = m["NLL"], m["AUROC"], m["context_minus_own_NLL_bootstrap95ci"]
        populated = bool(m["tp"] and m["fp"])
        checks[half] = dict(tp=m["tp"] >= policy["tp_min"], fp=m["fp"] >= policy["nonduplicate_fp_min"],
            images=m["images"] >= policy["images_min"],
            vs_base=populated and nll["base"]-nll["own"] >= policy["NLL_gain_vs_base_min"],
            vs_context=populated and nll["context"]-nll["own"] >= policy["NLL_gain_vs_context_min"],
            vs_shuffled=populated and nll["shuffled"]-nll["own"] >= policy["NLL_gain_vs_shuffled_min"],
            AUROC=populated and auc["own"]-auc["context"] >= policy["AUROC_gain_vs_context_min"],
            interval=ci is not None and ci[0] > policy["NLL_gain_vs_context_bootstrap_lower_min_exclusive"])
    passed = numerical and all(all(c.values()) for c in checks.values())
    return dict(numerical=numerical, checks=checks,
        decision="REPLICATED_EXTERNAL_CONTENT_INFORMATION_SUPPORTS_SEPARATE_DEPLOYMENT_COST_AND_TRANSFER_PILOT" if passed else "STOP_THIS_FROZEN_DINO_OWN_BOX_LINEAR_RECIPE",
        scope="Fixed-query external-feature information on previously diagnosed train images;no deployed gain,AP,affordable-transfer or novelty proof.")


def run(output=OUTPUT):
    started = perf_counter()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["XFORMERS_DISABLED"] = "1"
    root = project_path(output)
    if root.exists():
        raise FileExistsError(root)
    prior, previous, old_manifest, previous_result_path = checked_prior()
    encoder_root, encoder_source, provenance = checked_encoder()
    registration = json.loads(project_path(REGISTRATION).read_text())
    selection = {s: previous["image_selection"][s] for s in ("fit", "screen", "confirmation")}
    assert all(len(v) == 64 for v in selection.values())
    paths = (REGISTRATION, "sdcl/evaluation/external_content.py", "scripts/evaluation/diagnose_external_content.py",
             "tests/test_external_content.py", "sdcl/evaluation/classifier_nullspace.py", "sdcl/evaluation/native_capacity.py",
             "sdcl/config.py", "sdcl/evaluation/size_analysis.py", "sdcl/postprocess/rescue.py")
    sources = {p: file_hash(project_path(p)) for p in paths}
    root.mkdir(parents=True)
    (root/"frames").mkdir()
    with zipfile.ZipFile(root/"source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(project_path(p), p)
        for p in provenance["source_python_hashes"]:
            z.write(encoder_source/p, "encoder/"+p)
        z.write(encoder_source/"LICENSE", "encoder/LICENSE")
    plan = dict(registration=registration, source_hashes=sources, source_archive_sha256=file_hash(root/"source.zip"),
        encoder=provenance, encoder_provenance_sha256=file_hash(encoder_root/"provenance.json"),
        prior_result_sha256=file_hash(previous_result_path), prior_plan_sha256=file_hash(prior/"plan.json"),
        prior_frames_manifest_sha256=file_hash(prior/"frames.json"), image_selection=selection,
        selected_before_new_features=True, torch_version=torch.__version__, cv2_version=cv2.__version__,
        xformers_disabled=True, TF32_disabled=True)
    save_json(root/"plan.json", plan)
    sys.path.insert(0, str(encoder_source))
    from dinov2.hub.backbones import dinov2_vits14
    model = dinov2_vits14(pretrained=False)
    model.load_state_dict(torch.load(encoder_root/"dinov2_vits14_pretrain.pth", map_location="cpu", weights_only=True), strict=True)
    model = model.float().cuda().eval().requires_grad_(False)
    initial_state = encoder_state_hash(model)
    frames, manifest, forwards, patch_count, identity = {s: [] for s in selection}, {}, 0, 0, 0
    timings = dict(patch_preprocessing_seconds=0., synchronized_GPU_seconds=0., identity_GPU_seconds=0., batches=[])
    extraction_tick = perf_counter()
    with torch.inference_mode():
        model(torch.zeros(32, 3, 224, 224, device="cuda"))
        torch.cuda.synchronize()
        for split, stems in selection.items():
            for position, stem in enumerate(stems):
                entry = old_manifest[stem]
                assert file_hash(prior/entry["file"]) == entry["sha256"]
                path = project_path(f"data/visdrone/images/train/{stem}.jpg")
                assert file_hash(path) == previous["inputs"][stem]["image_sha256"]
                with np.load(prior/entry["file"], allow_pickle=False) as z:
                    f = {k: z[k].copy() for k in ("reference", "selected", "base", "labels", "eligible", "duplicate", "ignored")}
                image = cv2.imread(str(path))
                tick = perf_counter()
                inputs, patch_records = [], []
                for j in f["selected"]:
                    pair = []
                    for scale in (1., 2.):
                        tensor, rectangle, canvas = crop_input(image, f["reference"][j, :4], scale)
                        inputs.append(tensor)
                        pair.append(dict(rectangle=rectangle.tolist(), canvas_sha256=hashlib.sha256(canvas.tobytes()).hexdigest(),
                                         tensor_sha256=hashlib.sha256(tensor.tobytes()).hexdigest()))
                    patch_records.append(pair)
                timings["patch_preprocessing_seconds"] += perf_counter()-tick
                n = len(inputs)
                assert n <= 24
                features = np.empty((0, 384), np.float32)
                if n:
                    batch = torch.from_numpy(np.array(inputs)).cuda()
                    torch.cuda.synchronize()
                    tick = perf_counter()
                    output_features = model(batch)
                    torch.cuda.synchronize()
                    timings["synchronized_GPU_seconds"] += perf_counter()-tick
                    timings["batches"].append(n)
                    forwards += 1
                    features = output_features.cpu().numpy().copy()
                    assert features.shape == (n, 384) and np.isfinite(features).all()
                    if position == 0:
                        torch.cuda.synchronize()
                        tick = perf_counter()
                        replay = model(batch)
                        torch.cuda.synchronize()
                        timings["identity_GPU_seconds"] += perf_counter()-tick
                        assert torch.equal(output_features, replay)
                        identity += 1
                f.update(own=features[0::2], context=features[1::2], stem=stem)
                frames[split].append(f)
                destination = root/"frames"/(stem+".npz")
                np.savez_compressed(destination, **{k: v for k, v in f.items() if k != "stem"})
                manifest[stem] = dict(split=split, file=destination.relative_to(root).as_posix(), sha256=file_hash(destination),
                    prior_frame_sha256=entry["sha256"], image_sha256=file_hash(path), patches=patch_records)
                patch_count += n
                if len(manifest) % 32 == 0:
                    print(f"Frozen external content: {len(manifest)}/192;patches={patch_count}", flush=True)
    timings["extraction_wall_seconds"] = perf_counter()-extraction_tick
    assert patch_count == 4014 and identity == 3
    stats = inputs_for(frames)
    fit_frames = frames["fit"]
    y = np.concatenate([f["labels"][f["selected"]][f["eligible"]] for f in fit_frames])
    p = np.concatenate([f["reference"][f["selected"], 4][f["eligible"]] for f in fit_frames])
    p = np.clip(p, 1e-6, 1-1e-6)
    offset = np.log(p/(1-p))
    coefficients, training = {}, {}
    for arm in ARMS:
        x = np.concatenate([f["inputs"][arm][f["eligible"]] for f in fit_frames])
        coefficients[arm], training[arm] = fit(x, offset, y)
        print(f"External readout {arm}: {training[arm]}", flush=True)
    metrics, traces = {}, {}
    for split in ("screen", "confirmation"):
        metrics[split], traces[split] = summarize(frames[split], coefficients)
    (root/"inputs").mkdir()
    for fs in frames.values():
        for f in fs:
            destination = root/"inputs"/(f["stem"]+".npz")
            np.savez_compressed(destination, donors=f["donors"], **f["inputs"],
                                **{a+"_logits": z for a, z in f.get("output_logits", {}).items()})
            manifest[f["stem"]].update(inputs_file=destination.relative_to(root).as_posix(), inputs_sha256=file_hash(destination))
    np.savez_compressed(root/"coefficients.npz", **coefficients)
    decision = decide(metrics, registration["policy_per_half"], all(v["success"] for v in training.values()))
    verification = dict(JPEG_decodes=192, encoder_patch_features=patch_count, encoder_normal_forward_batches=forwards,
        encoder_zero_warmup_batches=1, encoder_exact_identity_replay_batches=identity, YOLO_detector_forwards=0,
        encoder_updates=0, detector_updates=0, linear_models_fit=4, AP_evaluations=0,
        encoder_state_sha256_before=initial_state, encoder_state_sha256_after=encoder_state_hash(model),
        encoder_weight_unchanged=file_hash(encoder_root/"dinov2_vits14_pretrain.pth") == WEIGHT_SHA,
        encoder_sources_unchanged=all(file_hash(encoder_source/n) == d for n, d in provenance["source_python_hashes"].items()),
        source_unchanged=all(file_hash(project_path(p)) == d for p, d in sources.items()),
        elapsed_seconds=perf_counter()-started)
    assert verification["encoder_state_sha256_before"] == verification["encoder_state_sha256_after"]
    for name, value in (("frames.json", manifest), ("normalization.json", stats), ("training.json", training),
        ("per_image.json", traces), ("metrics.json", metrics), ("decision.json", decision),
        ("verification.json", verification), ("timing.json", timings)):
        save_json(root/name, value)
    result = dict(experiment=registration["experiment"], date="2026-10-15", output=str(root), metrics=metrics,
        decision=decision, training=training, verification=verification, timing=timings,
        artifacts_sha256={p.name: file_hash(p) for p in sorted(root.iterdir()) if p.is_file()})
    save_json(root/"result.json", result)
    return result
