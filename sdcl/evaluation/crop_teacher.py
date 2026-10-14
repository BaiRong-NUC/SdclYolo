"""Prerequisite for training-only crop-teacher transfer, not detection efficacy."""

import inspect
import json
from pathlib import Path
from time import perf_counter
import zipfile

import cv2
import numpy as np
from scipy.special import logsumexp, softmax
import torch
from ultralytics import YOLO
from ultralytics.data import augment as augment_module
from ultralytics.data.augment import LetterBox
from ultralytics.nn.modules import head as head_module

from sdcl.config import project_path
from sdcl.evaluation.phase_screen import BASELINE, BASELINE_SHA256
from sdcl.evaluation.scale_gradient_conflict import model_digest
from sdcl.evaluation.size_analysis import file_hash
from sdcl.evaluation.slicing_screen import tile_windows
from sdcl.postprocess.rescue import save_json
from sdcl.training.trainer import check_version


REGISTRATION = "docs/experiments/candidate52_crop_teacher_registration_20261014.json"
OUTPUT = "output/analysis/crop-teacher-fixed-20261014"
PRIOR = "output/analysis/assignment-reachability-fixed-20261014"
ARMS = ("student", "teacher1", "confidence", "consensus")
GROUPS = {"all": np.inf, "tiny": 256., "small_all": 1024.}


def center_indices(gt, window):
    x1, y1, x2, y2 = window
    width, height = x2-x1, y2-y1
    ratio = min(960/width, 960/height)
    pad = np.array([round((960-round(width*ratio))/2-.1),
                    round((960-round(height*ratio))/2-.1)])
    centers = (gt[:, :2]+gt[:, 2:])/2
    position = (centers-[x1, y1])*ratio+pad
    xy = np.floor(position/8).astype(int).clip(0, 119)
    return xy[:, 1]*120+xy[:, 0]


def choose_teachers(gt, windows):
    windows = np.asarray(windows, float)
    available = ((gt[:, None, :2] >= windows[None, :, :2]) &
                 (gt[:, None, 2:] <= windows[None, :, 2:])).all(-1)
    available[:, 0] = False
    wh = windows[:, 2:]-windows[:, :2]
    margin = np.minimum((gt[:, None, :2]-windows[None, :, :2])/wh[None],
                        (windows[None, :, 2:]-gt[:, None, 2:])/wh[None]).min(-1)
    teachers = np.zeros((len(gt), 2), int)
    for row in range(len(gt)):
        views = sorted(np.flatnonzero(available[row]), key=lambda v: (-margin[row, v], int(v)))
        teachers[row, :min(2, len(views))] = views[:2]
    return available, margin, teachers, available.sum(1) >= 2


def teacher_arms(sampled, teachers, active):
    student = sampled[:, 0].astype(float)
    row = np.arange(len(student))
    first, second = (sampled[row, teachers[:, k]].astype(float) for k in (0, 1))
    confidence = active & (softmax(first, axis=1).max(1) > softmax(student, axis=1).max(1))
    consensus = active & (first.argmax(1) == second.argmax(1))
    masks = {"student": np.zeros(len(row), bool), "teacher1": active,
             "confidence": confidence, "consensus": consensus}
    arms = {a: np.where(mask[:, None], first, student) for a, mask in masks.items()}
    return arms, masks


def frame_metrics(arms, masks, active, classes, areas):
    row = np.arange(len(classes))
    nll = {a: logsumexp(v, axis=1)-v[row, classes] for a, v in arms.items()}
    correct = {a: v.argmax(1) == classes for a, v in arms.items()}
    result = {}
    for group, limit in GROUPS.items():
        selected = active & (areas < limit)
        result[group] = dict(rows=int(selected.sum()), images=int(selected.any()),
            NLL_sum={a: float(nll[a][selected].sum()) for a in ARMS},
            correct={a: int(correct[a][selected].sum()) for a in ARMS},
            teacher1_recovered=int((selected & ~correct["student"] & correct["teacher1"]).sum()),
            teacher1_lost=int((selected & correct["student"] & ~correct["teacher1"]).sum()),
            action_rows={a: int((selected & masks[a]).sum()) for a in ARMS})
    return result


def summarize(traces):
    metrics = {}
    for group in GROUPS:
        frames = [t[group] for t in traces]
        count = sum(f["rows"] for f in frames)
        sums = {a: sum(f["NLL_sum"][a] for f in frames) for a in ARMS}
        correct = {a: sum(f["correct"][a] for f in frames) for a in ARMS}
        metrics[group] = dict(rows=count, images=sum(f["images"] for f in frames),
            NLL={a: v/count if count else None for a, v in sums.items()},
            top1_percent={a: 100*v/count if count else None for a, v in correct.items()},
            teacher1_recovered=sum(f["teacher1_recovered"] for f in frames),
            teacher1_lost=sum(f["teacher1_lost"] for f in frames),
            action_rows={a: sum(f["action_rows"][a] for f in frames) for a in ARMS})
    counts = np.array([t["tiny"]["rows"] for t in traces])
    delta = np.array([t["tiny"]["NLL_sum"]["confidence"]-t["tiny"]["NLL_sum"]["consensus"] for t in traces])
    weights = np.random.default_rng(0).multinomial(len(traces), np.full(len(traces), 1/len(traces)), size=400)
    denominators = weights@counts
    if np.any(denominators <= 0):
        interval = None
    else:
        interval = np.percentile(weights@delta/denominators, [2.5, 97.5]).tolist()
    metrics["tiny"]["NLL_gain_vs_confidence_image_bootstrap95ci"] = interval
    return metrics


def decide(metrics, policy):
    tiny, all_metrics = metrics["tiny"], metrics["all"]
    nll, top = tiny["NLL"], tiny["top1_percent"]
    interval = tiny["NLL_gain_vs_confidence_image_bootstrap95ci"]
    checks = dict(rows=tiny["rows"] >= policy["tiny_primary_rows_min"],
                  images=tiny["images"] >= policy["tiny_primary_images_min"],
                  recovered=tiny["teacher1_recovered"] >= policy["tiny_teacher1_recovered_min"],
                  lost=tiny["teacher1_lost"] >= policy["tiny_teacher1_lost_min"],
                  NLL_vs_student=nll["student"] is not None and nll["student"]-nll["consensus"] >= policy["tiny_consensus_NLL_gain_vs_student_min"],
                  NLL_vs_confidence=nll["confidence"] is not None and nll["confidence"]-nll["consensus"] >= policy["tiny_consensus_NLL_gain_vs_confidence_min"],
                  top1=top["student"] is not None and top["consensus"]-top["student"] >= policy["tiny_consensus_top1_gain_vs_student_min_pp"],
                  interval=interval is not None and interval[0] > policy["tiny_consensus_NLL_gain_vs_confidence_bootstrap_lower_min"],
                  all_NLL=all_metrics["NLL"]["student"] is not None and all_metrics["NLL"]["student"]-all_metrics["NLL"]["consensus"] >= policy["all_consensus_NLL_gain_vs_student_min"])
    return dict(decision="TEACHER_SUPERVISION_SUPPORTS_SEPARATE_BOUNDED_TRANSFER_PILOT" if all(checks.values()) else
                "STOP_THIS_TWO_FULL_CROP_CONSENSUS_TEACHER_RECIPE", checks=checks,
                scope="GT-anchored conditional classification-supervision proxy; no learned transfer, AP, detection performance or novelty evidence.")


def run(output=OUTPUT, device="0"):
    started = perf_counter()
    torch.set_num_threads(1)
    check_version()
    if device != "0" or not torch.cuda.is_available():
        raise ValueError("Fixed CUDA0 FP32 screen.")
    output, prior = project_path(output), project_path(PRIOR)
    if output.exists():
        raise FileExistsError(output)
    registration = json.loads(project_path(REGISTRATION).read_text())
    previous = json.loads((prior/"plan.json").read_text())
    prior_frames = json.loads((prior/"frames.json").read_text())
    assert file_hash(project_path(BASELINE)) == BASELINE_SHA256
    paths = (REGISTRATION, "sdcl/evaluation/crop_teacher.py", "scripts/evaluation/diagnose_crop_teacher.py",
             "tests/test_crop_teacher.py", "sdcl/evaluation/slicing_screen.py",
             "sdcl/evaluation/scale_gradient_conflict.py", "sdcl/config.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    installed = {"augment.py": Path(inspect.getfile(augment_module)), "head.py": Path(inspect.getfile(head_module))}
    output.mkdir(parents=True)
    (output/"frames").mkdir()
    with zipfile.ZipFile(output/"source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(project_path(p), p)
        for name, path in installed.items():
            z.write(path, "installed/"+name)
    plan = dict(registration=registration, source_hashes=hashes, source_archive_sha256=file_hash(output/"source.zip"),
                installed_sources={n: dict(path=str(p), sha256=file_hash(p)) for n, p in installed.items()},
                images=previous["images"], prior_plan_sha256=file_hash(prior/"plan.json"),
                prior_frames_sha256=file_hash(prior/"frames.json"), checkpoint_sha256=BASELINE_SHA256)
    save_json(output/"plan.json", plan)
    model = YOLO(str(project_path(BASELINE))).model.float().cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    digest = model_digest(model)
    letterbox = LetterBox(new_shape=(960, 960), auto=False, scaleup=True, center=True)
    traces, manifest, all_gt, primary_gt = [], {}, 0, 0
    with torch.inference_mode():
        for index, stem in enumerate(plan["images"], 1):
            image_path = project_path(f"data/visdrone/images/train/{stem}.jpg")
            metadata_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            assert file_hash(image_path) == prior_frames[stem]["image_sha256"]
            assert file_hash(metadata_path) == prior_frames[stem]["metadata_sha256"]
            metadata = json.loads(metadata_path.read_text())
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(image_path)
            height, width = image.shape[:2]
            assert [width, height] == metadata["original_size"]
            objects = metadata["valid_objects"]
            gt = np.array([o["box_xyxy"] for o in objects], float).reshape(-1, 4)
            classes = np.array([o["class_id"] for o in objects], int)
            areas = (gt[:, 2:]-gt[:, :2]).prod(1)*(640/max(width, height))**2
            windows = [(0, 0, width, height), *tile_windows(width, height)]
            sources, indices = [], []
            for window in windows:
                x1, y1, x2, y2 = window
                crop = image[y1:y2, x1:x2].copy()
                rgb = letterbox(image=crop)[:, :, ::-1].transpose(2, 0, 1)
                tensor = torch.from_numpy(np.ascontiguousarray(rgb))[None].float().cuda()/255
                raw = model(tensor)[1]
                assert tuple(raw["feats"][0].shape[2:]) == (120, 120)
                logits = raw["scores"][0, :, :14400].T.float().cpu().numpy().copy()
                assert logits.shape == (14400, 10) and np.isfinite(logits).all()
                sources.append(logits)
                indices.append(center_indices(gt, window))
            source_logits, native_indices = np.stack(sources), np.stack(indices, axis=1)
            sampled = np.stack([source_logits[v, native_indices[:, v]] for v in range(5)], axis=1)
            available, margin, teachers, active = choose_teachers(gt, windows)
            arms, masks = teacher_arms(sampled, teachers, active)
            trace = frame_metrics(arms, masks, active, classes, areas)
            traces.append(trace)
            all_gt += len(gt)
            primary_gt += int(active.sum())
            path = output/"frames"/(stem+".npz")
            np.savez_compressed(path, source_logits=source_logits, native_indices=native_indices,
                                sampled_logits=sampled, gt=gt, classes=classes, areas=areas,
                                windows=np.array(windows), available=available, margins=margin,
                                teachers=teachers, active=active, **{a: v for a, v in arms.items()})
            manifest[stem] = dict(file=path.relative_to(output).as_posix(), sha256=file_hash(path),
                                 image_sha256=file_hash(image_path), metadata_sha256=file_hash(metadata_path))
            if index % 16 == 0:
                print(f"Crop-teacher supervision audit: {index}/64", flush=True)
    assert digest == model_digest(model)
    metrics = summarize(traces)
    decision = decide(metrics, registration["policy"])
    verification = dict(image_decodes=64, frozen_model_forwards=320, parameter_updates=0, gradients=0,
                        AP_evaluations=0, all_gt=all_gt, primary_gt=primary_gt,
                        parameter_and_buffer_digest_before=digest, parameter_and_buffer_digest_after=model_digest(model),
                        source_unchanged=all(file_hash(project_path(p)) == h for p, h in hashes.items()),
                        elapsed_seconds=perf_counter()-started)
    for name, value in (("frames.json", manifest), ("per_image.json", traces), ("metrics.json", metrics),
                        ("decision.json", decision), ("verification.json", verification)):
        save_json(output/name, value)
    result = dict(experiment=registration["experiment"], date="2026-10-14", output=str(output),
                  metrics=metrics, decision=decision, verification=verification,
                  artifacts_sha256={p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()})
    save_json(output/"result.json", result)
    return result
