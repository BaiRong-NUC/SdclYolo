"""A fixed shared-parameter scale-conflict prerequisite, without any updates."""

import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import loss as loss_module
from ultralytics.utils import tal as tal_module
import yaml

from sdcl.config import SDCLConfig, project_path
from sdcl.data.dataset import IgnoreDataset
from sdcl.data.geometry import anchor_ignore_mask
from sdcl.evaluation.phase_screen import BASELINE, BASELINE_SHA256
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.rescue import save_json
from sdcl.training.criterion import SDCLLoss
from sdcl.training.trainer import check_version


REGISTRATION = "docs/experiments/candidate51_scale_gradient_conflict_registration_20261014.json"
OUTPUT = "output/analysis/scale-gradient-conflict-fixed-20261014"
PRIOR = "output/analysis/assignment-reachability-fixed-20261014"
PARAMETER = "model.16.cv2.conv.weight"


def shuffled_membership(stem, classes, tiny):
    membership = np.asarray(tiny, bool).copy()
    for cls in sorted(set(np.asarray(classes, int).tolist())):
        ids = np.flatnonzero(classes == cls)
        seed = int.from_bytes(hashlib.sha256(
            f"candidate51-scale-control-v1:{stem}:{cls}".encode()).digest()[:8], "little")
        membership[ids] = np.random.default_rng(seed).permutation(membership[ids])
    return membership


def gradient_statistics(total, tiny, control):
    total, tiny, control = (np.asarray(x, float) for x in (total, tiny, control))
    rest, other = total-tiny, total-control
    norms = [np.linalg.norm(x) for x in (tiny, rest, control, other)]
    if min(norms) <= 1e-12:
        return {"valid": False, "norms": norms}
    def describe(a, b):
        dot = float(a@b)
        return dict(cosine=dot/(np.linalg.norm(a)*np.linalg.norm(b)),
                    opposition_ratio=max(0., -dot)/float(a@a),
                    blocked=bool(a@total < 0), tiny_norm=float(np.linalg.norm(a)),
                    rest_norm=float(np.linalg.norm(b)))
    actual, shuffled = describe(tiny, rest), describe(control, other)
    return dict(valid=True, actual=actual, control=shuffled,
                control_minus_actual_cosine=shuffled["cosine"]-actual["cosine"])


def summarize(traces):
    valid = [t for t in traces if t["gradients"]["valid"]]
    effect = np.array([t["gradients"]["control_minus_actual_cosine"] for t in valid])
    if len(effect):
        weights = np.random.default_rng(0).multinomial(len(effect), np.full(len(effect), 1/len(effect)), size=400)
        interval = np.percentile(weights@effect/len(effect), [2.5, 97.5]).tolist()
    else:
        interval = None
    return dict(images=len(traces), valid_images=len(valid),
                changed_control_images=sum(t["changed_productive_gt"] > 0 for t in traces),
                changed_productive_gt=sum(t["changed_productive_gt"] for t in traces),
                actual_blocked_fraction=float(np.mean([t["gradients"]["actual"]["blocked"] for t in valid])) if valid else 0.,
                control_blocked_fraction=float(np.mean([t["gradients"]["control"]["blocked"] for t in valid])) if valid else 0.,
                actual_mean_cosine=float(np.mean([t["gradients"]["actual"]["cosine"] for t in valid])) if valid else None,
                control_mean_cosine=float(np.mean([t["gradients"]["control"]["cosine"] for t in valid])) if valid else None,
                mean_control_minus_actual_cosine=float(effect.mean()) if len(effect) else None,
                image_bootstrap95ci_control_minus_actual_cosine=interval)


def decide(metrics, policy):
    lower = metrics["image_bootstrap95ci_control_minus_actual_cosine"]
    checks = {
        "valid_images": metrics["valid_images"] >= policy["valid_images_min"],
        "changed_images": metrics["changed_control_images"] >= policy["changed_control_images_min"],
        "changed_gt": metrics["changed_productive_gt"] >= policy["changed_productive_gt_min"],
        "actual_blocking": metrics["actual_blocked_fraction"] >= policy["actual_blocked_fraction_min"],
        "excess_conflict": metrics["mean_control_minus_actual_cosine"] is not None and metrics["mean_control_minus_actual_cosine"] >= policy["mean_control_minus_actual_cosine_min"],
        "interval": lower is not None and lower[0] > policy["bootstrap_lower_control_minus_actual_cosine_min"],
    }
    return dict(decision="SCALE_CONFLICT_SUPPORTS_SEPARATE_BOUNDED_UPDATE_SCREEN" if all(checks.values()) else
                "STOP_THIS_SCALE_SPECIFIC_P3_GRADIENT_ROUTE", checks=checks,
                scope="Single-block first-order mechanism only; no updates, efficacy, AP, official evaluation or innovation proof.")


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(output=OUTPUT, device="0"):
    started = perf_counter()
    torch.set_num_threads(1)
    check_version()
    if device != "0" or not torch.cuda.is_available():
        raise ValueError("Fixed CUDA0 screen.")
    output, prior = project_path(output), project_path(PRIOR)
    if output.exists():
        raise FileExistsError(output)
    registration = json.loads(project_path(REGISTRATION).read_text())
    prior_result = json.loads((prior/"result.json").read_text())
    prior_plan = json.loads((prior/"plan.json").read_text())
    for name, digest in prior_result["artifacts_sha256"].items():
        assert file_hash(prior/name) == digest, name
    assert file_hash(project_path(BASELINE)) == BASELINE_SHA256
    paths = (REGISTRATION, "sdcl/evaluation/scale_gradient_conflict.py",
             "scripts/evaluation/diagnose_scale_gradient_conflict.py", "tests/test_scale_gradient_conflict.py",
             "sdcl/training/criterion.py", "sdcl/data/dataset.py", "sdcl/data/geometry.py",
             "sdcl/config.py", "sdcl/training/trainer.py")
    hashes = {p: file_hash(project_path(p)) for p in paths}
    installed = {"loss.py": Path(inspect.getfile(loss_module)), "tal.py": Path(inspect.getfile(tal_module))}
    output.mkdir(parents=True)
    (output/"frames").mkdir()
    with zipfile.ZipFile(output/"source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(project_path(p), p)
        for name, path in installed.items():
            z.write(path, "installed/"+name)
    plan = dict(registration=registration, source_hashes=hashes,
                source_archive_sha256=file_hash(output/"source.zip"),
                installed_sources={n: dict(path=str(p), sha256=file_hash(p)) for n, p in installed.items()},
                prior_plan_sha256=file_hash(prior/"plan.json"), prior_result_sha256=file_hash(prior/"result.json"),
                images=prior_plan["images"], checkpoint_sha256=BASELINE_SHA256, parameter=PARAMETER)
    save_json(output/"plan.json", plan)
    args = get_cfg(overrides={"imgsz": 960, "batch": 1, "workers": 0, "rect": False, "plots": False, "device": "0"})
    data_path = project_path("data/visdrone/dataset.yaml")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data.update(nc=len(data["names"]), channels=3, path=str(data_path.parent))
    dataset = IgnoreDataset(img_path=str(data_path.parent/"images/train"), imgsz=960, batch_size=1,
                            augment=False, hyp=args, rect=False, cache=False, single_cls=False,
                            stride=32, pad=0., prefix="gradient-proxy: ", task="detect", data=data)
    indices = {Path(p).stem: i for i, p in enumerate(dataset.im_files)}
    model = YOLO(str(project_path(BASELINE))).model.float().cuda().eval()
    for name, p in model.named_parameters():
        p.requires_grad_(name == PARAMETER)
    parameter = dict(model.named_parameters())[PARAMETER]
    assert tuple(parameter.shape) == (128, 192, 1, 1)
    model.args, model.sdcl_settings = args, SDCLConfig(enabled=False).to_dict()
    model.sdcl_ignore_regions, model.coverage_settings = True, {"enabled": False}
    criterion = SDCLLoss(model)
    captured = {}
    hook = criterion.assigner.register_forward_hook(lambda module, inputs, result: captured.update(result=result))
    original_digest = model_digest(model)
    manifest, traces = {}, []
    maximum = dict(probability=0., decoded_coordinate=0., target_mass=0., loss_reconstruction=0.)
    prior_frames = json.loads((prior/"frames.json").read_text())
    for index, stem in enumerate(plan["images"], 1):
        item = prior_frames[stem]
        assert file_hash(prior/item["file"]) == item["sha256"]
        with np.load(prior/item["file"]) as z:
            old = {k: z[k].copy() for k in z.files}
        batch = dataset.collate_fn([dataset[indices[stem]]])
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch["img"] = batch["img"].float()/255
        raw = model(batch["img"])[1]
        (fg, owner, assigned, anchors, stride), losses, _ = criterion.get_assigned_targets_and_loss(raw, batch)
        _, _, scores, captured_fg, captured_owner = captured["result"]
        assert torch.equal(fg, captured_fg) and torch.equal(owner, captured_owner)
        logits = raw["scores"].permute(0, 2, 1).contiguous()
        distances = raw["boxes"].permute(0, 2, 1).contiguous()
        decoded = criterion.bbox_decode(anchors, distances)
        as_numpy = lambda x: x.detach().cpu().numpy().copy()
        errors = dict(probability=float(np.max(np.abs(as_numpy(logits.sigmoid()[0])-old["probabilities"]))),
                      decoded_coordinate=float(np.max(np.abs(as_numpy(decoded[0]*stride)-old["decoded_boxes"]))),
                      target_mass=float(np.max(np.abs(as_numpy(scores[0].sum(-1))-old["target_mass"]))))
        assert errors["probability"] <= 2e-6 and errors["decoded_coordinate"] <= .002 and errors["target_mass"] <= 5e-6, errors
        np.testing.assert_array_equal(as_numpy(owner[0]), old["owner"])
        np.testing.assert_array_equal(as_numpy(fg[0]), old["foreground"])
        target_input = torch.cat((batch["batch_idx"][:, None], batch["cls"][:, :1], batch["bboxes"]), 1)
        gt_all = criterion.preprocess(target_input, 1, logits.new_tensor([960]*4))[0]
        gt, classes = gt_all[:, 1:], as_numpy(gt_all[:, 0]).astype(int)
        np.testing.assert_array_equal(as_numpy(gt), old["gt"])
        tiny = as_numpy((gt[:, 2:]-gt[:, :2]).prod(1)*(640/960)**2 < 256)
        shuffled = shuffled_membership(stem, classes, tiny)
        productive = fg & (scores.sum(-1) > 0)
        tiny_rows = productive & torch.as_tensor(tiny, device="cuda")[owner]
        control_rows = productive & torch.as_tensor(shuffled, device="cuda")[owner]
        ignored = anchor_ignore_mask(anchors*stride, batch["ignore_bboxes"], batch["ignore_batch_idx"], 1, (960, 960))
        bce = F.binary_cross_entropy_with_logits(logits, scores.to(logits.dtype), reduction="none")
        bce = bce.masked_fill((ignored & ~fg)[..., None], 0)
        denominator = scores.sum().clamp_min(1)
        size = logits.new_tensor([960, 960])
        def component(rows):
            iou, dfl = criterion.bbox_loss(distances, decoded, anchors, assigned/stride,
                                           scores, denominator, rows, size, stride)
            return iou*args.box+(bce[rows].sum()/denominator)*args.cls+dfl*args.dfl
        rebuilt = component(fg)+(bce[~fg].sum()/denominator)*args.cls
        total, tiny_loss, control_loss = losses.sum(), component(tiny_rows), component(control_rows)
        np.testing.assert_allclose(as_numpy(rebuilt), as_numpy(total), atol=1e-5, rtol=5e-5)
        errors["loss_reconstruction"] = abs(float(rebuilt.detach()-total.detach()))
        for k, value in errors.items():
            maximum[k] = max(maximum[k], value)
        gradients = [as_numpy(torch.autograd.grad(value, parameter, retain_graph=n < 2)[0]).reshape(-1)
                     for n, value in enumerate((total, tiny_loss, control_loss))]
        active_gt = np.unique(old["owner"][old["foreground"] & (old["target_mass"] > 0)])
        trace = dict(stem=stem, gradients=gradient_statistics(*gradients),
                     gt=len(tiny), tiny_gt=int(tiny.sum()),
                     changed_productive_gt=int((tiny[active_gt] != shuffled[active_gt]).sum()),
                     losses=dict(total=float(total.detach()), tiny=float(tiny_loss.detach()), control=float(control_loss.detach())),
                     replay_errors=errors)
        path = output/"frames"/(stem+".npz")
        np.savez_compressed(path, total=gradients[0], tiny=gradients[1], control=gradients[2],
                            tiny_gt=tiny, control_gt=shuffled, classes=classes, productive_gt=active_gt)
        manifest[stem] = dict(file=path.relative_to(output).as_posix(), sha256=file_hash(path),
                             prior_frame_sha256=item["sha256"])
        traces.append(trace)
        del raw, captured["result"], total, tiny_loss, control_loss, rebuilt, bce, decoded, distances, logits, losses
        if index % 16 == 0:
            print(f"Scale-gradient audit: {index}/64", flush=True)
    hook.remove()
    assert model_digest(model) == original_digest
    metrics = summarize(traces)
    decision = decide(metrics, registration["policy"])
    verification = dict(image_decodes=64, frozen_model_forwards=64, derivative_queries=192, parameter_updates=0,
                        parameter_and_buffer_digest_before=original_digest, parameter_and_buffer_digest_after=model_digest(model),
                        AP_evaluations=0, original64_assignments_and_GT_replayed=True,
                        maximum_replay_errors=maximum,
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
