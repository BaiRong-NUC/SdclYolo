"""Independently audit saved gradient statistics and replay three gradient traces."""

import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.tal import make_anchors
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import SDCLConfig, project_path
from sdcl.data.dataset import IgnoreDataset
from sdcl.data.geometry import anchor_ignore_mask
from sdcl.evaluation.size_analysis import file_hash
from sdcl.training.criterion import SDCLLoss


OUTPUT = "output/analysis/scale-gradient-conflict-fixed-20261014"


def statistics(total, tiny, control):
    vectors = [np.asarray(v, float) for v in (tiny, total-tiny, control, total-control)]
    norm = lambda v: math.sqrt(math.fsum(float(x)*float(x) for x in v))
    norms = [norm(v) for v in vectors]
    if min(norms) <= 1e-12:
        return dict(valid=False, norms=norms)
    arms = []
    for a, b in (vectors[:2], vectors[2:]):
        dot = math.fsum(float(x)*float(y) for x, y in zip(a, b))
        arms.append(dict(cosine=dot/(norm(a)*norm(b)),
                         opposition_ratio=max(0., -dot)/(norm(a)**2),
                         blocked=math.fsum(float(x)*float(y) for x, y in zip(a, total)) < 0,
                         tiny_norm=norm(a), rest_norm=norm(b)))
    return dict(valid=True, actual=arms[0], control=arms[1],
                control_minus_actual_cosine=arms[1]["cosine"]-arms[0]["cosine"])


def compare(a, b):
    if isinstance(b, dict):
        assert a.keys() == b.keys()
        for k in b:
            compare(a[k], b[k])
    elif b is None or isinstance(b, str):
        assert a == b
    else:
        np.testing.assert_allclose(a, b, atol=1e-11, rtol=1e-10)


def verify():
    torch.set_num_threads(1)
    output = project_path(OUTPUT)
    read = lambda n: json.loads((output/n).read_text())
    result, plan, manifest, traces = read("result.json"), read("plan.json"), read("frames.json"), read("per_image.json")
    for n, h in result["artifacts_sha256"].items():
        assert file_hash(output/n) == h
    with zipfile.ZipFile(output/"source.zip") as z:
        for n, h in plan["source_hashes"].items():
            assert hashlib.sha256(z.read(n)).hexdigest() == h and file_hash(project_path(n)) == h
        for n, item in plan["installed_sources"].items():
            assert hashlib.sha256(z.read("installed/"+n)).hexdigest() == item["sha256"]
            assert file_hash(Path(item["path"])) == item["sha256"]
    prior = project_path("output/analysis/assignment-reachability-fixed-20261014")
    assert file_hash(prior/"plan.json") == plan["prior_plan_sha256"]
    assert file_hash(prior/"result.json") == plan["prior_result_sha256"]
    prior_plan = json.loads((prior/"plan.json").read_text())
    prior_manifest = json.loads((prior/"frames.json").read_text())
    assert plan["images"] == prior_plan["images"]
    frames, rebuilt, changed_images, changed_gt = {}, [], 0, 0
    for trace, stem in zip(traces, plan["images"]):
        assert trace["stem"] == stem
        item = manifest[stem]
        assert file_hash(output/item["file"]) == item["sha256"]
        assert file_hash(prior/prior_manifest[stem]["file"]) == item["prior_frame_sha256"]
        with np.load(output/item["file"]) as z:
            f = {k: z[k].copy() for k in z.files}
        with np.load(prior/prior_manifest[stem]["file"]) as z:
            gt, classes, owner, mass, fg = (z[k].copy() for k in ("gt", "classes", "owner", "target_mass", "foreground"))
        tiny = (gt[:, 2:]-gt[:, :2]).prod(1)*(640/960)**2 < 256
        np.testing.assert_array_equal(tiny, f["tiny_gt"])
        np.testing.assert_array_equal(classes, f["classes"])
        control = tiny.copy()
        for cls in sorted(set(classes.astype(int))):
            ids = [i for i, c in enumerate(classes) if int(c) == cls]
            seed = int.from_bytes(hashlib.sha256(f"candidate51-scale-control-v1:{stem}:{cls}".encode()).digest()[:8], "little")
            permutation = np.random.default_rng(seed).permutation(np.array([tiny[i] for i in ids]))
            control[ids] = permutation
            assert control[ids].sum() == tiny[ids].sum()
        np.testing.assert_array_equal(control, f["control_gt"])
        productive_gt = np.unique(owner[fg & (mass > 0)])
        np.testing.assert_array_equal(productive_gt, f["productive_gt"])
        changed = int((tiny[productive_gt] != control[productive_gt]).sum())
        assert changed == trace["changed_productive_gt"]
        changed_images += changed > 0
        changed_gt += changed
        values = statistics(f["total"].astype(float), f["tiny"].astype(float), f["control"].astype(float))
        compare(values, trace["gradients"])
        rebuilt.append(values)
        frames[stem] = f
    valid = [s for s in rebuilt if s["valid"]]
    effects = [s["control_minus_actual_cosine"] for s in valid]
    weights = np.random.default_rng(0).multinomial(len(valid), np.full(len(valid), 1/len(valid)), size=400)
    bootstrap = [math.fsum(int(w)*e for w, e in zip(row, effects))/len(valid) for row in weights]
    metrics = dict(images=len(rebuilt), valid_images=len(valid), changed_control_images=changed_images,
                   changed_productive_gt=changed_gt,
                   actual_blocked_fraction=sum(s["actual"]["blocked"] for s in valid)/len(valid),
                   control_blocked_fraction=sum(s["control"]["blocked"] for s in valid)/len(valid),
                   actual_mean_cosine=math.fsum(s["actual"]["cosine"] for s in valid)/len(valid),
                   control_mean_cosine=math.fsum(s["control"]["cosine"] for s in valid)/len(valid),
                   mean_control_minus_actual_cosine=math.fsum(effects)/len(valid),
                   image_bootstrap95ci_control_minus_actual_cosine=np.percentile(bootstrap, [2.5, 97.5]).tolist())
    compare(metrics, read("metrics.json"))
    policy = plan["registration"]["policy"]
    checks = dict(valid_images=len(valid) >= policy["valid_images_min"],
                  changed_images=changed_images >= policy["changed_control_images_min"],
                  changed_gt=changed_gt >= policy["changed_productive_gt_min"],
                  actual_blocking=metrics["actual_blocked_fraction"] >= policy["actual_blocked_fraction_min"],
                  excess_conflict=metrics["mean_control_minus_actual_cosine"] >= policy["mean_control_minus_actual_cosine_min"],
                  interval=metrics["image_bootstrap95ci_control_minus_actual_cosine"][0] > policy["bootstrap_lower_control_minus_actual_cosine_min"])
    assert checks == result["decision"]["checks"] and not all(checks.values())
    assert result["decision"]["decision"] == "STOP_THIS_SCALE_SPECIFIC_P3_GRADIENT_ROUTE"
    selected = [t["stem"] for t in traces if t["gradients"]["valid"]][:3]
    args = get_cfg(overrides={"imgsz": 960, "batch": 1, "workers": 0, "rect": False, "plots": False, "device": "0"})
    data_path = project_path("data/visdrone/dataset.yaml")
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    data.update(nc=10, channels=3, path=str(data_path.parent))
    dataset = IgnoreDataset(img_path=str(data_path.parent/"images/train"), imgsz=960, batch_size=1,
                            augment=False, hyp=args, rect=False, cache=False, single_cls=False,
                            stride=32, pad=0., prefix="gradient-replay: ", task="detect", data=data)
    indices = {Path(p).stem: i for i, p in enumerate(dataset.im_files)}
    checkpoint = project_path(prior_plan["checkpoint"])
    assert file_hash(checkpoint) == plan["checkpoint_sha256"]
    model = YOLO(str(checkpoint)).model.float().cuda().eval()
    for name, p in model.named_parameters():
        p.requires_grad_(name == plan["parameter"])
    parameter = dict(model.named_parameters())[plan["parameter"]]
    model.args, model.sdcl_settings = args, SDCLConfig(enabled=False).to_dict()
    model.sdcl_ignore_regions, model.coverage_settings = True, {"enabled": False}
    criterion = SDCLLoss(model)
    maximum, maximum_relative_norm, replay = 0., 0., []
    for stem in selected:
        batch = dataset.collate_fn([dataset[indices[stem]]])
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch["img"] = batch["img"].float()/255
        raw = model(batch["img"])[1]
        anchors, stride = make_anchors(raw["feats"], criterion.stride, .5)
        logits = raw["scores"].permute(0, 2, 1).contiguous()
        distances = raw["boxes"].permute(0, 2, 1).contiguous()
        decoded = criterion.bbox_decode(anchors, distances)
        targets = criterion.preprocess(torch.cat((batch["batch_idx"][:, None], batch["cls"][:, :1], batch["bboxes"]), 1), 1, logits.new_tensor([960]*4))
        gt = targets[..., 1:]
        _, boxes, scores, fg, owner = criterion.assigner(logits.detach().sigmoid(), decoded.detach()*stride,
                                                        anchors*stride, targets[..., :1], gt, gt.sum(-1, keepdim=True) > 0)
        denominator = scores.sum().clamp_min(1)
        ignored = anchor_ignore_mask(anchors*stride, batch["ignore_bboxes"], batch["ignore_batch_idx"], 1, (960, 960))
        bce = F.binary_cross_entropy_with_logits(logits, scores, reduction="none").masked_fill((ignored & ~fg)[..., None], 0)
        def partition_loss(rows, class_rows):
            iou, dfl = criterion.bbox_loss(distances, decoded, anchors, boxes/stride, scores,
                                           denominator, rows, logits.new_tensor([960, 960]), stride)
            return iou*args.box+dfl*args.dfl+bce[class_rows].sum()/denominator*args.cls
        total = partition_loss(fg, torch.ones_like(fg))
        trace = next(t for t in traces if t["stem"] == stem)
        np.testing.assert_allclose(float(total.detach()), trace["losses"]["total"], atol=1e-5, rtol=5e-5)
        for arm in ("tiny", "control"):
            mask = fg & (scores.sum(-1) > 0) & torch.as_tensor(frames[stem][arm+"_gt"], device="cuda")[owner]
            # Directly differentiate the complement, not total-minus-component.
            remainder = partition_loss(fg & ~mask, ~mask)
            rest_grad = torch.autograd.grad(remainder, parameter, retain_graph=True)[0].detach().cpu().numpy().reshape(-1)
            expected = frames[stem]["total"]-frames[stem][arm]
            component = partition_loss(mask, mask)
            np.testing.assert_allclose(float(component.detach()), trace["losses"][arm], atol=1e-5, rtol=5e-5)
            component_grad = torch.autograd.grad(component, parameter, retain_graph=True)[0].detach().cpu().numpy().reshape(-1)
            direct_error = float(np.linalg.norm(component_grad-frames[stem][arm])/np.linalg.norm(frames[stem][arm]))
            relative_error = float(np.linalg.norm(rest_grad-expected)/np.linalg.norm(expected))
            print(stem, arm, "component relative error", direct_error, "complement relative error", relative_error, flush=True)
            assert np.isfinite(component_grad).all() and np.isfinite(rest_grad).all()
            replay.append(dict(stem=stem, arm=arm, component_relative_L2_error=direct_error,
                               complement_relative_L2_error=relative_error,
                               complement_meets_initial_elementwise_audit_tolerance=bool(np.all(
                                   np.abs(rest_grad-expected) <= 3e-6+5e-4*np.abs(expected)))))
            maximum_relative_norm = max(maximum_relative_norm, relative_error, direct_error)
            maximum = max(maximum, float(np.max(np.abs(rest_grad-expected))))
        total_grad = torch.autograd.grad(total, parameter)[0].detach().cpu().numpy().reshape(-1)
        relative_error = float(np.linalg.norm(total_grad-frames[stem]["total"])/np.linalg.norm(frames[stem]["total"]))
        assert np.isfinite(total_grad).all()
        replay.append(dict(stem=stem, arm="total", relative_L2_error=relative_error,
                           meets_initial_elementwise_audit_tolerance=bool(np.all(
                               np.abs(total_grad-frames[stem]["total"]) <= 3e-6+5e-4*np.abs(frames[stem]["total"])))))
        maximum_relative_norm = max(maximum_relative_norm, relative_error)
        maximum = max(maximum, float(np.max(np.abs(total_grad-frames[stem]["total"]))))
    verified = dict(all_sources_installed_sources_prior_records_and64_gradient_files_hashed=True,
                    all64_GT_memberships_class_preserving_shuffles_and_productive_GT_rebuilt=True,
                    all64_gradient_norms_cosines_blocking_and_metrics_independently_recomputed=True,
                    all400_bootstrap_draws_and_six_gates_recomputed=True,
                    extra_verification_image_decodes=3, extra_frozen_forwards=3, extra_derivative_queries=15,
                    independently_replayed_images=selected,
                    maximum_direct_complement_gradient_absolute_error=maximum,
                    maximum_replay_gradient_relative_L2_error=maximum_relative_norm,
                    gradient_replay_audit_status="DIAGNOSTIC_ONLY_NO_TOLERANCE_RELAXATION",
                    gradient_replay_diagnostics=replay,
                    verifier_sha256=file_hash(Path(__file__)),
                    limits="Saved-statistic audit passes, but independent complement derivatives exceed the initial3e-6+5e-4relative elementwise tolerance. Do not call full gradient replay verified or change thresholds to rescue it; discrepancy origin not fully resolved. Covers3of64 frames with same pinned primitives, other61 vectors only hash/statistic-verified. No AP/update/generalization proof.")
    (output/"independent_verification.json").write_text(json.dumps(verified, indent=2)+"\n", encoding="utf-8")
    return verified


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
