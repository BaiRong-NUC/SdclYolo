"""Rebuild candidate52 alignment, teacher rules and conditional metrics from logits."""

import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash


OUTPUT = "output/analysis/crop-teacher-fixed-20261014"
ARMS = ("student", "teacher1", "confidence", "consensus")
LIMITS = {"all": math.inf, "tiny": 256., "small_all": 1024.}


def compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for k in expected:
            compare(actual[k], expected[k])
    elif expected is None or isinstance(expected, str):
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


def classification(logits, label):
    values = [float(x) for x in logits]
    maximum = max(values)
    exponential = [math.exp(x-maximum) for x in values]
    total = math.fsum(exponential)
    return maximum+math.log(total)-values[label], values.index(maximum), 1/total


def verify():
    output = project_path(OUTPUT)
    read = lambda n: json.loads((output/n).read_text())
    result, plan, manifest, saved_traces = read("result.json"), read("plan.json"), read("frames.json"), read("per_image.json")
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(output/name) == digest
    with zipfile.ZipFile(output/"source.zip") as z:
        for name, digest in plan["source_hashes"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
            assert file_hash(project_path(name)) == digest
        for name, item in plan["installed_sources"].items():
            assert hashlib.sha256(z.read("installed/"+name)).hexdigest() == item["sha256"]
            assert file_hash(Path(item["path"])) == item["sha256"]
    prior = project_path("output/analysis/assignment-reachability-fixed-20261014")
    assert file_hash(prior/"plan.json") == plan["prior_plan_sha256"]
    assert file_hash(prior/"frames.json") == plan["prior_frames_sha256"]
    previous = json.loads((prior/"plan.json").read_text())
    assert previous["images"] == plan["images"] and len(plan["images"]) == 64
    assert file_hash(project_path(previous["checkpoint"])) == plan["checkpoint_sha256"]
    traces, checked_locations, all_gt = [], 0, 0
    for index, stem in enumerate(plan["images"]):
        item = manifest[stem]
        path = output/item["file"]
        assert file_hash(path) == item["sha256"]
        assert file_hash(project_path(f"data/visdrone/images/train/{stem}.jpg")) == item["image_sha256"]
        metadata_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
        assert file_hash(metadata_path) == item["metadata_sha256"]
        metadata = json.loads(metadata_path.read_text())
        with np.load(path, allow_pickle=False) as z:
            f = {k: z[k].copy() for k in z.files}
        gt = np.asarray([o["box_xyxy"] for o in metadata["valid_objects"]], float)
        classes = [int(o["class_id"]) for o in metadata["valid_objects"]]
        np.testing.assert_array_equal(gt, f["gt"])
        np.testing.assert_array_equal(classes, f["classes"])
        width, height = metadata["original_size"]
        tw, th = min(width, math.ceil(width/1.8)), min(height, math.ceil(height/1.8))
        windows = [(0, 0, width, height), *((x, y, x+tw, y+th)
                   for y in (0, height-th) for x in (0, width-tw))]
        np.testing.assert_array_equal(windows, f["windows"])
        assert f["source_logits"].shape == (5, 14400, 10)
        assert np.isfinite(f["source_logits"]).all()
        native, available, margins = [], [], []
        teachers, active, areas = [], [], []
        sampled = []
        for row, box in enumerate(gt):
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            locs, flags, distances, logits = [], [], [], []
            for view, (x1, y1, x2, y2) in enumerate(windows):
                w, h = x2-x1, y2-y1
                ratio = min(960/w, 960/h)
                px, py = round((960-round(w*ratio))/2-.1), round((960-round(h*ratio))/2-.1)
                ix = min(119, max(0, math.floor(((cx-x1)*ratio+px)/8)))
                iy = min(119, max(0, math.floor(((cy-y1)*ratio+py)/8)))
                locs.append(iy*120+ix)
                flags.append(view > 0 and box[0] >= x1 and box[1] >= y1 and box[2] <= x2 and box[3] <= y2)
                distances.append(min((box[0]-x1)/w, (box[1]-y1)/h, (x2-box[2])/w, (y2-box[3])/h))
                logits.append(f["source_logits"][view, locs[-1]])
                checked_locations += 1
            chosen = sorted([v for v in range(5) if flags[v]], key=lambda v: (-distances[v], v))[:2]
            teachers.append((chosen+[0, 0])[:2])
            active.append(sum(flags) >= 2)
            native.append(locs)
            available.append(flags)
            margins.append(distances)
            sampled.append(logits)
            areas.append(float((box[2]-box[0])*(box[3]-box[1])*(640/max(width, height))**2))
        for key, value in dict(native_indices=native, available=available, teachers=teachers,
                               active=active, sampled_logits=sampled).items():
            np.testing.assert_array_equal(value, f[key])
        np.testing.assert_allclose(margins, f["margins"], atol=1e-12)
        np.testing.assert_allclose(areas, f["areas"], atol=1e-12)
        trace = {g: dict(rows=0, images=0, NLL_sum=dict.fromkeys(ARMS, 0.),
                        correct=dict.fromkeys(ARMS, 0), teacher1_recovered=0, teacher1_lost=0,
                        action_rows=dict.fromkeys(ARMS, 0)) for g in LIMITS}
        for row, label in enumerate(classes):
            student = np.asarray(sampled[row][0], float)
            first, second = (np.asarray(sampled[row][v], float) for v in teachers[row])
            s, t1, t2 = (classification(x, label) for x in (student, first, second))
            masks = dict(student=False, teacher1=active[row],
                         confidence=active[row] and t1[2] > s[2],
                         consensus=active[row] and t1[1] == t2[1])
            for arm in ARMS:
                np.testing.assert_array_equal(first if masks[arm] else student, f[arm][row])
            if not active[row]:
                continue
            for group, limit in LIMITS.items():
                if areas[row] >= limit:
                    continue
                values = trace[group]
                values["rows"] += 1
                values["images"] = 1
                values["teacher1_recovered"] += s[1] != label and t1[1] == label
                values["teacher1_lost"] += s[1] == label and t1[1] != label
                for arm in ARMS:
                    nll, predicted, _ = t1 if masks[arm] else s
                    values["NLL_sum"][arm] += nll
                    values["correct"][arm] += predicted == label
                    values["action_rows"][arm] += masks[arm]
        compare(trace, saved_traces[index])
        traces.append(trace)
        all_gt += len(gt)
    metrics = {}
    for group in LIMITS:
        count = sum(t[group]["rows"] for t in traces)
        metrics[group] = dict(rows=count, images=sum(t[group]["images"] for t in traces),
            NLL={a: math.fsum(t[group]["NLL_sum"][a] for t in traces)/count for a in ARMS},
            top1_percent={a: 100*sum(t[group]["correct"][a] for t in traces)/count for a in ARMS},
            teacher1_recovered=sum(t[group]["teacher1_recovered"] for t in traces),
            teacher1_lost=sum(t[group]["teacher1_lost"] for t in traces),
            action_rows={a: sum(t[group]["action_rows"][a] for t in traces) for a in ARMS})
    counts = [t["tiny"]["rows"] for t in traces]
    gains = [t["tiny"]["NLL_sum"]["confidence"]-t["tiny"]["NLL_sum"]["consensus"] for t in traces]
    weights = np.random.default_rng(0).multinomial(64, np.full(64, 1/64), size=400)
    samples = [math.fsum(int(w)*g for w, g in zip(row, gains))/sum(int(w)*n for w, n in zip(row, counts)) for row in weights]
    metrics["tiny"]["NLL_gain_vs_confidence_image_bootstrap95ci"] = np.percentile(samples, [2.5, 97.5]).tolist()
    compare(metrics, read("metrics.json"))
    policy, tiny = plan["registration"]["policy"], metrics["tiny"]
    checks = dict(rows=tiny["rows"] >= policy["tiny_primary_rows_min"],
                  images=tiny["images"] >= policy["tiny_primary_images_min"],
                  recovered=tiny["teacher1_recovered"] >= policy["tiny_teacher1_recovered_min"],
                  lost=tiny["teacher1_lost"] >= policy["tiny_teacher1_lost_min"],
                  NLL_vs_student=tiny["NLL"]["student"]-tiny["NLL"]["consensus"] >= policy["tiny_consensus_NLL_gain_vs_student_min"],
                  NLL_vs_confidence=tiny["NLL"]["confidence"]-tiny["NLL"]["consensus"] >= policy["tiny_consensus_NLL_gain_vs_confidence_min"],
                  top1=tiny["top1_percent"]["consensus"]-tiny["top1_percent"]["student"] >= policy["tiny_consensus_top1_gain_vs_student_min_pp"],
                  interval=tiny["NLL_gain_vs_confidence_image_bootstrap95ci"][0] > policy["tiny_consensus_NLL_gain_vs_confidence_bootstrap_lower_min"],
                  all_NLL=metrics["all"]["NLL"]["student"]-metrics["all"]["NLL"]["consensus"] >= policy["all_consensus_NLL_gain_vs_student_min"])
    assert checks == result["decision"]["checks"] and not all(checks.values())
    assert result["decision"]["decision"] == "STOP_THIS_TWO_FULL_CROP_CONSENSUS_TEACHER_RECIPE"
    assert all_gt == result["verification"]["all_gt"] == 3948
    assert metrics["all"]["rows"] == result["verification"]["primary_gt"] == 985
    verified = dict(all_archived_and_current_sources_checkpoint_prior_selection_image_metadata_and64_frames_hashed=True,
                    full_metadata_GT_and_five_windows_rebuilt_without_detector_forward=True,
                    checked_native_sample_locations=checked_locations,
                    all_full_containment_margins_teacher_order_and_actions_independently_rebuilt=True,
                    all_classification_NLL_top1_recoveries_losses_and_action_counts_recomputed_with_scalar_math=True,
                    all400_image_bootstrap_samples_and_nine_gates_recomputed=True,
                    extra_image_decodes=0, extra_detector_forwards=0,
                    verifier_sha256=file_hash(Path(__file__)),
                    limits="Complete saved P3 tensors are hashed and sampled independently but not re-extracted. Conditional labeled-object center classification is not whole-detector distillation, AP, background handling or novelty proof.")
    (output/"independent_verification.json").write_text(json.dumps(verified, indent=2)+"\n", encoding="utf-8")
    return verified


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
