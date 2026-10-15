"""Recompute cached candidate67 measurements and independently audit intervention geometry."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.special import expit

from sdcl.config import project_path
from sdcl.evaluation.neighbor_interference import OUTPUT, aggregate, frame_metrics
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.rescue import save_json


def verify(output=OUTPUT):
    root = project_path(output)
    plan = json.loads((root/"plan.json").read_text())
    result = json.loads((root/"result.json").read_text())
    manifest = json.loads((root/"frames.json").read_text())
    for name, sha in result["artifacts_sha256"].items():
        assert file_hash(root/name) == sha, name
    with zipfile.ZipFile(root/"source.zip") as archive:
        for name, sha in plan["source_hashes"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha
            assert file_hash(project_path(name)) == sha, name
        for name, entry in plan["installed_sources"].items():
            assert hashlib.sha256(archive.read("installed/"+name)).hexdigest() == entry["sha256"]
    sequences, traces, count = set(), {"screen": [], "confirmation": []}, 0
    def overlap(a, b):
        return min(a[2], b[2]+2) > max(a[0], b[0]-2) and min(a[3], b[3]+2) > max(a[1], b[1]-2)
    for item in plan["images"]:
        stem = item["stem"]
        assert item["sequence"] not in sequences
        sequences.add(item["sequence"])
        meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
        assert file_hash(meta_path) == item["metadata_sha256"]
        assert file_hash(project_path(f"data/visdrone/images/train/{stem}.jpg")) == item["image_sha256"]
        metadata = json.loads(meta_path.read_text())
        gt_all = [o["box_xyxy"] for o in metadata["valid_objects"]]
        entry = manifest[stem]
        assert file_hash(root/entry["file"]) == entry["sha256"]
        with np.load(root/entry["file"], allow_pickle=False) as z:
            values = {k: z[k].copy() for k in z.files}
        computed = frame_metrics(values["boxes"], values["logits"], values["gt"],
                                 values["classes"], values["neck"], values["features"])
        for name, value in computed.items():
            np.testing.assert_array_equal(value, values[name])
        # Scalar IoU implementation does not reuse the project's box-IoU routine.
        for i, pair in enumerate(entry["pairs"]):
            target, neighbor = pair["target"], pair["neighbor"]
            assert target != neighbor
            other = [b for j, b in enumerate(gt_all) if j != neighbor]+metadata["ignore_boxes_xyxy"]
            assert not any(overlap(pair["neighbor_rect"], b) for b in other)
            assert not any(overlap(pair["sham_rect"], b) for b in gt_all+metadata["ignore_boxes_xyxy"])
            nr, sr = np.array(pair["neighbor_rect"]), np.array(pair["sham_rect"])
            np.testing.assert_array_equal(nr[2:]-nr[:2], sr[2:]-sr[:2])
            center = (np.array(gt_all[target][:2])+gt_all[target][2:])/2
            assert abs(np.linalg.norm((nr[:2]+nr[2:])/2-center)-np.linalg.norm((sr[:2]+sr[2:])/2-center)) <= 1
            for perturbation in pair["perturbations"][:4]:
                assert perturbation["target_tensor_sha256"] == pair["target_tensor_sha256"]
            query = int(values["queries"][i])
            np.testing.assert_array_equal(values["boxes"][i, 0], values["base_boxes"][query])
            np.testing.assert_array_equal(values["logits"][i, 0], values["base_logits"][query])
            for a, box in enumerate(values["boxes"][i]):
                target_box = values["gt"][i]
                intersection = max(0, min(box[2], target_box[2])-max(box[0], target_box[0]))*max(
                    0, min(box[3], target_box[3])-max(box[1], target_box[1]))
                area = max(0, box[2]-box[0])*max(0, box[3]-box[1])
                target_area = (target_box[2]-target_box[0])*(target_box[3]-target_box[1])
                iou = intersection/max(float(area+target_area-intersection), 1e-12)
                np.testing.assert_allclose(iou, values["iou"][i, a], atol=1e-7, rtol=1e-6)
                q = float(expit(float(values["logits"][i, a, values["classes"][i]])))*iou
                np.testing.assert_allclose(q, values["Q"][i, a], atol=1e-7, rtol=1e-6)
        traces[item["split"]].append(dict(stem=stem, same_class=values["same_class"], **computed))
        count += len(values["gt"])
    metrics, decision = aggregate(traces, plan["registration"]["policy_per_half"])
    assert metrics == result["metrics"]
    assert decision == result["decision"]
    detector = json.loads((root/"detector.json").read_text())
    assert detector["state_before"] == detector["state_after"]
    assert file_hash(detector["checkpoint"]["path"]) == detector["checkpoint"]["sha256"]
    report = dict(passed=True, images=len(sequences), pairs=count, decision=decision["decision"],
        verified="Saved input/metadata/source hashes, archived sources, protected geometry and tensor hashes; scalar independent IoU/Q; shared metrics/bootstrap/gates; detector state unchanged.",
        limits="No independent re-inference or pixel/tensor recapture. Saved target-tensor hashes are execution attestations. Feature extraction and bootstrap engine shared.")
    save_json(root/"verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    print(json.dumps(verify(**vars(parser.parse_args())), indent=2))
