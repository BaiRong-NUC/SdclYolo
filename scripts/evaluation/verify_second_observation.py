"""Recheck source integrity, geometric correspondences and cached candidate68 metrics."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from sdcl.config import project_path
from sdcl.evaluation.second_observation import (
    OUTPUT, correspondences, extract_queries, measurements, mirror_boxes, project_boxes, summarize,
)
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.rescue import save_json


def verify(output=OUTPUT):
    root = project_path(output)
    plan = json.loads((root/"plan.json").read_text())
    manifest = json.loads((root/"frames.json").read_text())
    result = json.loads((root/"result.json").read_text())
    for name, sha in result["artifacts_sha256"].items():
        assert file_hash(root/name) == sha, name
    own_name = Path(__file__).resolve().relative_to(project_path(".").resolve()).as_posix()
    with zipfile.ZipFile(root/"source.zip") as archive:
        for name, sha in plan["source_hashes"].items():
            if name != own_name:
                assert file_hash(project_path(name)) == sha
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha
        for name, record in plan["installed_sources"].items():
            assert hashlib.sha256(archive.read("installed/"+name)).hexdigest() == record["sha256"]
    traces, accepted, targets = {"screen": [], "confirmation": []}, 0, 0
    assert len({p["group"] for p in plan["pairs"]}) == 64
    for pair in plan["pairs"]:
        entry = manifest[pair["group"]]
        assert file_hash(root/entry["file"]) == entry["sha256"]
        metas = []
        for stem, hashes in pair["inputs"].items():
            assert file_hash(project_path(f"data/visdrone/images/train/{stem}.jpg")) == hashes["image_sha256"]
            meta_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
            assert file_hash(meta_path) == hashes["metadata_sha256"]
            metas.append(json.loads(meta_path.read_text()))
        with np.load(root/entry["file"], allow_pickle=False) as saved:
            z = {k: saved[k].copy() for k in saved.files}
        measured = measurements(np.zeros((0, 3, 4)), np.zeros((0, 3, 10)), np.zeros((0, 4)), np.zeros(0, int))
        if entry["alignment"]["accepted"]:
            accepted += 1
            h = z["homography"]
            # Homogeneous coordinate check is independent of cv2.perspectiveTransform.
            points = np.column_stack((z["source_points"], np.ones(len(z["source_points"]))))@h.T
            projected = points[:, :2]/points[:, 2:]
            residual = np.linalg.norm(projected-z["current_points"], axis=1)*z["current_resize_ratio"]
            np.testing.assert_allclose(np.median(residual[z["inliers"]]),
                                       entry["alignment"]["median_resized_error"], atol=1e-6)
            matches, overlap = correspondences(*metas, h)
            order = sorted(range(len(matches)), key=lambda i: hashlib.sha256(
                f"candidate68-object-v1:{pair['current']}:{matches[i, 0]}".encode()).hexdigest())[:12]
            np.testing.assert_array_equal(matches[order], z["matched"])
            np.testing.assert_array_equal(overlap, z["correspondence_iou"])
            gt_sets = (z["gt"], mirror_boxes(z["gt"], metas[0]["original_size"][0]), z["source_gt"])
            boxes, logits = [], []
            for i, name in enumerate(("current", "flip", "source")):
                size = metas[1 if i == 2 else 0]["original_size"]
                query = extract_queries(z[name+"_boxes"], z[name+"_logits"], gt_sets[i], z["classes"], size)
                np.testing.assert_array_equal(query, z["queries"][:, i])
                # Main inference stacks native float32 boxes into float64 BEFORE unmirroring.
                selected = z[name+"_boxes"][query].astype(float)
                if i == 1:
                    selected = mirror_boxes(selected, size[0])
                elif i == 2:
                    selected = project_boxes(selected, h)
                boxes.append(selected)
                logits.append(z[name+"_logits"][query])
            np.testing.assert_array_equal(np.stack(boxes, 1), z["boxes"])
            np.testing.assert_array_equal(np.stack(logits, 1), z["logits"])
            measured = measurements(z["boxes"], z["logits"], z["gt"], z["classes"])
            for key, value in measured.items():
                np.testing.assert_array_equal(value, z[key])
            targets += len(z["gt"])
        traces[pair["split"]].append(measured)
    metrics, decision = summarize(traces, plan["registration"]["policy_per_half"])
    assert metrics == result["metrics"] and decision == result["decision"]
    detector = json.loads((root/"detector.json").read_text())
    assert detector["state_before"] == detector["state_after"]
    assert file_hash(detector["checkpoint"]["path"]) == detector["checkpoint"]["sha256"]
    report = dict(passed=True, pairs=64, accepted_alignments=accepted, targets=targets,
        decision=decision["decision"], detector_forwards=0,
        verified="Input/source/archive/cache hashes;independent homogeneous reprojection;shared correspondence/query reconstruction;cached metrics/bootstrap/gates;unchanged detector state.",
        limits="No independent SIFT/RANSAC or detector rerun;associations are GT/spatial proxies,not verified object tracks.",
        verifier_revision=dict(archived_sha256=plan["source_hashes"][own_name],
            current_sha256=file_hash(Path(__file__)),
            reason="Initial verifier unmirrored native float32 before casting;main used float64 before unmirroring. Correct verifier arithmetic to match main exactly;no tolerance relaxation or metric/data/gate changes. Original verifier remains in source.zip and initial-failure log."))
    save_json(root/"verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    print(json.dumps(verify(**vars(parser.parse_args())), indent=2))
