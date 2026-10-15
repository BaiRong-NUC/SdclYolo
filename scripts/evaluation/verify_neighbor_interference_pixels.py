"""Reconstruct all candidate67 interventions and protected input pixels without inference."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash
from sdcl.postprocess.rescue import save_json


def verify(output="output/analysis/neighbor-interference-fixed-20261015"):
    started = perf_counter()
    root = project_path(output)
    plan = json.loads((root/"plan.json").read_text())
    manifest = json.loads((root/"frames.json").read_text())
    cv2.setNumThreads(1)
    count = 0
    digest = lambda array: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    for item in plan["images"]:
        stem = item["stem"]
        path = project_path(f"data/visdrone/images/train/{stem}.jpg")
        assert file_hash(path) == item["image_sha256"]
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        ratio = 960/max(h, w)
        neww, newh = round(w*ratio), round(h*ratio)
        left, top = round((960-neww)/2-.1), round((960-newh)/2-.1)
        with np.load(root/manifest[stem]["file"], allow_pickle=False) as saved:
            gt = saved["gt"].copy()
            classes = saved["classes"].copy()
            boxes = saved["base_boxes"].copy()
            logits = saved["base_logits"].copy()
            queries = saved["queries"].copy()
        def target_tensor_hash(source, box):
            # Independently reconstruct predictor LetterBox/BGR->RGB/CHW/half normalization.
            resized = cv2.resize(source, (neww, newh), interpolation=cv2.INTER_LINEAR) if (neww, newh) != (w, h) else source
            padded = cv2.copyMakeBorder(resized, top, 960-newh-top, left, 960-neww-left,
                                       cv2.BORDER_CONSTANT, value=(114, 114, 114))
            x1, y1 = np.floor(box[:2]*ratio+[left, top]).astype(int).clip(0, 960)
            x2, y2 = np.ceil(box[2:]*ratio+[left, top]).astype(int).clip(0, 960)
            patch = padded[y1:y2, x1:x2, ::-1].transpose(2, 0, 1)
            return digest((patch.astype(np.float32)/255).astype(np.float16))
        for i, pair in enumerate(manifest[stem]["pairs"]):
            assert target_tensor_hash(image, gt[i]) == pair["target_tensor_sha256"]
            # Independent scalar baseline-query argmax, including geometry-only pool reconstruction.
            center = (gt[i, :2]+gt[i, 2:])/2*ratio+[left, top]
            cx, cy = np.rint(center/8-.5).astype(int)
            pool = sorted(y*120+x for y in range(max(0, cy-1), min(120, cy+2))
                          for x in range(max(0, cx-1), min(120, cx+2)))
            scores = []
            for query in pool:
                b, target = boxes[query].astype(float), gt[i]
                intersection = max(0, min(b[2], target[2])-max(b[0], target[0]))*max(
                    0, min(b[3], target[3])-max(b[1], target[1]))
                union = max(0, b[2]-b[0])*max(0, b[3]-b[1])+(target[2]-target[0])*(target[3]-target[1])-intersection
                scores.append((1/(1+np.exp(-float(logits[query, classes[i]]))))*intersection/max(union, 1e-12))
            assert pool[int(np.argmax(scores))] == queries[i]
            for entry in pair["perturbations"]:
                x1, y1, x2, y2 = entry["rectangle"]
                patch = image[y1:y2, x1:x2]
                altered = image.copy()
                if entry["arm"].endswith("_mean"):
                    value = np.rint(patch.astype(np.float64).mean(axis=(0, 1))).astype(np.uint8)
                else:
                    sigma = max(1, min(patch.shape[:2])/3)
                    value = cv2.GaussianBlur(patch, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
                altered[y1:y2, x1:x2] = value
                assert digest(altered) == entry["altered_image_sha256"]
                if not entry["arm"].startswith("self_"):
                    tx1, ty1, tx2, ty2 = pair["target_rect"]
                    np.testing.assert_array_equal(altered[ty1:ty2, tx1:tx2], image[ty1:ty2, tx1:tx2])
                    assert target_tensor_hash(altered, gt[i]) == entry["target_tensor_sha256"]
                count += 1
    report = dict(passed=True, images=len(plan["images"]), pairs=sum(len(e["pairs"]) for e in manifest.values()),
        reconstructed_perturbations=count, detector_forwards=0, optimizer_steps=0,
        elapsed_seconds=perf_counter()-started, verifier_sha256=file_hash(Path(__file__)),
        verified="Independent baseline 3x3 query argmax; all1830 perturbed image hashes; all protected original and manually preprocessed960 FP16 target pixels match actual execution hashes.",
        limits="OpenCV image decode/resize/Gaussian primitives are shared. No independent detector feature/logit re-inference.")
    save_json(root/"pixel_verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/analysis/neighbor-interference-fixed-20261015")
    print(json.dumps(verify(**vars(parser.parse_args())), indent=2))
