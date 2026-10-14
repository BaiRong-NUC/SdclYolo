"""Independently rebuild candidate53 derivatives and the frozen decision."""

import hashlib
from itertools import product
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash


OUTPUT = "output/analysis/label-gradient-fixed-20261014"


class Dual:
    """NumPy forward derivatives with four independent pixel coordinates."""

    def __init__(self, value, derivative=None):
        self.v = np.asarray(value, dtype=float)
        self.d = np.zeros(self.v.shape+(4,)) if derivative is None else derivative

    @staticmethod
    def cast(other):
        return other if isinstance(other, Dual) else Dual(other)

    def __add__(self, other):
        other = self.cast(other)
        return Dual(self.v+other.v, self.d+other.d)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, -self.d)

    def __sub__(self, other):
        return self+-self.cast(other)

    def __rsub__(self, other):
        return self.cast(other)+-self

    def __mul__(self, other):
        other = self.cast(other)
        return Dual(self.v*other.v, self.d*other.v[..., None]+other.d*self.v[..., None])

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.cast(other)
        return Dual(self.v/other.v, (self.d*other.v[..., None]-other.d*self.v[..., None])/other.v[..., None]**2)

    def square(self):
        return self*self

    def atan(self):
        return Dual(np.arctan(self.v), self.d/(1+self.v**2)[..., None])

    def positive(self):
        return Dual(np.maximum(self.v, 0), self.d*(self.v >= 0)[..., None])


def extreme(a, b, maximum):
    # torch.minimum/maximum split equal-value derivatives equally.
    weight = (a.v > b.v).astype(float) if maximum else (a.v < b.v).astype(float)
    weight += .5*(a.v == b.v)
    value = np.maximum(a.v, b.v) if maximum else np.minimum(a.v, b.v)
    return Dual(value, weight[..., None]*a.d+(1-weight[..., None])*b.d)


def independent_gradient(prediction, target, stride):
    stride = stride.reshape(-1)
    p = [Dual(prediction[:, k]/stride, np.broadcast_to(np.eye(4)[k], (len(stride), 4))/stride[:, None])
         for k in range(4)]
    t = [Dual(target[:, k]/stride) for k in range(4)]
    w, h = p[2]-p[0], p[3]-p[1]+1e-7
    tw, th = t[2]-t[0], t[3]-t[1]+1e-7
    inter = ((extreme(p[2], t[2], False)-extreme(p[0], t[0], True)).positive() *
             (extreme(p[3], t[3], False)-extreme(p[1], t[1], True)).positive())
    iou = inter/(w*h+tw*th-inter+1e-7)
    cw = extreme(p[2], t[2], True)-extreme(p[0], t[0], False)
    ch = extreme(p[3], t[3], True)-extreme(p[1], t[1], False)
    distance = ((t[0]+t[2]-p[0]-p[2]).square()+(t[1]+t[3]-p[1]-p[3]).square())/4
    v = (4/math.pi**2)*((tw/th).atan()-(w/h).atan()).square()
    alpha = Dual(v.v/(1-iou.v+v.v+1e-7))
    return (1-iou+distance/(cw.square()+ch.square()+1e-7)+v*alpha).d


def compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, (str, bool)) or expected is None:
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


def verify():
    output = project_path(OUTPUT)
    read = lambda path: json.loads(path.read_text())
    result, plan = read(output/"result.json"), read(output/"plan.json")
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(output/name) == digest
    with zipfile.ZipFile(output/"source.zip") as z:
        for name, digest in plan["source_hashes"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
            assert file_hash(project_path(name)) == digest
        installed = plan["installed_source"]
        assert hashlib.sha256(z.read("installed/metrics.py")).hexdigest() == installed["sha256"]
        assert file_hash(Path(installed["path"])) == installed["sha256"]
    prior = project_path("output/analysis/assignment-reachability-fixed-20261014")
    assert file_hash(prior/"plan.json") == plan["prior_plan_sha256"]
    assert file_hash(prior/"result.json") == plan["prior_result_sha256"]
    prior_plan, prior_result = read(prior/"plan.json"), read(prior/"result.json")
    for name, digest in prior_result["artifacts_sha256"].items():
        assert file_hash(prior/name) == digest
    assert plan["images"] == prior_plan["images"] and len(plan["images"]) == 64
    assert file_hash(project_path(prior_plan["checkpoint"])) == prior_plan["checkpoint_sha256"]
    prior_frames, frames = read(prior/"frames.json"), read(output/"frames.json")
    signs = np.array(list(product((-1., 1.), repeat=4)))
    np.testing.assert_array_equal(signs, plan["signs"])
    traces, checked, maximum_error = [], 0, 0.
    for stem in plan["images"]:
        old, new = prior_frames[stem], frames[stem]
        assert file_hash(prior/old["file"]) == old["sha256"] == new["prior_frame_sha256"]
        assert file_hash(output/new["file"]) == new["sha256"]
        metadata_path = project_path(f"data/visdrone/metadata/train/{stem}.json")
        assert file_hash(metadata_path) == old["metadata_sha256"] == new["metadata_sha256"]
        assert file_hash(project_path(f"data/visdrone/images/train/{stem}.jpg")) == old["image_sha256"]
        assert file_hash(project_path(f"data/visdrone/labels/train/{stem}.txt")) == old["label_sha256"]
        metadata = read(metadata_path)
        with np.load(prior/old["file"], allow_pickle=False) as z:
            old_arrays = {k: z[k].copy() for k in z.files}
        with np.load(output/new["file"], allow_pickle=False) as z:
            saved = {k: z[k].copy() for k in z.files}
        gt = old_arrays["gt"].astype(float)
        width, height = metadata["original_size"]
        scale = 960/max(width, height)
        rw, rh = min(math.ceil(width*scale), 960), min(math.ceil(height*scale), 960)
        left, top = round((960-rw)/2-.1), round((960-rh)/2-.1)
        absolute = np.repeat([[rw/width/2, rh/height/2]*2], len(gt), axis=0)
        relative = np.column_stack(((gt[:, 2]-gt[:, 0])/32, (gt[:, 3]-gt[:, 1])/32)*2)
        # Test every corner in both arms rather than using a max-amplitude guard.
        guard = np.ones(len(gt), bool)
        for amplitude in (absolute, relative):
            for sign in signs:
                corners = gt+sign*amplitude
                guard &= ((corners[:, 2:] > corners[:, :2]).all(1) &
                          (corners[:, :2] >= [left, top]).all(1) &
                          (corners[:, 2:] <= [left+rw, top+rh]).all(1))
        rows = np.array([i for i in range(len(old_arrays["owner"]))
                         if old_arrays["foreground"][i] and old_arrays["target_mass"][i] > 0
                         and guard[old_arrays["owner"][i]]], dtype=int)
        owners = old_arrays["owner"][rows]
        pred, target, stride = old_arrays["decoded_boxes"][rows], gt[owners], old_arrays["stride"][rows]
        mass = old_arrays["target_mass"][rows].astype(float)
        areas = (gt[:, 2]-gt[:, 0])*(gt[:, 3]-gt[:, 1])*(640/960)**2
        for key, value in dict(rows=rows, owners=owners, prediction=pred, target=target, stride=stride,
                               mass=mass, all_gt=gt, all_areas=areas, absolute=absolute, relative=relative, guard=guard).items():
            np.testing.assert_allclose(saved[key], value, atol=1e-12, rtol=1e-12)
        nominal = independent_gradient(pred, target, stride)
        perturbed = np.array([[independent_gradient(pred, target+sign*amplitude[owners], stride)
                               for sign in signs] for amplitude in (absolute, relative)])
        for key, value in (("nominal", nominal), ("perturbed", perturbed)):
            np.testing.assert_allclose(value, saved[key], atol=1e-11, rtol=1e-9)
            maximum_error = max(maximum_error, float(np.max(np.abs(value-saved[key]), initial=0)))
        checked += 33*len(rows)
        valid = np.linalg.norm(nominal, axis=1) > 1e-12
        flip = np.einsum("arkd,kd->ark", perturbed, nominal) < 0
        probabilities = flip.mean(1)
        np.testing.assert_array_equal(saved["valid"], valid)
        np.testing.assert_array_equal(saved["flip"], flip)
        np.testing.assert_array_equal(saved["probabilities"], probabilities)
        trace = {}
        for name, limit in (("all", math.inf), ("tiny", 256.), ("small_all", 1024.)):
            gt_mask = areas < limit
            selected = gt_mask[owners]
            eligible = selected & valid
            ids = set(owners[eligible])
            affected = set(owners[eligible & np.any(flip[0], axis=0)])
            trace[name] = dict(gt=int(sum(gt_mask)), guard_gt=int(sum(gt_mask & guard)),
                productive_rows=int(sum(selected)), productive_covered_gt=len(ids), productive_images=int(bool(ids)),
                zero_nominal_rows=int(sum(selected & ~valid)), zero_nominal_mass=float(sum(mass[selected & ~valid])),
                mass=math.fsum(mass[eligible]), affected_gt=len(affected), affected_images=int(bool(affected)),
                absolute_flip_mass=math.fsum(mass[eligible]*probabilities[0, eligible]),
                relative_flip_mass=math.fsum(mass[eligible]*probabilities[1, eligible]))
        traces.append(trace)
    compare(traces[0], read(output/"per_image.json")[0])
    for trace, expected in zip(traces, read(output/"per_image.json"), strict=True):
        compare(trace, expected)
    metrics = {}
    for name in traces[0]:
        group = {key: math.fsum(t[name][key] for t in traces) for key in traces[0][name]}
        group["absolute_flip_rate"] = group["absolute_flip_mass"]/group["mass"] if group["mass"] else 0.
        group["relative_flip_rate"] = group["relative_flip_mass"]/group["mass"] if group["mass"] else 0.
        group["absolute_minus_relative_flip_rate"] = group["absolute_flip_rate"]-group["relative_flip_rate"]
        metrics[name] = group
    weights = np.random.default_rng(0).multinomial(64, [1/64]*64, size=400)
    samples = []
    for weight in weights:
        denominator = math.fsum(w*t["tiny"]["mass"] for w, t in zip(weight, traces))
        assert denominator > 0
        samples.append(math.fsum(w*(t["tiny"]["absolute_flip_mass"]-t["tiny"]["relative_flip_mass"])
                                 for w, t in zip(weight, traces))/denominator)
    tiny = metrics["tiny"]
    tiny["absolute_minus_relative_image_bootstrap95ci"] = np.percentile(samples, [2.5, 97.5]).tolist()
    compare(metrics, result["metrics"])
    policy = plan["registration"]["policy"]
    checks = dict(covered_gt=tiny["productive_covered_gt"] >= policy["tiny_productive_covered_gt_min"],
                  images=tiny["productive_images"] >= policy["tiny_productive_images_min"],
                  absolute_flip=tiny["absolute_flip_rate"] >= policy["tiny_absolute_flip_rate_min"],
                  excess=tiny["absolute_minus_relative_flip_rate"] >= policy["tiny_absolute_minus_relative_flip_rate_min"],
                  interval=tiny["absolute_minus_relative_image_bootstrap95ci"][0] > policy["tiny_difference_bootstrap_lower_min"])
    assert checks == result["decision"]["checks"]
    assert not all(checks.values()) and result["decision"]["decision"] == "STOP_THIS_HALF_PIXEL_SPECIFIC_CIOU_ROUTE"
    audit = dict(all64_sources_archives_prior_frames_metadata_images_labels_checkpoint_hashed=True,
                 all_rows_guards_amplitudes_and_metrics_rebuilt=True,
                 numpy_forward_derivative_vectors_checked=checked, maximum_gradient_absolute_error=maximum_error,
                 all_sign_flips_400_bootstrap_draws_and_five_gates_recomputed=True,
                 extra_detector_forwards=0, extra_image_decodes=0, extra_model_updates=0,
                 verifier_sha256=file_hash(Path(__file__)),
                 limits="Same cached checkpoint tensors; independent NumPy derivative with detached alpha. Relative noise is not amplitude matched. No detector training, annotation-error measurement, AP or novelty proof.")
    destination = output/"independent_verification.json"
    assert not destination.exists()
    destination.write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    verify()
