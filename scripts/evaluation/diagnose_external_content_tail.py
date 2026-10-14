"""Post-outcome attribution of saved candidate58 logits; no refitting."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash


def audit():
    root = project_path("output/analysis/external-content-fixed-20261015")
    read = lambda name: json.loads((root/name).read_text(encoding="utf-8"))
    result, manifest, stats = read("result.json"), read("frames.json"), read("normalization.json")
    destination = root/"posthoc_tail_attribution.json"
    if destination.exists():
        raise FileExistsError(destination)
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(root/name) == digest
    with np.load(root/"coefficients.npz", allow_pickle=False) as z:
        coefficients = {a: z[a].copy() for a in ("base", "own")}
    per_image, detail = [], {}
    for stem, entry in manifest.items():
        assert file_hash(root/entry["file"]) == entry["sha256"]
        assert file_hash(root/entry["inputs_file"]) == entry["inputs_sha256"]
        if entry["split"] == "fit":
            continue
        with np.load(root/entry["file"], allow_pickle=False) as z:
            frame = {k: z[k].copy() for k in z.files}
        with np.load(root/entry["inputs_file"], allow_pickle=False) as z:
            inputs = {a: z[a].copy() for a in ("base", "own")}
            saved = {a: z[a+"_logits"].copy() for a in inputs}
        keep = frame["eligible"]
        if not keep.any():
            continue
        y = frame["labels"][frame["selected"]][keep]
        p = np.clip(frame["reference"][frame["selected"], 4], 1e-6, 1-1e-6)
        offset = np.log(p/(1-p))
        losses = {}
        for arm, x in inputs.items():
            theta = coefficients[arm]
            logits = offset+x@theta[:-1]+theta[-1]
            np.testing.assert_allclose(logits, saved[arm], atol=1e-11, rtol=1e-10)
            losses[arm] = np.logaddexp(0, logits[keep])-y*logits[keep]
        nbase = frame["base"].shape[1]
        x, theta = inputs["own"], coefficients["own"]
        standardized = (frame["base"]-stats["base"]["mean"])/stats["base"]["std"]
        np.testing.assert_allclose(x[:, :nbase], standardized, atol=1e-11, rtol=1e-10)
        base_contribution = x[keep, :nbase]@theta[:nbase]
        content_contribution = x[keep, nbase:]@theta[nbase:-1]
        columns = x[keep, :nbase]*theta[:nbase]
        column = int(np.unravel_index(np.argmax(abs(columns)), columns.shape)[1])
        # Candidate56 appended 10 native logits, one association IoU, then 12 texture values.
        assert nbase == 92 and column < nbase
        summary = dict(stem=stem, split=entry["split"], rows=int(keep.sum()),
            mean_NLL={a: float(v.mean()) for a, v in losses.items()},
            own_minus_base_loss_sum=float((losses["own"]-losses["base"]).sum()))
        per_image.append(summary)
        span = lambda v: [float(np.min(v)), float(np.max(v))]
        detail[stem] = dict(
            labels=y.tolist(), own_common_base_logit_contribution_range=span(base_contribution),
            own_added_DINO_logit_contribution_range=span(content_contribution),
            own_intercept=float(theta[-1]),
            maximum_abs_standardized_base=float(abs(x[keep, :nbase]).max()),
            maximum_abs_standardized_DINO=float(abs(x[keep, nbase:]).max()),
            dominant_common_base_column=column,
            dominant_column_name="native_association_IoU" if column == nbase-13 else "other",
            fit_mean=float(stats["base"]["mean"][column]),
            fit_std=float(stats["base"]["std"][column]),
            raw_values_range=span(frame["base"][keep, column]),
            own_head_coefficient=float(theta[column]),
            dominant_column_logit_contribution_range=span(columns[:, column]))
    per_image.sort(key=lambda row: (-row["own_minus_base_loss_sum"], row["stem"]))
    for split in ("screen", "confirmation"):
        rows = [row for row in per_image if row["split"] == split]
        for arm in coefficients:
            nll = sum(row["rows"]*row["mean_NLL"][arm] for row in rows)/sum(row["rows"] for row in rows)
            np.testing.assert_allclose(nll, result["metrics"][split]["NLL"][arm], atol=1e-11)
    worst = per_image[0]
    output = dict(
        selection="Post-outcome maximum own-minus-base loss sum over every eligible screen/confirmation image; not preregistered.",
        result_sha256=file_hash(root/"result.json"),
        verifier_sha256=file_hash(Path(__file__)),
        independent_verification_file_sha256=file_hash(root/"independent_verification.json"),
        per_image=per_image, worst_image=dict(**worst, **detail[worst["stem"]]),
        interpretation="The worst frame's extreme loss is dominated by extrapolation of the almost-constant common-base native-association IoU, not extreme DINO embeddings. Both original split comparisons versus base still fail. No rows excluded, normalization changed, refit, new gate or rescue arm.",
        additional_JPEG_decodes=0, additional_encoder_forwards=0,
        parameter_updates=0, AP_evaluations=0,
        limitation="Saved tensors only; no independent encoder replay of the posthoc worst image. This does not isolate intrinsic DINO information or invalidate unrelated historical experiments.")
    destination.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(output["worst_image"], indent=2))


if __name__ == "__main__":
    audit()
