"""Numerical engineering audit only: never refit or rescore stopped arms."""

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.screen_scaling import FitEnvelope
from sdcl.evaluation.size_analysis import file_hash


ROOT = "output/analysis/screen-conditioning-fixed-20261015"
PRIOR = "output/analysis/external-content-fixed-20261015"
REGISTRATION = "docs/experiments/candidate59_screen_conditioning_registration_20261015.json"


def run():
    started = perf_counter()
    root, prior = project_path(ROOT), project_path(PRIOR)
    if root.exists():
        raise FileExistsError(root)
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    old_result, old_plan = read(prior/"result.json"), read(prior/"plan.json")
    for name, digest in old_result["artifacts_sha256"].items():
        assert file_hash(prior/name) == digest
    with zipfile.ZipFile(prior/"source.zip") as z:
        for name, digest in old_plan["source_hashes"].items():
            assert file_hash(project_path(name)) == hashlib.sha256(z.read(name)).hexdigest() == digest
    original = project_path("output/analysis/texture-deletion-fixed-20261015")
    original_manifest = read(original/"frames.json")
    assert file_hash(original/"frames.json") == old_plan["prior_frames_manifest_sha256"]
    sources = {p: file_hash(project_path(p)) for p in (
        REGISTRATION, "scripts/evaluation/diagnose_screen_conditioning.py",
        "sdcl/evaluation/screen_scaling.py", "tests/test_screen_scaling.py")}
    root.mkdir(parents=True)
    with zipfile.ZipFile(root/"source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in sources:
            z.write(project_path(name), name)
    manifest, old_stats = read(prior/"frames.json"), read(prior/"normalization.json")["base"]
    input_hashes = {
        "prior_result": file_hash(prior/"result.json"),
        "prior_frames_manifest": file_hash(prior/"frames.json"),
        "prior_normalization": file_hash(prior/"normalization.json")}
    plan = dict(registration=read(project_path(REGISTRATION)), source_hashes=sources,
                source_archive_sha256=file_hash(root/"source.zip"), prior_hashes=input_hashes)
    (root/"plan.json").write_text(json.dumps(plan, indent=2)+"\n", encoding="utf-8")
    frames, fit_values = [], []
    for stem, entry in manifest.items():
        assert file_hash(prior/entry["file"]) == entry["sha256"]
        assert file_hash(prior/entry["inputs_file"]) == entry["inputs_sha256"]
        old_entry = original_manifest[stem]
        assert file_hash(original/old_entry["file"]) == old_entry["sha256"] == entry["prior_frame_sha256"]
        with np.load(prior/entry["file"], allow_pickle=False) as z:
            values, eligible = z["base"].copy(), z["eligible"].copy()
        with np.load(original/old_entry["file"], allow_pickle=False) as z:
            np.testing.assert_array_equal(values, z["base"])
            np.testing.assert_array_equal(eligible, z["eligible"])
        with np.load(prior/entry["inputs_file"], allow_pickle=False) as z:
            old_inputs = z["base"][:, :92].copy()
        assert values.shape == (len(eligible), 92)
        np.testing.assert_array_equal(old_inputs, (values-old_stats["mean"])/old_stats["std"])
        frames.append(dict(stem=stem, split=entry["split"], values=values, eligible=eligible, old=old_inputs))
        if entry["split"] == "fit":
            fit_values.append(values[eligible])
    fit_values = np.concatenate(fit_values)
    np.testing.assert_array_equal(fit_values.mean(0), old_stats["mean"])
    np.testing.assert_array_equal(np.maximum(fit_values.std(0), 1e-6), old_stats["std"])
    envelope = FitEnvelope.fit(fit_values)
    for column in range(92):
        assert envelope.lower[column] == min(float(row[column]) for row in fit_values)
        assert envelope.upper[column] == max(float(row[column]) for row in fit_values)
    frozen_state = json.dumps(envelope.to_dict(), sort_keys=True)
    clone = FitEnvelope.from_dict(json.loads(frozen_state))
    traces, outputs = [], []
    for frame in frames:
        values, keep = frame["values"], frame["eligible"]
        bounded = envelope.transform(values)
        np.testing.assert_array_equal(bounded, clone.transform(values))
        scalar = np.zeros(values.shape)
        for j in range(92):
            lo, hi = float(envelope.lower[j]), float(envelope.upper[j])
            radius, center = .5*hi-.5*lo, .5*lo+.5*hi
            if radius <= 64*sys.float_info.epsilon*max(1., abs(lo), abs(hi)):
                continue
            for i, value in enumerate(values[:, j]):
                x = max(lo, min(hi, float(value)))
                scalar[i, j] = -1. if x == lo else 1. if x == hi else max(-1., min(1., (x-center)/radius))
        np.testing.assert_array_equal(bounded, scalar)
        assert np.isfinite(bounded).all() and np.all(abs(bounded) <= 1.)
        assert np.all(bounded[:, ~envelope.active] == 0.)
        outside = ((values < envelope.lower) | (values > envelope.upper)) & envelope.active
        traces.append(dict(stem=frame["stem"], split=frame["split"], rows=len(values),
            eligible_rows=int(keep.sum()),
            old_absmax=float(abs(frame["old"][keep]).max()) if keep.any() else 0.,
            old_abs_over1000_rows=int((abs(frame["old"][keep]) > 1000).any(1).sum()),
            clipped_rows=int(outside[keep].any(1).sum()), clipped_elements=int(outside[keep].sum()),
            new_absmax=float(abs(bounded[keep]).max()) if keep.any() else 0.))
        outputs.append((frame["stem"], hashlib.sha256(bounded.tobytes()).hexdigest()))
    assert frozen_state == json.dumps(envelope.to_dict(), sort_keys=True)
    assert len(frames) == 192 and sum(t["rows"] for t in traces) == 2007
    splits = {}
    for split in ("fit", "screen", "confirmation"):
        rows = [t for t in traces if t["split"] == split]
        splits[split] = dict(images=len(rows),
            **{k: sum(t[k] for t in rows) for k in ("rows", "eligible_rows", "old_abs_over1000_rows", "clipped_rows", "clipped_elements")},
            old_abs_over1000_images=sum(t["old_abs_over1000_rows"] > 0 for t in rows),
            old_absmax=max(t["old_absmax"] for t in rows), new_absmax=max(t["new_absmax"] for t in rows))
    columns = []
    for j in range(92):
        holdout = {}
        for split in ("screen", "confirmation"):
            values = np.concatenate([f["values"][f["eligible"], j] for f in frames if f["split"] == split])
            holdout[split] = [float(values.min()), float(values.max())]
        columns.append(dict(column=j, active=bool(envelope.active[j]), fit_mean=float(fit_values[:, j].mean()),
            fit_std=float(fit_values[:, j].std()), fit_range=[float(envelope.lower[j]), float(envelope.upper[j])],
            holdout_ranges=holdout))
    for name, value in (
        ("envelope.json", envelope.to_dict()), ("per_image.json", traces),
        ("columns.json", columns), ("transformed_hashes.json", dict(outputs))):
        (root/name).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    assert all(file_hash(project_path(p)) == digest for p, digest in sources.items())
    assert file_hash(prior/"result.json") == input_hashes["prior_result"]
    result = dict(experiment=plan["registration"]["experiment"], date="2026-10-15", output=str(root),
        decision="NUMERICAL_ENVELOPE_VERIFIED_FOR_FUTURE_OPT_IN_SCREENS",
        splits=splits, inactive_columns=np.flatnonzero(~envelope.active).tolist(),
        original_normalization_exactly_reproduced=True, fit_only_scalar_reconstruction_exact=True,
        all2007vectors_independently_scalar_checked=True, state_unchanged_and_JSON_roundtrip_exact=True,
        historical_recipe_and_STOP_unchanged=True,
        costs=dict(cached_frames=192, JPEG_decodes=0, model_forwards=0, gradient_calls=0,
                   linear_refits=0, parameter_updates=0, AP_evaluations=0,
                   elapsed_seconds=perf_counter()-started),
        limits="Engineering transform only. No old coefficients applied to new inputs, no NLL or deployment gain estimated. Fit-range clipping and zeroing inactive columns can remove useful information; no innovation or detector-training authorization.",
        artifacts_sha256={p.name: file_hash(p) for p in sorted(root.iterdir()) if p.is_file()})
    (root/"result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
