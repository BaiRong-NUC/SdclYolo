from copy import deepcopy
import json

import pytest

from sdcl.config import load_experiment
from sdcl.phase_screen import (
    SCREEN_CONFIG, check_phase_training_pair, decide_phase, phase_deltas,
)
from sdcl.size_analysis import file_hash


def reports():
    report = {
        "protocol": "diagnostic", "reference_size": 640,
        "inference_sizes": {"baseline": 960, "sdcl": 960},
        "captures": {name: {"settings": {"checkpoint_sha256": name, "imgsz": 960,
                                       "records_sha256": "same"}}
                     for name in ("baseline", "sdcl")},
        "coco_reference": {
            name: {group: {"ap50_95": .2 if name == "baseline" else .21}
                   for group in ("all", "tiny", "small_all")}
            for name in ("baseline", "sdcl")},
        "matched_fp_budgets": [{
            "label": "fppi_10", "delta_recall_pp": {"small_all": 1.},
            "paired_image_bootstrap_delta_recall_95ci_pp": {"small_all": [.1, 2.]}}],
    }
    probe = {"settings": {"imgsz": 960}, "data_hashes": {"data": "same"},
             "source_hashes": {"phase.py": "same"},
             "groups": {g: {"objects": 100, "comparisons": 100,
                            "mean_cosine_distance": .2, "hit_flip_rate": .1}
                        for g in ("tiny", "small", "medium", "large")}}
    other = deepcopy(probe)
    other["groups"]["tiny"].update(mean_cosine_distance=.15, hit_flip_rate=.08)
    return report, probe, other


def test_requires_utility_and_mechanism_and_stop_does_not_reject_960():
    report, first, second = reports()
    result = decide_phase(report, first, second)
    assert result["decision"] == "CONTINUE" and len(result["checks"]) == 7
    second["groups"]["tiny"]["mean_cosine_distance"] = .2
    result = decide_phase(report, first, second)
    assert result["decision"] == "STOP" and result["utility_passed"]
    assert "retains the verified 960 baseline" in result["scope"]
    second["groups"]["tiny"]["mean_cosine_distance"] = .15
    report["coco_reference"]["sdcl"]["tiny"]["ap50_95"] = .201
    assert decide_phase(report, first, second)["decision"] == "STOP"


@pytest.mark.parametrize("key", ["settings", "data_hashes", "source_hashes", "counts", "nan"])
def test_invalid_probe_is_not_a_negative_research_result(key):
    _, first, second = reports()
    if key == "counts":
        second["groups"]["small"]["objects"] -= 1
    elif key == "nan":
        second["groups"]["tiny"]["mean_cosine_distance"] = float("nan")
    else:
        second[key]["changed"] = True
    with pytest.raises(ValueError):
        phase_deltas(first, second)


def make_pair(tmp_path):
    from pathlib import Path
    import sdcl.phase_screen
    train, settings, ignore = load_experiment("configs/experiments/baseline_960_100.yaml")
    train.update(resume=False)
    source = {p.name: file_hash(p) for p in sorted(Path(sdcl.phase_screen.__file__).parent.glob("*.py"))}
    checkpoints = []
    for name, phase in (("baseline", {}), ("candidate", SCREEN_CONFIG.to_dict())):
        run = tmp_path / name
        (run / "weights").mkdir(parents=True)
        checkpoint = run / "weights/best.pt"
        checkpoint.write_bytes(name.encode())
        saved = {"train": {**train, "name": name, "save_dir": str(run)},
                 "sdcl": settings.to_dict(), "ignore_regions": ignore,
                 "phase": phase, "dataset_hashes": {"data": "same"},
                 "packages": {"torch": "same"}, "sdcl_source_hashes": source}
        (run / "experiment.json").write_text(json.dumps(saved), encoding="utf-8")
        (run / "results.csv").write_text("epoch\n" + "\n".join(map(str, range(1, 101))) + "\n")
        if name == "candidate":
            (run / "phase_statistics.jsonl").write_text("\n".join(json.dumps({
                "epoch": epoch, "active_steps": 1, "mean_weighted_loss_all_batches": .1,
            }) for epoch in range(100)))
        checkpoints.append(checkpoint)
    return checkpoints


def test_manifest_rejects_partial_runs_changed_recipe_and_inactive_objective(tmp_path):
    pair = make_pair(tmp_path)
    check_phase_training_pair(*pair, verify_checkpoints=False)
    manifest_path = pair[1].parent.parent / "experiment.json"
    original = json.loads(manifest_path.read_text())
    for key, value in (("imgsz", 640), ("epochs", 99), ("lr0", .02), ("model", "best.pt")):
        changed = deepcopy(original)
        changed["train"][key] = value
        manifest_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            check_phase_training_pair(*pair, verify_checkpoints=False)
    manifest_path.write_text(json.dumps(original))
    log = pair[1].parent.parent / "phase_statistics.jsonl"
    log.write_text("\n".join(json.dumps({
        "epoch": epoch, "active_steps": 0, "mean_weighted_loss_all_batches": 0,
    }) for epoch in range(100)))
    with pytest.raises(ValueError, match="inactive"):
        check_phase_training_pair(*pair, verify_checkpoints=False)


def test_screen_writes_reviewable_decision_and_recaptures_both_models(tmp_path, monkeypatch):
    import sdcl.phase_screen as screen
    report, first, second = reports()
    pair = make_pair(tmp_path)
    monkeypatch.setattr(screen, "check_phase_training_pair", lambda *args: {"verified": True})
    captures = []

    def capture(baseline, candidate, **kwargs):
        captures.append(kwargs)
        destination = screen.project_path(kwargs["output"])
        destination.mkdir()
        (destination / "report.json").write_text(json.dumps(report))

    probes = []

    def probe(checkpoint, output, **kwargs):
        probes.append(checkpoint)
        path = screen.project_path(output)
        path.write_text(json.dumps(first if checkpoint == str(pair[0]) else second))

    monkeypatch.setattr(screen, "run_size_analysis", capture)
    monkeypatch.setattr(screen, "run_phase_probe", probe)
    destination = tmp_path / "screen"
    result = screen.run_phase_screen(*pair, output=destination)
    assert result["decision"] == "CONTINUE"
    assert captures[0]["baseline_imgsz"] == captures[0]["sdcl_imgsz"] == 960
    assert captures[0]["reference_size"] == 640
    assert "baseline_prediction_cache" not in captures[0]
    assert probes == list(map(str, pair))
    saved = json.loads((destination / "decision.json").read_text())
    assert len(saved["checks"]) == 7
    assert (destination / "plan.json").is_file() and (destination / "decision.md").is_file()
    with pytest.raises(FileExistsError):
        screen.run_phase_screen(*pair, output=destination)
