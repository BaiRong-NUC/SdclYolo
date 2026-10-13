from copy import deepcopy
import json

import pytest

from sdcl.evaluation.coverage_screen import SCREEN_CONFIG, check_coverage_training_pair
from sdcl.evaluation.size_analysis import file_hash
from test_phase_screen import make_pair, reports


def coverage_pair(tmp_path):
    pair = make_pair(tmp_path)
    data = tmp_path / "data.txt"
    data.write_text("frozen data")
    for checkpoint in pair:
        path = checkpoint.parent.parent / "experiment.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["phase"] = {"enabled": False}
        manifest["coverage"] = SCREEN_CONFIG.to_dict() if checkpoint == pair[1] else {"enabled": False}
        manifest["dataset_hashes"] = {str(data): file_hash(data)}
        path.write_text(json.dumps(manifest), encoding="utf-8")
    (pair[1].parent.parent / "coverage_statistics.jsonl").write_text("\n".join(json.dumps({
        "epoch": epoch, "batches": 809, "rescued_gt": 1 if epoch >= 5 else 0,
        "small_uncovered_before": 10, "small_uncovered_after": 9 if epoch >= 5 else 10,
        "donor_starvation": 0,
    }) for epoch in range(100)), encoding="utf-8")
    return pair


def test_rejects_recipe_data_source_changes_and_absent_mechanism(tmp_path):
    pair = coverage_pair(tmp_path)
    result = check_coverage_training_pair(*pair, verify_checkpoints=False)
    assert result["mechanism"]["last10_rescued_gt"] == 10
    path = pair[1].parent.parent / "experiment.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    for field, value in [("lr0", .02), ("model", "best.pt"), ("resume", True)]:
        changed = deepcopy(original)
        changed["train"][field] = value
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            check_coverage_training_pair(*pair, verify_checkpoints=False)
    path.write_text(json.dumps(original), encoding="utf-8")
    log = pair[1].parent.parent / "coverage_statistics.jsonl"
    stats = [json.loads(line) for line in log.read_text().splitlines()]
    stats[-1]["rescued_gt"] = 0
    log.write_text("\n".join(json.dumps(row) for row in stats))
    with pytest.raises(ValueError, match="inactive"):
        check_coverage_training_pair(*pair, verify_checkpoints=False)
    stats[-1]["rescued_gt"] = 1
    stats[-1]["donor_starvation"] = 1
    log.write_text("\n".join(json.dumps(row) for row in stats))
    with pytest.raises(ValueError, match="invariant"):
        check_coverage_training_pair(*pair, verify_checkpoints=False)


def test_screen_recaptures_both_models_and_uses_five_detection_gates(tmp_path, monkeypatch):
    import sdcl.evaluation.coverage_screen as screen
    pair = coverage_pair(tmp_path)
    report, _, _ = reports()
    captures = []
    monkeypatch.setattr(screen, "check_coverage_training_pair",
                        lambda *a: {"mechanism": {"last10_rescued_gt": 10}})

    def capture(baseline, candidate, **kwargs):
        captures.append(kwargs)
        destination = screen.project_path(kwargs["output"])
        destination.mkdir()
        (destination / "report.json").write_text(json.dumps(report))

    monkeypatch.setattr(screen, "run_size_analysis", capture)
    destination = tmp_path / "screen"
    result = screen.run_coverage_screen(*pair, output=destination)
    assert result["decision"] == "CONTINUE" and len(result["checks"]) == 5
    assert "coverage-assignment" in result["rule"]
    assert captures[0]["baseline_imgsz"] == captures[0]["sdcl_imgsz"] == 960
    assert captures[0]["reference_size"] == 640
    assert "baseline_prediction_cache" not in captures[0]
    assert (destination / "decision.md").is_file()
    with pytest.raises(FileExistsError):
        screen.run_coverage_screen(*pair, output=destination)
    report["coco_reference"]["sdcl"]["tiny"]["ap50_95"] = .19
    assert screen.run_coverage_screen(*pair, output=tmp_path / "negative")["decision"] == "STOP"
