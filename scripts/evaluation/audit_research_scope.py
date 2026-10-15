"""Retrospective evidence audit; never runs a new detector or changes old gates."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash, load_records, record_hash
from sdcl.evaluation.testdev_transfer import transfer_decision
from sdcl.postprocess.class_margin_probe import checked_control
from sdcl.postprocess.localization import frozen_control
from sdcl.postprocess.rescue import save_json


PROTOCOL = "docs/experiments/research_scope_audit_protocol_20261015.json"
LITERATURE = "docs/literature/research_scope_prior_audit_20261015.json"
OUTPUT = "output/analysis/research-scope-audit-fixed-20261015"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def outcome_strings(result):
    value = result.get("decision")
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        if isinstance(value.get("decision"), str):
            return [value["decision"]]
        return [value[k] for k in ("ranking_task", "regret_weight")
                if isinstance(value.get(k), str)]
    raise ValueError("Unrecognized historical decision; do not silently classify.")


def positive_lead(values, keywords):
    return any(not v.startswith("STOP") and any(k in v for k in keywords) for v in values)


def inventory():
    rows = {}
    for path in project_path("docs/experiments").glob("candidate*.json"):
        match = re.match(r"candidate(\d+)_", path.name)
        if not match:
            continue
        number = int(match[1])
        if not 1 <= number <= 69 or ("result" not in path.name and number > 7):
            continue
        if number in rows:
            raise ValueError(f"Ambiguous result for candidate{number}.")
        result = read(path)
        rows[number] = dict(file=path.relative_to(project_path(".")).as_posix(),
            sha256=file_hash(path), decisions=outcome_strings(result),
            recorded_scope=result.get("limits", result.get("scope", result.get("interpretation"))))
    if set(rows) != set(range(1, 70)):
        raise ValueError("Incomplete candidate1..69 inventory.")
    return dict(sorted(rows.items()))


def trace_leads(rows, protocol):
    detected = {str(n) for n, row in rows.items()
                if positive_lead(row["decisions"], protocol["positive_keyword_detection"])}
    declared = set(protocol["positive_lead_dispositions"])
    if detected != declared:
        raise ValueError(f"Untraced positive lead or stale mapping: {detected ^ declared}")
    traces = {}
    for n, disposition in protocol["positive_lead_dispositions"].items():
        if not disposition["evidence"] or not all(e in rows for e in disposition["evidence"]):
            raise ValueError(f"Incomplete positive lead disposition{n}.")
        traces[n] = dict(**disposition, source=rows[int(n)],
                        followup_decisions={str(e): rows[e]["decisions"] for e in disposition["evidence"]})
    return traces


def audit():
    protocol, literature = read(project_path(PROTOCOL)), read(project_path(LITERATURE))
    rows = inventory()
    traces = trace_leads(rows, protocol)
    selected = {}
    for number in (66, 67, 68, 69):
        result = read(project_path(rows[number]["file"]))
        root = project_path(result["output"])
        checks = {}
        for field in ("artifacts_sha256", "additional_audit_artifacts_sha256"):
            for name, expected in result.get(field, {}).items():
                if isinstance(expected, str):
                    alias = protocol.get("artifact_aliases", {}).get(str(number), {}).get(name)
                    path = project_path(alias) if alias else root/name
                    checks[name] = file_hash(path) == expected
        if not checks or not all(checks.values()):
            raise ValueError(f"Selected artifact corruption{number}.")
        selected[str(number)] = dict(output=str(root), hash_checks=checks,
            boundary="Current bytes match recorded hashes; not a rerun or independent revalidation of all historical scientific assumptions.")
    official = checked_control()
    ordinary = frozen_control()
    official_metrics = read(Path(official["evaluation"]["official_report"]))
    if file_hash(official["evaluation"]["official_report"]) != official["evaluation"]["sha256"]:
        raise ValueError("Changed official-control report.")
    transfer = read(project_path("docs/experiments/testdev_transfer_20261013.json"))
    transfer_root = project_path("output/analysis/testdev-transfer-fixed-20261013")
    transfer_gate_replay = transfer_decision(transfer["official_metrics_percent"], True)
    if transfer_gate_replay != transfer["decision"] or transfer_gate_replay != read(transfer_root/"decision.json"):
        raise ValueError("Historical official transfer gate arithmetic differs.")
    if not all(read(transfer_root/"verification.json")[k] for k in (
            "all_input_hashes_unchanged", "registered_source_unchanged",
            "detector_calibrator_unchanged", "official_toolkit_body_unchanged",
            "no_annotations_in_prediction", "no_training_or_parameter_selection")):
        raise ValueError("Recorded transfer audit is incomplete.")
    records = {s: load_records(project_path("data/visdrone"), s) for s in ("train", "val", "test")}
    if {s: len(v) for s, v in records.items()} != dict(train=6471, val=548, test=1610):
        raise ValueError("Dataset inventory differs.")
    transfer_plan = read(transfer_root/"plan.json")
    if record_hash(records["test"]) != transfer_plan["records_sha256"]:
        raise ValueError("Current test-dev records differ from the consumed transfer records.")
    if record_hash(records["val"]) != ordinary["frozen_validation"]["records_sha256"]:
        raise ValueError("Current val differs from the frozen strong control.")
    prior = project_path("output/analysis/transport-quality-fixed-20261015")
    prior_plan = read(prior/"plan.json")
    prior_frames = read(prior/"frames.json")
    selection = prior_plan["image_selection"]
    train_names = {r["stem"] for r in records["train"]}
    if not all(stem in train_names for stems in selection.values() for stem in stems):
        raise ValueError("Incorrect train-only pilot provenance.")
    for stem, frame in prior_frames.items():
        row = next(r for r in records["train"] if r["stem"] == stem)
        if row["image_sha256"] != frame["image_sha256"] or row["metadata_sha256"] != frame["metadata_sha256"]:
            raise ValueError("Pilot input content changed.")
    args_path = project_path("output/runs/baseline_yolo11s_960_100ep_seed0/args.yaml")
    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    if args["imgsz"] != 960 or args["epochs"] != 100 or args["fraction"] != 1:
        raise ValueError("Recorded detector training provenance differs.")
    byte_overlaps, stem_overlaps = {}, {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        byte_overlaps[a+":"+b] = len({r["image_sha256"] for r in records[a]} &
                                    {r["image_sha256"] for r in records[b]})
        stem_overlaps[a+":"+b] = len({r["stem"] for r in records[a]} &
                                    {r["stem"] for r in records[b]})
    if any(byte_overlaps.values()) or any(stem_overlaps.values()):
        raise ValueError("Unexpected split overlap; reassess the evidence before a program decision.")
    papers = []
    for p in literature["papers"]:
        path = project_path(p["pdf"])
        if file_hash(path) != p["pdf_sha256"]:
            raise ValueError("Changed primary PDF.")
        papers.append(dict(title=p["title"], pdf=p["pdf"], sha256=p["pdf_sha256"],
                           overlap_or_limit=p["overlap_or_limit"]))
    literature_readback = project_path(literature["zotero_readback"]["file"])
    if file_hash(literature_readback) != literature["zotero_readback"]["sha256"]:
        raise ValueError("Changed literature readback.")
    data = {s: dict(images=len(r), records_sha256=record_hash(r)) for s, r in records.items()}
    data.update(pilot_counts={s: len(v) for s, v in selection.items()},
        pilot_inputs_all_detector_training_images=True,
        exposure=dict(train="Recorded100epoch fraction1training; pilot heads reuse train images.",
            val="Already-used official-control and candidate20official outcomes exist.",
            test="All1610test-dev outcomes and further early-head comparisons were already consumed."),
        detector_training_args_sha256=file_hash(args_path),
        identical_image_bytes=byte_overlaps, identical_stems=stem_overlaps,
        boundary="Current byte/name separation does not prove geographic/video identity independence. Historical outcome exposure is not a measured overfitting effect; unused training images could still be held out from NEW heads, not from this backbone.")
    return dict(audit=protocol["audit"], decision=protocol["on_verified_evidence"],
        investment_scope=protocol["scope"], rejection_exclusions=protocol["not_in_scope_of_rejection"],
        candidates=rows, positive_lead_dispositions=traces, selected_artifacts=selected,
        dataset=data, official_control=dict(detector=official["detector"], scorer=official["scorer"],
            report_sha256=file_hash(official["evaluation"]["official_report"]),
            recorded_headline_metrics=official["evaluation"]["headline_metrics_percent"]),
        official_testdev=dict(metrics=transfer["official_metrics_percent"], gate_replay=transfer_gate_replay),
        papers=papers, qualified_records=protocol["engineering_qualified_records"],
        audit_limits=["No pooled p-value, independent-trial count or field-wide impossibility claim.",
            "No new AP/inference/learning. Selected artifact hashes and gate arithmetic are rechecked; old scientific conclusions retain their original limitations.",
            "Positive means and oracle gaps remain recorded. No established independent, differentiated deployment gain is inferred from them.",
            "Recommendation is a discretionary allocation decision within the stated low-cost scope; new data/representation mechanisms remain untested."],
        reopening_conditions=protocol["future_reopening_conditions"])


def main(verify=False):
    root = project_path(OUTPUT)
    if verify:
        stored = read(root/"audit.json")
        expected = read(root/"plan.json")
        for path, sha in expected["source_hashes"].items():
            if file_hash(project_path(path)) != sha:
                raise ValueError("Audit source drift.")
        for path, sha in expected["evidence_sha256"].items():
            if file_hash(project_path(path)) != sha:
                raise ValueError("Historical evidence drift.")
        if file_hash(root/"source.zip") != expected["source_archive_sha256"]:
            raise ValueError("Audit source archive changed.")
        with zipfile.ZipFile(root/"source.zip") as archive:
            for path, sha in expected["source_hashes"].items():
                if hashlib.sha256(archive.read(path)).hexdigest() != sha:
                    raise ValueError("Archived audit source differs.")
        rebuilt = json.loads(json.dumps(audit()))
        if rebuilt != stored:
            raise ValueError("Audit reconstruction differs.")
        result = dict(passed=True, candidate_records=69, positive_leads_traced=len(stored["positive_lead_dispositions"]),
            current_image_byte_hashes=8629, detector_forwards=0, optimizer_steps=0,
            boundary="Deterministic replay of the SAME audit code, not an independent scientific review.")
        save_json(root/"verification.json", result)
        return result
    if root.exists():
        raise FileExistsError(root)
    started = perf_counter()
    sources = [PROTOCOL, LITERATURE, "scripts/evaluation/audit_research_scope.py",
               "tests/test_research_scope_audit.py",
               "sdcl/config.py", "sdcl/evaluation/size_analysis.py",
               "sdcl/evaluation/testdev_transfer.py", "sdcl/postprocess/class_margin_probe.py",
               "sdcl/postprocess/localization.py", "sdcl/postprocess/rescue.py"]
    hashes = {p: file_hash(project_path(p)) for p in sources}
    evidence = {r["file"]: r["sha256"] for r in inventory().values()}
    for name in ("docs/experiments/testdev_transfer_20261013.json",
                 "docs/experiments/ranking_nearest_prior_20261013.json",
                 "docs/literature/crop_teacher_prior_audit_20261014.json"):
        evidence[name] = file_hash(project_path(name))
    root.mkdir(parents=True)
    with zipfile.ZipFile(root/"source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(project_path(path), path)
    save_json(root/"plan.json", dict(type="Retrospective audit, not blind experiment.",
        source_hashes=hashes, evidence_sha256=evidence,
        source_archive_sha256=file_hash(root/"source.zip"), base_commit="a0ffeff"))
    result = audit()
    if any(file_hash(project_path(p)) != h for p, h in hashes.items()):
        raise ValueError("Audit sources changed during execution.")
    save_json(root/"audit.json", result)
    save_json(root/"costs.json", dict(elapsed_seconds=perf_counter()-started,
        detector_forwards=0, image_decodes=0, optimizer_steps=0, training_steps=0,
        boundary="File hashing/JSON analysis only; no statistical test of a new method."))
    return dict(decision=result["decision"], records=len(result["candidates"]),
                positive_leads=len(result["positive_lead_dispositions"]), output=str(root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    print(json.dumps(main(**vars(parser.parse_args())), indent=2))
