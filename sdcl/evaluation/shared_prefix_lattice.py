"""Test a necessary sampling-lattice condition before shared-prefix inference."""

from fractions import Fraction
import hashlib
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
import zipfile

import cv2
import numpy as np

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash
from sdcl.evaluation.slicing_screen import BASELINE, BASELINE_SHA256, tile_windows


REGISTRATION = "docs/experiments/candidate46_shared_prefix_lattice_registration_20261014.json"
OUTPUT = "output/analysis/shared-prefix-lattice-fixed-20261014"
STRIDES = (1, 2, 4, 8)


def geometry(width, height):
    windows = tile_windows(width, height)
    tw, th = windows[0][2:]
    ratio = min(960 / tw, 960 / th)
    rw, rh = round(tw * ratio), round(th * ratio)
    nx, ny = (width - tw) * rw, (height - th) * rh
    aligned = {str(s): nx % (tw * s) == 0 and ny % (th * s) == 0 for s in STRIDES}
    return {
        "size": [width, height], "windows": [list(w) for w in windows],
        "crop_size": [tw, th], "resized_content": [rw, rh],
        "shift_x_fraction": [nx, tw], "shift_y_fraction": [ny, th],
        "aligned": aligned,
    }


def independent_geometry(width, height):
    # Separate rational-arithmetic implementation; no call to geometry/tile_windows.
    tw, th = min(width, (5 * width + 8) // 9), min(height, (5 * height + 8) // 9)
    ratio = min(960 / tw, 960 / th)
    rw, rh = round(tw * ratio), round(th * ratio)
    dx, dy = Fraction((width - tw) * rw, tw), Fraction((height - th) * rh, th)
    result = {str(s): (dx / s).denominator == 1 and (dy / s).denominator == 1 for s in STRIDES}
    return [tw, th], [rw, rh], result


def overlap_statistics(a, b, offset_a, offset_b, border=2):
    """Compare nearest-phase content grids only within their common interior."""
    ax, ay = offset_a
    bx, by = offset_b
    lo_x, lo_y = max(ax, bx) + border, max(ay, by) + border
    hi_x = min(ax + a.shape[1], bx + b.shape[1]) - border
    hi_y = min(ay + a.shape[0], by + b.shape[0]) - border
    if hi_x <= lo_x or hi_y <= lo_y:
        raise ValueError("No common interior in selected views.")
    first = a[lo_y-ay:hi_y-ay, lo_x-ax:hi_x-ax]
    second = b[lo_y-by:hi_y-by, lo_x-bx:hi_x-bx]
    error = np.abs(first.astype(np.int16) - second.astype(np.int16))
    return {
        "pixels": int(error.shape[0] * error.shape[1]),
        "channels": int(error.size), "nonzero_pixels": int(np.any(error, axis=2).sum()),
        "nonzero_channels": int(np.count_nonzero(error)),
        "absolute_error_sum": int(error.sum()), "max_absolute_error": int(error.max()),
        "first_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
        "second_sha256": hashlib.sha256(second.tobytes()).hexdigest(),
    }


def pixel_probe(image, shape):
    from ultralytics.data.augment import LetterBox

    rw, rh = shape["resized_content"]
    contents, offsets, replay = [], [], []
    tw, th = shape["crop_size"]
    for x1, y1, x2, y2 in shape["windows"]:
        crop = image[y1:y2, x1:x2].copy()
        letterbox = LetterBox((960, 960), auto=False, center=True)
        params = letterbox.get_params({"img": crop})
        padded = letterbox(image=crop)
        content = padded[params["top"]:params["top"]+rh, params["left"]:params["left"]+rw]
        expected = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR)
        replay.append(bool(np.array_equal(content, expected)))
        contents.append(content)
        offsets.append([round(Fraction(x1*rw, tw)), round(Fraction(y1*rh, th))])
    comparisons = []
    for i, j in combinations(range(4), 2):
        comparisons.append({"pair": [i, j], **overlap_statistics(contents[i], contents[j], offsets[i], offsets[j])})
    return {"letterbox_replay": replay, "rounded_atlas_offsets": offsets, "comparisons": comparisons}


def decide(count, eligible, replay, independent, minimum=.8):
    fraction = eligible / count
    checks = {"stride2_coverage": fraction >= minimum, "letterbox_replay": bool(replay),
              "independent_geometry": bool(independent)}
    return {
        "decision": "PREFIX_PROTOTYPE_ALLOWED" if all(checks.values()) else "STOP_THIS_SINGLE_ATLAS_INTEGER_SLICE_RECIPE",
        "checks": checks, "stride2_fraction": fraction,
        "scope": "Necessary-condition screen only; no AP, feature error, speedup, efficacy or novelty result.",
    }


def run(output=OUTPUT):
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.engine.predictor import BasePredictor
    import inspect
    import ultralytics
    from sdcl.training.trainer import check_version

    check_version()
    started = perf_counter()
    output = project_path(output)
    if output.exists():
        raise FileExistsError("Refuse to overwrite an existing lattice experiment.")
    registration = json.loads(project_path(REGISTRATION).read_text())
    if file_hash(project_path(BASELINE)) != BASELINE_SHA256:
        raise ValueError("Frozen960 checkpoint mismatch.")
    model = YOLO(str(project_path(BASELINE))).model.eval()
    if tuple(model.model[0].conv.stride) != (2, 2):
        raise ValueError("Expected actual first convolution stride2.")
    metadata = sorted(project_path("data/visdrone/metadata/train").glob("*.json"))
    if len(metadata) != 6471:
        raise ValueError("Expected6471 training metadata files.")
    rows = []
    for path in metadata:
        item = json.loads(path.read_text(encoding="utf-8"))
        width, height = item["original_size"]
        rows.append({"stem": path.stem, "metadata_sha256": file_hash(path), **geometry(width, height)})
    selected = sorted(rows, key=lambda r: hashlib.sha256(
        f"{r['stem']}:shared-prefix-lattice-v1".encode()).digest())[:16]
    source_paths = [
        REGISTRATION, "sdcl/evaluation/shared_prefix_lattice.py",
        "scripts/evaluation/diagnose_shared_prefix_lattice.py", "tests/test_shared_prefix_lattice.py",
        "docs/literature/shared_prefix_lattice_prior_audit_20261014.json",
        "sdcl/config.py", "sdcl/evaluation/size_analysis.py", "sdcl/evaluation/slicing_screen.py",
        "sdcl/training/trainer.py",
    ]
    sources = {p: file_hash(project_path(p)) for p in source_paths}
    installed = {str(Path(inspect.getfile(t)).resolve()): file_hash(inspect.getfile(t))
                 for t in (LetterBox, BasePredictor, type(model), type(model.model[0]))}
    plan = {
        "registration": registration, "source_hashes": sources, "installed_source_hashes": installed,
        "ultralytics": ultralytics.__version__, "opencv": cv2.__version__,
        "detector_sha256": BASELINE_SHA256, "selected_stems": [r["stem"] for r in selected],
        "model_save_links": model.save,
        "graph": [{"index": layer.i, "from": layer.f, "type": layer.type} for layer in model.model],
    }
    output.mkdir(parents=True)
    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    with zipfile.ZipFile(output/"source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(project_path(path), path)
    save("plan.json", plan)
    save("geometry.json", rows)
    probes = []
    for row in selected:
        path = project_path(f"data/visdrone/images/train/{row['stem']}.jpg")
        image = cv2.imread(str(path))
        if image is None or list(image.shape[1::-1]) != row["size"]:
            raise ValueError("Invalid selected train image.")
        probes.append({"stem": row["stem"], "image_sha256": file_hash(path),
                       **pixel_probe(image, row)})
    save("pixels.json", probes)
    counts = {str(s): sum(r["aligned"][str(s)] for r in rows) for s in STRIDES}
    independent = all(
        independent_geometry(*r["size"]) == (r["crop_size"], r["resized_content"], r["aligned"])
        for r in rows)
    replay = all(all(p["letterbox_replay"]) for p in probes)
    pairs = [p for r in probes for p in r["comparisons"]]
    channels = sum(p["channels"] for p in pairs)
    metrics = {
        "train_images": len(rows), "eligible_images": counts,
        "eligible_fraction": {s: n/len(rows) for s, n in counts.items()},
        "selected_images": len(probes), "pair_comparisons": len(pairs),
        "overlap_pixels": sum(p["pixels"] for p in pairs), "overlap_channels": channels,
        "nonzero_channel_fraction": sum(p["nonzero_channels"] for p in pairs)/channels,
        "mean_absolute_uint8_error": sum(p["absolute_error_sum"] for p in pairs)/channels,
        "max_absolute_uint8_error": max(p["max_absolute_error"] for p in pairs),
        "exact_pair_comparisons": sum(p["nonzero_channels"] == 0 for p in pairs),
    }
    decision = decide(len(rows), counts["2"], replay, independent,
                      registration["policy"]["minimum_train_fraction_all_four_views_same_stride2_lattice"])
    verification = {
        "all6471_geometry_independently_recomputed": independent,
        "all64_content_replays_exact": replay, "detector_forwards": 0, "training_steps": 0,
        "image_decodes": 16, "elapsed_seconds": perf_counter()-started,
        "source_unchanged": all(file_hash(project_path(p)) == h for p, h in sources.items()),
        "scope": "CPU geometry/pixel screen. Nearest-phase disagreement is not detector feature error or AP harm.",
    }
    save("metrics.json", metrics)
    save("decision.json", decision)
    save("verification.json", verification)
    result = {
        "experiment": registration["experiment"], "date": "2026-10-14",
        "output": str(output), "decision": decision, "metrics": metrics, "verification": verification,
        "artifacts_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    save("result.json", result)
    return result
