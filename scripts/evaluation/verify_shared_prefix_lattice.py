"""Independently replay candidate46 geometry and all saved overlap comparisons."""

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sdcl.config import project_path
from sdcl.evaluation.size_analysis import file_hash


def verify():
    output = project_path("output/analysis/shared-prefix-lattice-fixed-20261014")
    read = lambda name: json.loads((output/name).read_text(encoding="utf-8"))
    plan, result = read("plan.json"), read("result.json")
    assert file_hash(project_path(
        "output/runs/baseline_yolo11s_960_100ep_seed0/weights/best.pt")) == plan["detector_sha256"]
    for name, digest in result["artifacts_sha256"].items():
        assert file_hash(output/name) == digest, name
    with zipfile.ZipFile(output/"source.zip") as archive:
        for name, digest in plan["source_hashes"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
            assert file_hash(project_path(name)) == digest, name
    for name, digest in plan["installed_source_hashes"].items():
        assert file_hash(name) == digest, name
    rows, pixels = read("geometry.json"), read("pixels.json")
    assert len(rows) == 6471
    assert [r["stem"] for r in rows] == sorted(
        p.stem for p in project_path("data/visdrone/metadata/train").glob("*.json"))
    counts = {str(s): 0 for s in (1, 2, 4, 8)}
    for row in rows:
        path = project_path(f"data/visdrone/metadata/train/{row['stem']}.json")
        assert file_hash(path) == row["metadata_sha256"]
        size = json.loads(path.read_text(encoding="utf-8"))["original_size"]
        assert size == row["size"]
        w, h = size
        tw, th = math.ceil(w*Fraction(5, 9)), math.ceil(h*Fraction(5, 9))
        assert [tw, th] == row["crop_size"]
        rw, rh = (round(tw*min(960/tw, 960/th)), round(th*min(960/tw, 960/th)))
        assert [rw, rh] == row["resized_content"]
        windows = [[0, 0, tw, th], [w-tw, 0, w, th], [0, h-th, tw, h], [w-tw, h-th, w, h]]
        assert windows == row["windows"]
        dx, dy = Fraction((w-tw)*rw, tw), Fraction((h-th)*rh, th)
        assert dx == Fraction(*row["shift_x_fraction"])
        assert dy == Fraction(*row["shift_y_fraction"])
        for s in (1, 2, 4, 8):
            aligned = dx.denominator == dy.denominator == 1 and dx.numerator % s == dy.numerator % s == 0
            assert aligned == row["aligned"][str(s)]
            counts[str(s)] += int(aligned)
    chosen = sorted(rows, key=lambda r: hashlib.sha256(
        f"{r['stem']}:shared-prefix-lattice-v1".encode()).digest())[:16]
    assert [r["stem"] for r in chosen] == plan["selected_stems"] == [r["stem"] for r in pixels]
    channels = error_sum = nonzero = exact = maximum = 0
    for row, probe in zip(chosen, pixels):
        path = project_path(f"data/visdrone/images/train/{row['stem']}.jpg")
        assert file_hash(path) == probe["image_sha256"]
        image = cv2.imread(str(path))
        assert list(image.shape[1::-1]) == row["size"]
        rw, rh = row["resized_content"]
        tw, th = row["crop_size"]
        contents, offsets = [], []
        for x1, y1, x2, y2 in row["windows"]:
            contents.append(cv2.resize(image[y1:y2, x1:x2].copy(), (rw, rh), interpolation=cv2.INTER_LINEAR))
            offsets.append([round(x1*Fraction(rw, tw)), round(y1*Fraction(rh, th))])
        assert offsets == probe["rounded_atlas_offsets"]
        assert probe["letterbox_replay"] == [True]*4
        assert len(probe["comparisons"]) == 6
        assert {tuple(p["pair"]) for p in probe["comparisons"]} == {
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
        for pair in probe["comparisons"]:
            i, j = pair["pair"]
            # Build common global coordinate vectors independently of slice arithmetic.
            xs = np.intersect1d(np.arange(offsets[i][0]+2, offsets[i][0]+rw-2),
                                np.arange(offsets[j][0]+2, offsets[j][0]+rw-2))
            ys = np.intersect1d(np.arange(offsets[i][1]+2, offsets[i][1]+rh-2),
                                np.arange(offsets[j][1]+2, offsets[j][1]+rh-2))
            a = contents[i][np.ix_(ys-offsets[i][1], xs-offsets[i][0])]
            b = contents[j][np.ix_(ys-offsets[j][1], xs-offsets[j][0])]
            delta = np.abs(a.astype(np.int32)-b.astype(np.int32))
            assert hashlib.sha256(a.tobytes()).hexdigest() == pair["first_sha256"]
            assert hashlib.sha256(b.tobytes()).hexdigest() == pair["second_sha256"]
            expected = {
                "pixels": len(xs)*len(ys), "channels": delta.size,
                "nonzero_pixels": int((delta.sum(axis=2) > 0).sum()),
                "nonzero_channels": int((delta != 0).sum()),
                "absolute_error_sum": int(delta.sum()), "max_absolute_error": int(delta.max()),
            }
            for name, value in expected.items():
                assert pair[name] == value, (probe["stem"], i, j, name)
            channels += delta.size
            error_sum += expected["absolute_error_sum"]
            nonzero += expected["nonzero_channels"]
            exact += int(expected["nonzero_channels"] == 0)
            maximum = max(maximum, expected["max_absolute_error"])
    metrics = read("metrics.json")
    assert metrics["eligible_images"] == counts
    assert metrics["nonzero_channel_fraction"] == nonzero/channels
    assert metrics["mean_absolute_uint8_error"] == error_sum/channels
    assert metrics["exact_pair_comparisons"] == exact
    assert metrics["max_absolute_uint8_error"] == maximum
    minimum = plan["registration"]["policy"]["minimum_train_fraction_all_four_views_same_stride2_lattice"]
    assert (counts["2"]/len(rows) >= minimum) == result["decision"]["checks"]["stride2_coverage"]
    verification = {
        "all6471_metadata_shapes_hashes_windows_resize_phases_independently_recomputed": True,
        "all16_deterministic_image_selection_and_image_hashes_verified": True,
        "all96_overlap_coordinate_sets_pixel_hashes_and_error_statistics_independently_recomputed": True,
        "aggregate_and_geometry_gate_recomputed": True,
        "all_sources_archive_and_installed_preprocessing_hashes_verified": True,
        "frozen_detector_weight_hash_verified": True,
        "additional_image_decodes": 16, "detector_forwards": 0,
        "verifier_sha256": file_hash(Path(__file__)),
        "limits": "Preprocessing/pixel necessary condition only. No inference, timing or accuracy verification.",
    }
    (output/"independent_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    return verification


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
