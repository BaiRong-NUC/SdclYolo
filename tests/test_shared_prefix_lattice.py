import numpy as np

from sdcl.evaluation.shared_prefix_lattice import decide, geometry, independent_geometry, overlap_statistics


def test_integer_and_fractional_resampling_phases_are_distinct():
    aligned = geometry(1800, 1800)
    assert aligned["crop_size"] == [1000, 1000]
    assert all(aligned["aligned"].values())
    fractional = geometry(1920, 1080)
    assert not fractional["aligned"]["1"]
    assert not fractional["aligned"]["2"]


def test_same_pixel_grid_does_not_imply_same_convolution_stride_phase():
    # Crop-size rounding must be reproduced independently of phase decisions.
    assert geometry(1843, 1843)["crop_size"] == [1024, 1024]
    # Construct a valid size with integral shifts but a fractional stride8 phase.
    found = False
    for side in range(2, 2000):
        row = geometry(side, side)
        assert independent_geometry(side, side) == (row["crop_size"], row["resized_content"], row["aligned"])
        if row["aligned"]["1"] and not row["aligned"]["8"]:
            found = True
    assert found


def test_overlapping_content_compares_global_positions_and_excludes_edges():
    atlas = np.arange(12*18*3, dtype=np.uint16).reshape(12, 18, 3).astype(np.uint8)
    first, second = atlas[:, :12].copy(), atlas[:, 6:].copy()
    result = overlap_statistics(first, second, [0, 0], [6, 0])
    assert result["nonzero_channels"] == 0
    second[4, 3, 0] ^= 1
    result = overlap_statistics(first, second, [0, 0], [6, 0])
    assert result["nonzero_channels"] == 1
    assert result["absolute_error_sum"] == result["max_absolute_error"] == 1


def test_geometry_gate_never_promotes_coverage_without_replay_and_verification():
    assert decide(100, 79, True, True)["decision"].startswith("STOP")
    assert decide(100, 80, True, True)["decision"] == "PREFIX_PROTOTYPE_ALLOWED"
    assert decide(100, 100, False, True)["decision"].startswith("STOP")
    assert decide(100, 100, True, False)["decision"].startswith("STOP")
