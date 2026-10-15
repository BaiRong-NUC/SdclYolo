import numpy as np

from sdcl.evaluation.neighbor_interference import (
    eligible_pairs, input_rectangle, perturb, query_pool, rectangles_overlap, sham_rectangles,
)


def test_rectangles_protect_two_pixel_margin_and_boundary_touch():
    boxes = np.array([[10, 10, 20, 20]])
    assert rectangles_overlap(np.array([21, 10, 25, 20]), boxes)
    assert not rectangles_overlap(np.array([22, 10, 26, 20]), boxes)


def test_reflections_keep_size_radius_and_avoid_all_objects():
    target, neighbor = np.array([40, 40, 50, 50]), np.array([58, 40, 68, 50])
    shams = sham_rectangles(target, neighbor, np.array([target, neighbor]), 100, 100)
    assert len(shams) == 1
    np.testing.assert_array_equal(shams[0], [22, 40, 32, 50])
    np.testing.assert_array_equal(shams[0][2:]-shams[0][:2], neighbor[2:]-neighbor[:2])


def test_fills_modify_only_patch_and_leave_target_input_pixels_unchanged():
    import cv2
    from ultralytics.data.augment import LetterBox
    image = np.random.default_rng(7).integers(0, 256, (160, 200, 3), dtype=np.uint8)
    original = image.copy()
    for fill in ("mean", "blur"):
        altered = perturb(image, [60, 40, 72, 50], fill)
        np.testing.assert_array_equal(altered[40:50, 40:50], image[40:50, 40:50])
        altered[40:50, 60:72] = image[40:50, 60:72]
        np.testing.assert_array_equal(altered, image)
        changed = perturb(image, [60, 40, 72, 50], fill)
        baseline_input = LetterBox((960, 960), auto=False)(image=image)
        changed_input = LetterBox((960, 960), auto=False)(image=changed)
        x1, y1, x2, y2 = input_rectangle([40, 40, 50, 50], 200, 160)
        np.testing.assert_array_equal(changed_input[y1:y2, x1:x2], baseline_input[y1:y2, x1:x2])
    np.testing.assert_array_equal(image, original)


def test_pair_selection_excludes_overlapping_neighbor_and_finds_safe_sham():
    metadata = dict(original_size=[100, 100], ignore_boxes_xyxy=[], valid_objects=[
        dict(box_xyxy=[40, 40, 42, 42], class_id=0),
        dict(box_xyxy=[47, 40, 49, 42], class_id=0)])
    assert eligible_pairs(metadata, "synthetic") == []
    # 4x4 at640-reference requires an image sufficiently large to remain tiny.
    metadata["original_size"] = [400, 400]
    for obj, box in zip(metadata["valid_objects"], ([140, 140, 148, 148], [160, 140, 168, 148])):
        obj["box_xyxy"] = box
    pairs = eligible_pairs(metadata, "synthetic")
    assert len(pairs) == 2
    for pair in pairs:
        assert not rectangles_overlap(np.array(pair["neighbor_rect"]),
                                      np.array([metadata["valid_objects"][pair["target"]]["box_xyxy"]]))


def test_query_lattice_is_fixed_p3_including_boundaries():
    assert len(query_pool([480, 480, 484, 484], 960, 960)) == 9
    pool = query_pool([0, 0, 4, 4], 960, 960)
    assert len(pool) == 4
    assert (pool >= 0).all() and (pool < 14400).all()
