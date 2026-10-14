import numpy as np

from sdcl.evaluation.texture_deletion import donors, erase, patch_rectangles


def test_mean_fill_changes_only_own_pixels_and_preserves_shape():
    image = np.arange(8*10*3, dtype=np.uint8).reshape(8, 10, 3)
    altered, texture, fill = erase(image, [1, 2, 4, 6])
    expected = image.copy()
    expected[2:6, 1:4] = np.rint(image[2:6, 1:4].mean((0, 1))).astype(np.uint8)
    np.testing.assert_array_equal(altered, expected)
    np.testing.assert_array_equal(image, np.arange(8*10*3, dtype=np.uint8).reshape(8, 10, 3))
    np.testing.assert_allclose(texture[:3], image[2:6, 1:4].mean((0, 1)))
    np.testing.assert_array_equal(fill, expected[2, 1])


def test_sham_priority_size_and_nonoverlap_at_boundaries():
    own, sham = patch_rectangles([15.2, 5.2, 18.2, 9.2], 20, 20)
    np.testing.assert_array_equal(own, [15, 5, 19, 10])
    np.testing.assert_array_equal(sham, [7, 5, 11, 10])
    np.testing.assert_array_equal(own[2:]-own[:2], sham[2:]-sham[:2])
    assert patch_rectangles([0, 0, 20, 20], 20, 20) is None


def test_donors_share_class_but_never_image():
    frames = []
    for stem, classes in (("a", [1, 1, 2]), ("b", [1, 2]), ("c", [3])):
        reference = np.zeros((len(classes), 6))
        reference[:, 5] = classes
        frames.append(dict(stem=stem, reference=reference, selected=np.arange(len(classes))))
    mapping = donors(frames)
    assert mapping[(2, 0)] is None
    for (i, j), donor in mapping.items():
        if donor is not None:
            di, dj = donor
            assert i != di
            assert frames[i]["reference"][j, 5] == frames[di]["reference"][dj, 5]
