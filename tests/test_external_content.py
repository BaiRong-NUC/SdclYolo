import numpy as np

from sdcl.evaluation.external_content import crop_input, donor_indices


def test_own_crop_preserves_pixel_content_aspect_and_normalization():
    image = np.zeros((6, 8, 3), np.uint8)
    image[:, :, 2] = 255
    tensor, rectangle, canvas = crop_input(image, [1, 1, 5, 3], 1.)
    np.testing.assert_array_equal(rectangle, [1, 1, 5, 3])
    assert tensor.shape == (3, 224, 224)
    np.testing.assert_array_equal(canvas[55, 100], [114, 114, 114])
    np.testing.assert_array_equal(canvas[56, 100], [0, 0, 255])
    np.testing.assert_allclose(tensor[:, 100, 100], [(1-.485)/.229, -.456/.224, -.406/.225], atol=1e-6)


def test_context_rectangle_uses_prediction_center_and_outside_padding():
    image = np.zeros((6, 8, 3), np.uint8)
    _, rectangle, canvas = crop_input(image, [0, 0, 4, 4], 2.)
    np.testing.assert_array_equal(rectangle, [-2, -2, 6, 6])
    np.testing.assert_array_equal(canvas[0, 0], [114, 114, 114])
    np.testing.assert_array_equal(canvas[180, 180], [0, 0, 0])


def test_donor_control_excludes_same_image_and_preserves_class():
    frames = []
    for stem, classes in (("a", [1, 2]), ("b", [1, 1])):
        reference = np.zeros((len(classes), 6))
        reference[:, 5] = classes
        frames.append(dict(stem=stem, reference=reference, selected=np.arange(len(classes))))
    mapping = donor_indices(frames)
    assert mapping[(0, 1)] is None
    for (i, j), donor in mapping.items():
        if donor is not None:
            di, dj = donor
            assert di != i and frames[di]["reference"][dj, 5] == frames[i]["reference"][j, 5]
