import unittest

import numpy as np
import torch

from sdcl.evaluation.instance_moments import coefficients, coverage, folded, weight_maps


class InstanceMomentsTests(unittest.TestCase):
    def test_capture_unique_conv_before_shared_inplace_activation(self):
        conv1, conv2 = torch.nn.Conv2d(1, 1, 1), torch.nn.Conv2d(1, 1, 1)
        activation = torch.nn.SiLU(inplace=True)
        captured = {}
        handle = conv1.register_forward_hook(
            lambda module, args, output: captured.update(z=output.detach().clone()))
        x = torch.ones((1, 1, 2, 2))
        expected = conv1(x).detach().clone()
        activation(conv1(x))
        activation(conv2(torch.ones((1, 1, 3, 3))))
        torch.testing.assert_close(captured["z"], expected, atol=0, rtol=0)
        handle.remove()
    def test_fractional_cells(self):
        np.testing.assert_array_equal(coverage([2, 2, 6, 6], side=2),
                                      np.array([[.25, 0], [0, 0]]))

    def test_equal_instances_not_equal_pixels(self):
        metadata = dict(original_size=[960, 960], ignore_boxes_xyxy=[], valid_objects=[
            dict(box_xyxy=[40, 40, 48, 48]), dict(box_xyxy=[80, 80, 96, 96])])
        maps, count = weight_maps(metadata, [0, 0, 960, 960], "test", 0)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(maps["instance"].sum(), 2)
        self.assertAlmostEqual(maps["tiny_pixel"].sum(), 5)
        np.testing.assert_array_equal(np.sort(maps["instance"].ravel()),
                                      np.sort(maps["shuffle"].ravel()))

    def test_matches_direct_bn_running_update(self):
        rng = np.random.default_rng(69)
        gamma = np.array([.2, -.7, 2.])
        beta, old_mean, fit_mean = rng.normal(size=(3, 3))
        old_var, fit_var = np.exp(rng.normal(size=(2, 3)))
        eps, rho = .001, .03
        mean_z = beta+gamma*(fit_mean-old_mean)/np.sqrt(old_var+eps)
        variance_z = gamma**2*fit_var/(old_var+eps)
        a, b = coefficients(mean_z, variance_z, gamma, beta, old_var, eps, rho)
        x = rng.normal(size=(5, 3))
        old_z = gamma*(x-old_mean)/np.sqrt(old_var+eps)+beta
        direct = gamma*(x-((1-rho)*old_mean+rho*fit_mean))/np.sqrt(
            (1-rho)*old_var+rho*fit_var+eps)+beta
        np.testing.assert_allclose(old_z*a+b, direct, atol=1e-14, rtol=1e-14)

    def test_zero_gamma_and_zero_update(self):
        g = np.array([0., .5])
        a, b = coefficients(np.ones(2), np.ones(2), g, np.zeros(2), np.ones(2), .001, rho=0)
        np.testing.assert_array_equal(a, np.ones(2))
        np.testing.assert_array_equal(b, np.zeros(2))
        w, bias = folded(np.ones((2, 2, 1, 1), np.float16), np.zeros(2, np.float16), a, b)
        np.testing.assert_array_equal(w, np.ones_like(w))
        np.testing.assert_array_equal(bias, np.zeros_like(bias))


if __name__ == "__main__":
    unittest.main()
