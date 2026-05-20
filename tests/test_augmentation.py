"""Unit tests for seismic_cnn.augmentation."""

import numpy as np
import pytest

from seismic_cnn.augmentation import add_gaussian_noise, apply_combo, COMBOS


class TestGaussianNoise:
    def test_output_shape_preserved(self):
        data = np.zeros((3, 2000), dtype=np.float32)
        out  = add_gaussian_noise(data)
        assert out.shape == data.shape

    def test_output_clipped(self):
        data = np.ones((3, 2000), dtype=np.float32)
        out  = add_gaussian_noise(data, sigma=1.0)
        assert out.max() <= 1.0
        assert out.min() >= -1.0

    def test_output_differs_from_input(self):
        np.random.seed(0)
        data = np.zeros((3, 2000), dtype=np.float32)
        out  = add_gaussian_noise(data, sigma=1e-3)
        assert not np.allclose(data, out)


class TestApplyCombo:
    def test_single_fn(self):
        data = np.zeros((3, 100), dtype=np.float32)
        fns, _ = COMBOS[0]
        out = apply_combo(data, fns)
        assert out.shape == data.shape

    def test_does_not_mutate_input(self):
        data = np.ones((3, 100), dtype=np.float32)
        original = data.copy()
        apply_combo(data, [add_gaussian_noise])
        np.testing.assert_array_equal(data, original)
