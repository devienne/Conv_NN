"""Unit tests for seismic_cnn.models.convnetquake."""

import numpy as np
import pytest


class TestConvNetQuake:
    def test_output_shape(self):
        from seismic_cnn.models.convnetquake import build_model
        model = build_model(n_samples=2000, n_channels=3)
        batch = np.zeros((4, 2000, 3), dtype=np.float32)
        out = model.predict(batch, verbose=0)
        assert out.shape == (4, 1)

    def test_output_in_0_1(self):
        from seismic_cnn.models.convnetquake import build_model
        model = build_model(n_samples=2000, n_channels=3)
        batch = np.random.randn(2, 2000, 3).astype(np.float32)
        out = model.predict(batch, verbose=0)
        assert (out >= 0).all() and (out <= 1).all()

    def test_parameter_count(self):
        from seismic_cnn.models.convnetquake import build_model
        model = build_model(n_samples=2000, n_channels=3)
        n_params = model.count_params()
        # Expect roughly 22k parameters (original paper); allow ±50%
        assert 10_000 < n_params < 35_000
