# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import numpy as np
import torch

from monai.metrics import KIDMetric, get_kid_score
from monai.metrics.kid import maximum_mean_discrepancy, poly_kernel
from monai.utils import min_version, optional_import

tm, has_tm = optional_import("torchmetrics", "1.0.0", min_version)
tm_kid, _ = optional_import("torchmetrics.image.kid")


def _poly_mmd_oracle(
    real: torch.Tensor, fake: torch.Tensor, degree: int = 3, gamma: float | None = None, coef: float = 1.0
) -> torch.Tensor:
    """Independent transcription of the polynomial-kernel unbiased squared MMD estimator used by KID
    (Bińkowski et al. 2018, the same formula as the torchmetrics reference implementation)."""

    if gamma is None:
        gamma = 1.0 / real.shape[1]

    def kernel(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        return (f1 @ f2.T * gamma + coef) ** degree

    n, m = real.shape[0], fake.shape[0]
    k_xx = kernel(real, real)
    k_yy = kernel(fake, fake)
    k_xy = kernel(real, fake)
    term_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1))
    term_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (m * (m - 1))
    term_xy = 2.0 * k_xy.sum() / (n * m)
    return term_xx + term_yy - term_xy


class TestKIDMetric(unittest.TestCase):

    def test_identical_inputs(self):
        torch.manual_seed(0)
        x = torch.randn(64, 32)
        x = x / x.norm(dim=1, keepdim=True)
        metric = KIDMetric(subset_size=32, num_subsets=20, generator=torch.Generator().manual_seed(42))
        results = metric(x, x)
        self.assertTrue(torch.isfinite(results).all())
        np.testing.assert_allclose(results.detach().cpu().numpy(), 0.0, atol=5e-2)

    def test_different_distributions(self):
        torch.manual_seed(0)
        x = torch.randn(128, 32)
        metric = KIDMetric(subset_size=64, num_subsets=20, generator=torch.Generator().manual_seed(7))
        same = metric(x, x)
        shifted = metric(x + 3.0, x)
        self.assertTrue(torch.isfinite(shifted).all())
        self.assertGreater(shifted.item(), 0.0)
        self.assertGreater(shifted.item(), same.item())

    def test_reproducible_subsets(self):
        torch.manual_seed(1)
        y_pred, y = torch.randn(100, 16) + 0.5, torch.randn(100, 16)
        results = [
            KIDMetric(subset_size=50, num_subsets=10, generator=torch.Generator().manual_seed(123))(y_pred, y)
            for _ in range(2)
        ]
        np.testing.assert_allclose(results[0].detach().cpu().numpy(), results[1].detach().cpu().numpy(), atol=1e-10)
        # the module-level getter is consistent with the metric class for the same subset draws
        from_class = KIDMetric(subset_size=50, num_subsets=10, generator=torch.Generator().manual_seed(123))(y_pred, y)
        from_func = get_kid_score(
            y_pred, y, subset_size=50, num_subsets=10, generator=torch.Generator().manual_seed(123)
        )
        np.testing.assert_allclose(from_class.item(), from_func.item(), atol=1e-10)

    def test_input_dimensions(self):
        with self.assertRaises(ValueError):
            KIDMetric()(torch.ones([3, 3, 144, 144]), torch.ones([3, 3, 145, 145]))

    def test_feature_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            KIDMetric()(torch.ones([8, 16]), torch.ones([8, 32]))

    def test_min_samples(self):
        with self.assertRaises(ValueError):
            KIDMetric()(torch.ones([1, 16]), torch.ones([1, 16]))

    def test_subset_size_clamped(self):
        # a subset size larger than the number of samples is clamped, so every subset covers the full
        # feature sets and KID deterministically matches the oracle computed on all samples at once
        torch.manual_seed(2)
        y = torch.randn(48, 16).double()
        y_pred = torch.randn(48, 16).double() + 0.5
        results = KIDMetric(subset_size=1000, num_subsets=4)(y_pred, y)
        expected = _poly_mmd_oracle(y, y_pred)
        np.testing.assert_allclose(results.detach().cpu().numpy(), expected.item(), atol=1e-9)
        # every subset is the full sample, hence zero variance over the subsets
        mean, std = get_kid_score(y_pred, y, subset_size=1000, num_subsets=4, return_std=True)
        np.testing.assert_allclose(mean.item(), expected.item(), atol=1e-9)
        np.testing.assert_allclose(std.item(), 0.0, atol=1e-9)

    def test_helpers_parity(self):
        torch.manual_seed(3)
        f1 = torch.randn(16, 24).double()
        f2 = torch.randn(16, 24).double()
        k_xx = poly_kernel(f1, f1)
        k_xy = poly_kernel(f1, f2)
        k_yy = poly_kernel(f2, f2)
        # the kernel matrices are symmetric and match an explicit gamma
        self.assertTrue(torch.allclose(k_xx, k_xx.T))
        np.testing.assert_allclose(poly_kernel(f1, f2, gamma=1.0 / 24).numpy(), k_xy.numpy(), atol=1e-12)
        mmd = maximum_mean_discrepancy(k_xx, k_xy, k_yy)
        np.testing.assert_allclose(mmd.item(), _poly_mmd_oracle(f1, f2).item(), atol=1e-10)

    @unittest.skipUnless(has_tm, "Requires torchmetrics")
    def test_parity_torchmetrics(self):
        torch.manual_seed(4)
        real = torch.randn(200, 64).double()
        fake = torch.randn(200, 64).double() + 0.2
        metric_tm = tm_kid.KernelInceptionDistance(
            feature=torch.nn.Identity(), subsets=10, subset_size=50, degree=3, gamma=None, coef=1.0
        )
        # feed the pre-extracted feature tensors directly into the metric's internal buffers,
        # bypassing its image-based inception feature extractor
        metric_tm.real_features.append(real)
        metric_tm.fake_features.append(fake)
        # both implementations draw subsets from the global torch RNG, in the same order (real then fake)
        torch.manual_seed(123)
        tm_mean, tm_std = metric_tm.compute()
        torch.manual_seed(123)
        ours_mean, ours_std = get_kid_score(fake, real, subset_size=50, num_subsets=10, return_std=True)
        np.testing.assert_allclose(ours_mean.item(), tm_mean.item(), atol=1e-7)
        np.testing.assert_allclose(ours_std.item(), tm_std.item(), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
