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

import torch

from monai.metrics.metric import Metric


class KIDMetric(Metric):
    """
    Kernel Inception Distance (KID). KID measures the squared maximum mean discrepancy between the distributions
    of real and generated feature vectors, estimated with a polynomial kernel and averaged over random subsets of
    the features. Compared to FID, the estimator is unbiased and its variance can be estimated from the subsets,
    which makes it a standard companion metric to FID when evaluating image synthesis and translation models.

    Based on: Bińkowski, M., Sutherland, D.J., Arbel, M., Gretton, A. "Demystifying MMD GANs."
    https://arxiv.org/abs/1801.01401

    The inputs for this metric should be two groups of feature vectors (with format (number of images, number of
    features)) extracted from a pretrained network, the same convention as `FIDMetric`. Originally, it was
    proposed to use the activations of the pool_3 layer of an Inception v3 pretrained on ImageNet, however other
    networks pretrained on medical datasets can be used as well (for example, RadImageNet for 2D and MedicalNet
    for 3D images).

    The kernel and MMD computations are a port-with-attribution from torchmetrics (Apache License 2.0),
    https://github.com/Lightning-AI/torchmetrics, see `torchmetrics/image/kid.py`.

    Requested in https://github.com/Project-MONAI/MONAI/issues/9151.

    Args:
        subset_size: number of feature vectors in each random subset used by the KID estimator. It is clamped to
            the number of available feature vectors if it is larger.
        num_subsets: number of random subsets to average the estimator over.
        degree: degree of the polynomial kernel.
        gamma: scale of the polynomial kernel, defaults to `1.0 / num_features` when None (as in torchmetrics).
        coef: constant term (coefficient) of the polynomial kernel.
        return_std: if True, `__call__` returns a tuple with the KID mean and standard deviation over the
            subsets, otherwise only the mean is returned.
        generator: optional `torch.Generator` used for the random subset draws, for reproducibility. When None,
            the global torch random number generator is used (matching the torchmetrics reference behaviour).
    """

    def __init__(
        self,
        subset_size: int = 1000,
        num_subsets: int = 100,
        degree: int = 3,
        gamma: float | None = None,
        coef: float = 1.0,
        return_std: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.subset_size = subset_size
        self.num_subsets = num_subsets
        self.degree = degree
        self.gamma = gamma
        self.coef = coef
        self.return_std = return_std
        self.generator = generator

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return get_kid_score(
            y_pred,
            y,
            subset_size=self.subset_size,
            num_subsets=self.num_subsets,
            degree=self.degree,
            gamma=self.gamma,
            coef=self.coef,
            return_std=self.return_std,
            generator=self.generator,
        )


def get_kid_score(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    subset_size: int = 1000,
    num_subsets: int = 100,
    degree: int = 3,
    gamma: float | None = None,
    coef: float = 1.0,
    return_std: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Computes the KID score metric on a batch of feature vectors, as the mean (and optionally the standard
    deviation) of the unbiased squared MMD with a polynomial kernel, over `num_subsets` random subsets of
    `subset_size` feature vectors.

    Args:
        y_pred: feature vectors extracted from a pretrained network run on generated images.
        y: feature vectors extracted from a pretrained network run on images from the real data distribution.
        subset_size: number of feature vectors in each random subset, clamped to the number of available samples.
        num_subsets: number of random subsets to average the estimator over.
        degree: degree of the polynomial kernel.
        gamma: scale of the polynomial kernel, defaults to `1.0 / num_features` when None.
        coef: constant term of the polynomial kernel.
        return_std: whether to also return the standard deviation of the estimator over the subsets.
        generator: optional `torch.Generator` for reproducible subset draws, defaults to the global torch RNG.
    """
    if y.ndimension() > 2 or y_pred.ndimension() > 2:
        raise ValueError("Inputs should have (number images, number of features) shape.")
    if y.shape[1] != y_pred.shape[1]:
        raise ValueError(f"Inputs should have the same number of features, got {y.shape[1]} and {y_pred.shape[1]}.")

    # the estimator cannot draw subsets larger than the number of available samples
    subset_size = int(min(subset_size, y.shape[0], y_pred.shape[0]))
    if subset_size < 2:
        raise ValueError("KID metric requires at least two samples in y and y_pred.")

    kid_scores: list[torch.Tensor] = []
    for _ in range(num_subsets):
        real_subset = y[torch.randperm(y.shape[0], generator=generator)[:subset_size]]
        fake_subset = y_pred[torch.randperm(y_pred.shape[0], generator=generator)[:subset_size]]
        kid_scores.append(
            maximum_mean_discrepancy(
                poly_kernel(real_subset, real_subset, degree=degree, gamma=gamma, coef=coef),
                poly_kernel(real_subset, fake_subset, degree=degree, gamma=gamma, coef=coef),
                poly_kernel(fake_subset, fake_subset, degree=degree, gamma=gamma, coef=coef),
            )
        )
    kid_scores_t = torch.stack(kid_scores)
    if return_std:
        return kid_scores_t.mean(), kid_scores_t.std(unbiased=False)
    return kid_scores_t.mean()


def poly_kernel(
    f1: torch.Tensor, f2: torch.Tensor, degree: int = 3, gamma: float | None = None, coef: float = 1.0
) -> torch.Tensor:
    """Computes the polynomial kernel `k(x, y) = (gamma * <x, y> + coef) ** degree` between two sets of feature
    vectors.

    Args:
        f1: feature vectors of shape (number of samples, number of features).
        f2: feature vectors of shape (number of samples, number of features).
        degree: degree of the polynomial kernel.
        gamma: scale of the polynomial kernel, defaults to `1.0 / f1.shape[1]` when None.
        coef: constant term of the polynomial kernel.
    """
    if gamma is None:
        gamma = 1.0 / f1.shape[1]
    return (f1 @ f2.T * gamma + coef) ** degree


def maximum_mean_discrepancy(k_xx: torch.Tensor, k_xy: torch.Tensor, k_yy: torch.Tensor) -> torch.Tensor:
    """Computes the unbiased squared maximum mean discrepancy (MMD^2) between two distributions from their
    kernel matrices, excluding the diagonals of the within-sample kernels `k_xx` and `k_yy` (Refs. Eq. 3 of
    Bińkowski et al. 2018). `k_xx` and `k_yy` are assumed to have the same size, as obtained from equally sized
    samples.

    Args:
        k_xx: kernel matrix between the samples of the first distribution.
        k_xy: kernel matrix between the samples of the two distributions.
        k_yy: kernel matrix between the samples of the second distribution.
    """
    m = k_xx.shape[0]
    diag_x = torch.diag(k_xx)
    diag_y = torch.diag(k_yy)
    kt_xx_sums = k_xx.sum(dim=-1) - diag_x
    kt_yy_sums = k_yy.sum(dim=-1) - diag_y
    k_xy_sums = k_xy.sum(dim=0)
    value = (kt_xx_sums.sum() + kt_yy_sums.sum()) / (m * (m - 1))
    value -= 2 * k_xy_sums.sum() / (m**2)
    return value
