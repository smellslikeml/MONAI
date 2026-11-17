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
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot

__all__ = ["SCNPLoss"]


def _scnp_transform(
    input: torch.Tensor, target: torch.Tensor, spatial_dims: int, neighborhood_size: int
) -> torch.Tensor:
    """
    Transform the logits by penalizing each pixel's worst same-class neighbor: every foreground pixel
    (``target == 1``) inherits the smallest logit among its same-class neighbors, and every background
    pixel (``target == 0``) the largest logit among its background neighbors, within a square/cubic
    neighborhood obtained with ``max_pool2d``/``max_pool3d``.
    """
    if spatial_dims != 2 and spatial_dims != 3:
        raise ValueError(f"spatial_dims must be either 2 or 3, got {spatial_dims}.")

    if input.shape != target.shape:
        raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

    # the reference implementation masks out the entries of the other class with a hard-coded offset of
    # 9999; deriving the offset from the input instead (one past its largest absolute logit) makes it
    # always representable in the input dtype, so that e.g. fp16 logits beyond ±9999 are masked as well
    offset = input.detach().abs().max() + 1
    padding = neighborhood_size // 2
    fg = input * target + offset * (1 - target)
    bg = input * (1 - target) - offset * target
    if spatial_dims == 2:
        # MinPooling in the foreground (background entries become ~offset, losing the negated max-pooling)
        t1 = -F.max_pool2d(-fg, kernel_size=neighborhood_size, stride=1, padding=padding)
        # MaxPooling in the background (foreground entries become ~-offset, losing the max-pooling)
        t2 = F.max_pool2d(bg, kernel_size=neighborhood_size, stride=1, padding=padding)
    else:
        t1 = -F.max_pool3d(-fg, kernel_size=neighborhood_size, stride=1, padding=padding)
        t2 = F.max_pool3d(bg, kernel_size=neighborhood_size, stride=1, padding=padding)

    return t1 * target + t2 * (1 - target)


class SCNPLoss(_Loss):
    """
    Wrapper class around a main loss function which first transforms the network logits with
    Same-Class Neighbor Penalization (SCNP): every foreground pixel inherits the smallest logit of its
    same-class neighbors, and every background pixel the largest logit of its background neighbors,
    within a square/cubic neighborhood. The base criterion is then evaluated on the transformed logits,
    so that predictions inconsistent with their same-class neighbors are penalized even when each pixel
    looks correct in isolation.

    SCNP is stateless (two pooling operations and a mask blend) and natively supports 2D and 3D inputs.
    Multi-class inputs are handled channel-wise, each channel acting as a binary foreground/background task.

    The base ``loss`` owns the activation (``sigmoid``/``softmax``), ``include_background`` and ``reduction``
    settings, which are applied to the transformed logits and are not re-implemented here. Since SCNP
    requires the full one-hot target (including the background channel), this wrapper owns ``to_onehot_y``
    and the base loss should therefore be created with its own ``to_onehot_y=False``.

    The original paper:

        Valverde, J. M. et al. SCNP: Topology Accuracy by Penalizing Neighbor Pixels.
        CVPR 2026. arXiv:2603.18671

    Re-derived from the MIT-licensed reference implementation by J. Miguel Valverde:
    https://github.com/jmlipman/SCNP-SameClassNeighborPenalization

    See Also:
        - :py:class:`monai.losses.DeepSupervisionLoss`
        - :py:class:`monai.losses.MaskedLoss`
    """

    def __init__(
        self,
        loss: _Loss,
        spatial_dims: int | None = None,
        neighborhood_size: int = 3,
        to_onehot_y: bool = False,
        include_base_loss: bool = False,
    ) -> None:
        """
        Args:
            loss: main loss instance applied to the transformed logits, e.g. ``DiceLoss(sigmoid=True)``.
                Its activation, ``include_background`` and ``reduction`` settings are delegated to it.
            spatial_dims: number of spatial dimensions, 2 or 3. If None (default), it is inferred from
                ``input`` as ``input.dim() - 2``.
            neighborhood_size: side of the square/cubic pooling neighborhood, a positive odd integer.
                Defaults to 3. With 1 the transform is the identity and the loss matches the wrapped
                base loss.
            to_onehot_y: whether to convert the ``target`` into the one-hot format, using the number of
                classes inferred from `input` (``input.shape[1]``). Defaults to False, i.e. ``target``
                is already one-hot and has the same shape as ``input``.
            include_base_loss: if True, the base loss evaluated on the raw logits is added to the loss
                on the transformed logits, as in the auxiliary variant of the paper. Defaults to False.

        Raises:
            ValueError: When ``neighborhood_size`` is not a positive odd integer.
        """
        super().__init__()
        if neighborhood_size < 1 or neighborhood_size % 2 == 0:
            raise ValueError(f"neighborhood_size must be a positive odd integer, got {neighborhood_size}.")
        self.loss = loss
        self.spatial_dims = spatial_dims
        self.neighborhood_size = neighborhood_size
        self.to_onehot_y = to_onehot_y
        self.include_base_loss = include_base_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes, with raw logits
                (before any activation, which is applied by the wrapped base loss).
            target: the shape should be BNH[WD] in the one-hot format, or B1H[WD] with integer
                class indices if ``to_onehot_y=True``.

        Returns:
            torch.Tensor: the base loss evaluated on the SCNP-transformed logits (plus the base loss
                evaluated on the raw logits if ``include_base_loss=True``).

        Example:
            >>> import torch
            >>> from monai.losses import DiceLoss, SCNPLoss
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C, size=(B, H, W)).long()
            >>> loss = SCNPLoss(DiceLoss(softmax=True), to_onehot_y=True)
            >>> result = loss(input, target_idx[:, None, ...])
        """
        if self.to_onehot_y:
            target = one_hot(target, num_classes=input.shape[1])
        spatial_dims = self.spatial_dims if self.spatial_dims is not None else input.dim() - 2
        z_tilde = _scnp_transform(input, target, spatial_dims, self.neighborhood_size)
        scnp_loss: torch.Tensor = self.loss(z_tilde, target)  # type: ignore[no-any-return]
        if self.include_base_loss:
            scnp_loss = scnp_loss + self.loss(input, target)
        return scnp_loss


scnp_loss = SCNPLoss
