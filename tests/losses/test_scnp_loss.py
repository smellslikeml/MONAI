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
from parameterized import parameterized

from monai.losses import DiceCELoss, DiceLoss, SCNPLoss
from monai.networks import one_hot
from tests.test_utils import test_script_save

INPUT_2D = torch.tensor([[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]])
TARGET_IDX = torch.tensor([[[[1, 1, 0]]]])  # B1HW class indices
TARGET_2D = one_hot(TARGET_IDX, num_classes=2)
INPUT_3D = torch.tensor(
    [[[[[2.0, -1.0], [3.0, 0.0]], [[-2.0, 1.0], [0.0, 4.0]]], [[[5.0, -3.0], [1.0, 2.0]], [[-1.0, 0.0], [6.0, -4.0]]]]]
)
TARGET_3D = one_hot(torch.tensor([[[[[0, 1], [0, 1]], [[1, 0], [1, 0]]]]]), num_classes=2)

TEST_CASES = [
    # min/max pooling over the same-class neighborhood gives z_tilde = [[[2., 2., 3.]], [[4., 4., 6.]]]
    [
        {},
        DiceLoss,
        {"sigmoid": True},
        {"input": INPUT_2D, "target": TARGET_2D},
        0.347680,
    ],
    # integer target converted to one-hot by the wrapper, matches the case above
    [
        {"to_onehot_y": True},
        DiceLoss,
        {"sigmoid": True},
        {"input": INPUT_2D, "target": TARGET_IDX},
        0.347680,
    ],
    [
        {"neighborhood_size": 5, "to_onehot_y": True},
        DiceCELoss,
        {"sigmoid": True},
        {"input": INPUT_2D, "target": TARGET_IDX},
        1.448494,
    ],
    [
        {"spatial_dims": 3},
        DiceLoss,
        {"sigmoid": True},
        {"input": INPUT_3D, "target": TARGET_3D},
        0.649086,
    ],
]


def make_neighbor_data(hi: float = 8.0, lo: float = -8.0):
    """One-hot 2D target with a 3x3 class-1 square, and logits that are either consistent with it or
    contain a single badly predicted pixel inside the square."""
    target_idx = torch.zeros(1, 1, 5, 5, dtype=torch.long)
    target_idx[0, 0, 1:4, 1:4] = 1
    target = one_hot(target_idx, num_classes=2)
    consistent = lo + (hi - lo) * target
    inconsistent = consistent.clone()
    inconsistent[0, 1, 2, 2] = lo
    return target, consistent, inconsistent


class TestSCNPLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_result(self, scnp_kwargs, base_class, base_kwargs, input_data, expected_val):
        loss = SCNPLoss(base_class(**base_kwargs), **scnp_kwargs)
        result = loss(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)

    def test_scalar_and_backward(self):
        for shape in ((2, 3, 8, 8), (2, 3, 8, 8, 8)):
            input = torch.rand(*shape, requires_grad=True)
            target_idx = torch.randint(0, 3, (shape[0], 1, *shape[2:]))
            loss = SCNPLoss(DiceCELoss(sigmoid=True), to_onehot_y=True)(input, target_idx)
            self.assertEqual(loss.dim(), 0)
            self.assertTrue(torch.isfinite(loss).all())
            self.assertTrue(loss.requires_grad)
            loss.backward()
            self.assertIsNotNone(input.grad)
            self.assertTrue(torch.isfinite(input.grad).all())

    def test_shape_2d_3d(self):
        base = DiceLoss(softmax=True)
        for shape in ((2, 3, 7, 6), (2, 3, 7, 6, 5)):
            input = torch.rand(*shape)
            target_idx = torch.randint(0, 3, (shape[0], 1, *shape[2:]))
            inferred = SCNPLoss(base, to_onehot_y=True)(input, target_idx)
            explicit = SCNPLoss(base, spatial_dims=len(shape) - 2, to_onehot_y=True)(input, target_idx)
            self.assertEqual(inferred.dim(), 0)
            self.assertTrue(torch.isfinite(inferred).all())
            np.testing.assert_allclose(inferred.detach().numpy(), explicit.detach().numpy(), atol=1e-7)

    def test_reduction(self):
        for reduction in ("none", "mean", "sum"):
            base = DiceLoss(sigmoid=True, reduction=reduction)
            result = SCNPLoss(base)(INPUT_2D, TARGET_2D)
            self.assertEqual(result.shape, base(INPUT_2D, TARGET_2D).shape)
            if reduction != "none":
                self.assertEqual(result.dim(), 0)

    def test_to_onehot_y(self):
        base = DiceLoss(sigmoid=True)
        from_indices = SCNPLoss(base, to_onehot_y=True)(INPUT_2D, TARGET_IDX)
        from_onehot = SCNPLoss(base)(INPUT_2D, TARGET_2D)
        np.testing.assert_allclose(from_indices.detach().numpy(), from_onehot.detach().numpy(), atol=1e-7)

    def test_monotonicity(self):
        target, consistent, inconsistent = make_neighbor_data()
        for base in (DiceLoss(sigmoid=True), DiceCELoss(sigmoid=True)):
            loss = SCNPLoss(base)
            consistent_loss = loss(consistent, target)
            inconsistent_loss = loss(inconsistent, target)
            self.assertTrue(torch.isfinite(consistent_loss).all())
            # a neighbor-inconsistent prediction is penalized more than a consistent one
            self.assertLess(consistent_loss.item(), inconsistent_loss.item())
            # logits already uniform within each class are left unchanged by the transform
            np.testing.assert_allclose(consistent_loss.item(), base(consistent, target).item(), rtol=1e-6)
            self.assertGreater(inconsistent_loss.item(), base(inconsistent, target).item())

    @parameterized.expand([(2,), (3,)])
    def test_identity_small_neighborhood(self, spatial_dims):
        shape = (2, 3, 8, 8) if spatial_dims == 2 else (2, 3, 8, 8, 8)
        input = torch.rand(*shape)
        target = one_hot(torch.randint(0, 3, (shape[0], 1, *shape[2:])), num_classes=3)
        base = DiceLoss(softmax=True)
        result = SCNPLoss(base, spatial_dims=spatial_dims, neighborhood_size=1)(input, target)
        np.testing.assert_allclose(result.detach().numpy(), base(input, target).detach().numpy(), atol=1e-7)

    def test_include_base_loss(self):
        base = DiceLoss(sigmoid=True)
        result = SCNPLoss(base, include_base_loss=True)(INPUT_2D, TARGET_2D)
        expected = SCNPLoss(base)(INPUT_2D, TARGET_2D) + base(INPUT_2D, TARGET_2D)
        np.testing.assert_allclose(result.detach().numpy(), expected.detach().numpy(), rtol=1e-6)

    def test_mask_offset_fp16(self):
        # fp16 logits beyond the ±9999 offset of the reference implementation must still be masked
        # correctly: with a window covering the full width, the background of channel 0 keeps the
        # larger logit (-59968) instead of the masked foreground entry
        input = torch.tensor([[[[2.0, -60000.0, -59968.0]], [[-3.0, -59968.0, 59968.0]]]], dtype=torch.float16)
        target = one_hot(torch.tensor([[[[0, 1, 1]]]]), num_classes=2)
        z_tilde = torch.tensor([[[[2.0, -59968.0, -59968.0]], [[-3.0, -59968.0, -59968.0]]]])
        base = torch.nn.L1Loss()
        result = SCNPLoss(base)(input, target)
        self.assertTrue(torch.isfinite(result).all())
        np.testing.assert_allclose(result.detach().numpy(), base(z_tilde, target).numpy(), rtol=1e-5)

    def test_ill_neighborhood(self):
        for size in (0, 2, -3):
            with self.assertRaisesRegex(ValueError, "neighborhood_size"):
                SCNPLoss(DiceLoss(sigmoid=True), neighborhood_size=size)

    def test_ill_spatial_dims(self):
        for dims in (1, 4):
            with self.assertRaisesRegex(ValueError, "spatial_dims"):
                SCNPLoss(DiceLoss(sigmoid=True), spatial_dims=dims)(INPUT_2D, TARGET_2D)

    def test_ill_shape(self):
        with self.assertRaisesRegex(AssertionError, ""):
            SCNPLoss(DiceLoss(sigmoid=True))(INPUT_2D, TARGET_IDX)

    def test_script(self):
        loss = SCNPLoss(DiceLoss(sigmoid=True))
        test_input = torch.rand(2, 2, 8, 8)
        test_target = torch.zeros(2, 2, 8, 8)
        test_target[:, 0] = 1
        test_script_save(loss, test_input, test_target)


if __name__ == "__main__":
    unittest.main()
