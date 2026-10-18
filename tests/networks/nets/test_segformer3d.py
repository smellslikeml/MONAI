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

import torch

from monai.networks import eval_mode
from monai.networks.nets import SegFormer3D

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_SEGFORMER3D = [
    # paper default configuration, single-channel volume
    [{"in_channels": 1, "out_channels": 2}, (1, 1, 32, 32, 32), (1, 2, 32, 32, 32)],
    # batched multi-channel input and multiple output classes
    [{"in_channels": 4, "out_channels": 3}, (2, 4, 32, 32, 32), (2, 3, 32, 32, 32)],
    # smaller custom encoder/decoder dimensions on a larger volume
    [
        {
            "in_channels": 4,
            "out_channels": 3,
            "embed_dims": (16, 32, 64, 128),
            "num_heads": (1, 2, 4, 8),
            "depths": (1, 1, 2, 1),
            "decoder_head_embedding_dim": 64,
        },
        (2, 4, 64, 64, 64),
        (2, 3, 64, 64, 64),
    ],
    # alternative spatial reduction ratios with regularization enabled
    [
        {
            "in_channels": 2,
            "out_channels": 1,
            "sr_ratios": (8, 4, 2, 1),
            "dropout": 0.1,
            "attn_drop_rate": 0.1,
            "drop_path_rate": 0.1,
        },
        (1, 2, 32, 32, 32),
        (1, 1, 32, 32, 32),
    ],
    # spatial size that is not a multiple of the first stage stride (positional-free design)
    [{"in_channels": 1, "out_channels": 2}, (1, 1, 40, 40, 40), (1, 2, 40, 40, 40)],
]

TEST_CASE_SEGFORMER3D_BACKWARD = [
    [{"in_channels": 1, "out_channels": 2}, (2, 1, 32, 32, 32)],
    [
        {"in_channels": 2, "out_channels": 3, "embed_dims": (8, 16, 32, 64), "num_heads": (1, 2, 4, 8)},
        (1, 2, 32, 32, 32),
    ],
]


class TestSegFormer3D(unittest.TestCase):
    def test_shape(self):
        for input_param, input_shape, expected_shape in TEST_CASE_SEGFORMER3D:
            with self.subTest(**input_param):
                net = SegFormer3D(**input_param).to(device)
                with eval_mode(net):
                    result = net(torch.randn(input_shape).to(device))
                    self.assertEqual(result.shape, expected_shape, msg=str(input_param))

    def test_shape_train_eval(self):
        for input_param, input_shape, expected_shape in TEST_CASE_SEGFORMER3D[:2]:
            with self.subTest(**input_param):
                net = SegFormer3D(**input_param).to(device)
                net.train()
                result = net(torch.randn(input_shape).to(device))
                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.shape, expected_shape, msg=str(input_param))

                net.eval()
                with torch.no_grad():
                    result = net(torch.randn(input_shape).to(device))
                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.shape, expected_shape, msg=str(input_param))

    def test_backward(self):
        for input_param, input_shape in TEST_CASE_SEGFORMER3D_BACKWARD:
            with self.subTest(**input_param):
                net = SegFormer3D(**input_param).to(device)
                net.train()
                net(torch.randn(input_shape).to(device)).sum().backward()
                param = dict(net.named_parameters())["stages.0.blocks.0.attn.q.weight"]
                self.assertIsNotNone(param.grad, msg=str(input_param))
                self.assertGreater(param.grad.abs().sum().item(), 0.0, msg=str(input_param))

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            SegFormer3D(spatial_dims=2)
        with self.assertRaises(ValueError):
            SegFormer3D(embed_dims=(32, 64, 160), num_heads=(1, 2, 5, 8))


if __name__ == "__main__":
    unittest.main()
