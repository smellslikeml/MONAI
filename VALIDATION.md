# SegFormer3D — validation status and deferred plan

## Clean-room statement

`monai/networks/nets/segformer3d.py` is a **clean-room reimplementation written from the paper
text only** — Perera, Navard, Yilmaz, "SegFormer3D: an Efficient Transformer for 3D Medical
Image Segmentation", [arXiv:2404.10156](https://arxiv.org/abs/2404.10156). The authors'
reference repository (OSUPCVLab/SegFormer3D) is GPL-3.0 and incompatible with MONAI's
Apache-2.0; it was **not** fetched, read, or consulted. No GPL-derived code, variable names,
or structure were copied. Design lineage that the paper itself cites (SegFormer / Mix Vision
Transformer, Xie et al. 2021; Pyramid Vision Transformer, Wang et al. 2021) was followed as
described *in the SegFormer3D paper's own method section*.

## What was implemented (validated here, CPU)

- `SegFormer3D(nn.Module)` in `monai/networks/nets/segformer3d.py`, registered in
  `monai/networks/nets/__init__.py` and `docs/source/networks.rst`.
  (`monai/networks/__init__.py` does not re-export individual nets, so it is untouched.)
- Encoder: 4-stage hierarchical transformer. Overlap patch embedding via `Conv3d`
  (stage 1: k=7, s=4, p=3; stages 2-4: k=3, s=2, p=1) + `LayerNorm` over tokens; per-stage
  transformer blocks (pre-norm) with efficient self-attention — K/V spatially reduced by a
  stride-`r` `Conv3d` + `LayerNorm` with `sr_ratios = (4, 2, 1, 1)` exactly as stated in the
  paper — and Mix-FFN (`Linear -> DepthwiseConv3d(3x3x3) -> GELU -> Linear`, positional-free).
- Decoder: all-MLP head per the paper's steps 1-4 — per-stage 1x1x1 conv projection to a
  common embedding dim, trilinear upsampling to the stage-1 resolution, concatenation,
  1x1x1 conv fusion, 1x1x1 conv classifier; logits trilinearly upsampled to the input size.
- Constructor defaults follow the paper: the SegFormer/MiT efficient (b0-lineage)
  configuration the paper builds on — `embed_dims=(32, 64, 160, 256)`,
  `depths=(2, 2, 2, 2)`, `num_heads=(1, 2, 5, 8)`, `mlp_ratios=(4, 4, 4, 4)`,
  `sr_ratios=(4, 2, 1, 1)`, `decoder_head_embedding_dim=128`.

### Cross-check against the paper's reported model size (this machine, CPU, defaults)

| Metric | Paper | This implementation |
| --- | --- | --- |
| Parameters (4-channel BraTS-style input) | 4.5 M | 4.25 M |
| GFLOPs @ (1, 4, 128, 128, 128) | 17.5 | 17.8 |

The paper text does not tabulate per-stage embed dims / depths / heads; those were derived
from the MiT design the paper extends, then cross-checked against the paper's reported
parameter count and GFLOPs (both within ~6%). The residual gap on parameters most plausibly
lives in head-level details that the paper does not specify.

### CI-runnable tests

`tests/networks/nets/test_segformer3d.py` — CPU forward-pass shape checks over five
configurations (defaults, multi-channel/batched, custom dims, alternate `sr_ratios` with
dropout/stochastic-depth, non-stride-multiple input size), train/eval shape checks, backward
gradient smoke tests, and constructor validation errors. Uses stdlib `unittest.subTest`
rather than `parameterized` so it runs without the optional test dependency.

## Deferred to human/GPU validation (NOT part of this change)

1. **Accuracy parity on BraTS / Synapse / ACDC** — train the default configuration with the
   paper's protocol (AdamW lr 3e-5, linear warmup 4e-6 -> 4e-4, PolyLR decay, Dice + CE
   loss, batch size 4, 1000 epochs, 128^3 patches) on a GPU and compare mean Dice to the
   paper's Tables 2-4. A Colab notebook (A100/L4) is sufficient: at ~4M params / ~18 GFLOPs
   per volume the model trains on a single consumer GPU.
2. **GPL-parity behavioral cross-check** — a human (not this change) may load the reference
   implementation's published weights into this implementation where shapes align, or compare
   activation statistics on identical inputs, as an oracle for the remaining config details
   (per-stage dims/depths reconciliation against the paper's Figure 2, fuse-head details).
   Note the paper reports the SR ratios (4, 2, 1, 1) but no per-stage table; if the oracle
   disagrees on a knob, the fix is a constructor-argument change, not a structural one.
3. **Minor deltas to reconcile during (2)**: this implementation adds a `LayerNorm` + `GELU`
   after the decoder fusion (per the design brief; the paper's decoder equations are purely
   linear), and defaults all dropout rates to 0 because the paper specifies none.

Nothing in this change is stubbed or incomplete relative to its scope; items above are the
brief's explicitly deferred GPU/parity work.
