"""ModernTCN backbone from the OFFICIAL implementation (Luo and Wang, ICLR 2024; github.com/luodhhh/ModernTCN,
commit 56a9a2c018385cd5acef015378cae7f084d1b11c, folder ModernTCN-classification), used unchanged through its
classification path (forward_feature -> GELU -> dropout -> flatten -> linear), with 3 outputs (SBP, DBP, MAP)
instead of class logits. Hyperparameters follow the authors' published configuration for SelfRegulationSCP2
(scripts/classification.sh), the UEA task whose input (1,152 samples) is closest to our windows (1,250 samples):
patch 32, stride 16, ffn ratio 4, blocks (1, 1), large kernels (51, 49), small kernels (5, 5), dims (64, 128),
backbone dropout 0.3, class dropout 0.1, downsample ratio 2 (default), no multi-scale head.
Importing this module registers it in s37_backbones.BACKBONES under the name "MODERNTCN"."""
import sys
from pathlib import Path

import torch.nn as nn

import s37_backbones as s37

OFFICIAL = Path(__file__).resolve().parents[1] / "external" / "ModernTCN" / "ModernTCN-classification"
if not (OFFICIAL / "models" / "ModernTCN.py").exists():
    raise FileNotFoundError(f"official ModernTCN code not found at {OFFICIAL}; see external/README.md")
if str(OFFICIAL) not in sys.path:
    sys.path.insert(0, str(OFFICIAL))
from models.ModernTCN import ModernTCN  # noqa: E402  (official code)


class ModernTCNOfficial(nn.Module):
    def __init__(self, dropout=s37.DROPOUT, seq_len=1250):
        super().__init__()
        self.core = ModernTCN(task_name="classification", patch_size=32, patch_stride=16, stem_ratio=6,
                              downsample_ratio=2, ffn_ratio=4, num_blocks=[1, 1], large_size=[51, 49],
                              small_size=[5, 5], dims=[64, 128], dw_dims=[256, 256], nvars=2,
                              small_kernel_merged=False, backbone_dropout=0.3, head_dropout=0.0,
                              use_multi_scale=False, revin=False, affine=False, subtract_last=False,
                              seq_len=seq_len, c_in=(2,), individual=False, target_window=3,
                              class_drop=0.1, class_num=3)
        self.core.head = None                         # unused forecasting head of the official class
        self.head = nn.Module()                       # exposes the output layer as head.net[-1] for the s37 recipe
        self.head.net = nn.Sequential(self.core.head_class)

    def forward(self, x):                             # x: [B, 2, 1250]
        return self.core(x)


s37.BACKBONES["MODERNTCN"] = ModernTCNOfficial
