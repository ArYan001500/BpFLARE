# External code

ModernTCN is run with its authors' official implementation (Luo and Wang, ICLR 2024), which is not redistributed
here. Clone it and check out the commit used in the paper:

```
git clone https://github.com/luodhhh/ModernTCN.git ModernTCN
cd ModernTCN
git checkout 56a9a2c018385cd5acef015378cae7f084d1b11c
```

The folder must be `external/ModernTCN`, so that `external/ModernTCN/ModernTCN-classification/models/ModernTCN.py`
exists. `scripts/bb_moderntcn.py` uses it unchanged.
