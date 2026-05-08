# Attribution and Acknowledgments

This project builds upon and adapts code from several sources. We gratefully acknowledge the following:

## DyT (Transformers without Normalization)

**Source**: https://github.com/jiachenzhu/DyT  
**License**: MIT License (Copyright 2025 Jiachen Zhu)  
**Paper**: "Transformers without Normalization" (CVPR 2025)  
**Authors**: Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, Zhuang Liu

### Adapted Components

The following components were adapted from the DyT repository:

1. **Dataset Loading** (`utils/datasets.py`)
   - Adapted from `datasets.py` in the DyT repository
   - MIT License (Copyright 2025 Jiachen Zhu)
   - Modifications: removed debug prints, removed `assert`, added docstrings,
     type annotations, `ValueError` instead of `NotImplementedError`,
     non-mutating `crop_pct` handling, `getattr` safe-defaults throughout

2. **Training Loop** (`utils/training.py`)
   - Adapted from `engine.py` in the DyT repository
   - MIT License (Copyright 2025 Jiachen Zhu)
   - Modifications: rewritten variable naming, fixed LR condition precedence
     bug, `update_now` refactor, `class_acc` guard, f-strings throughout,
     `train_one_epoch_logit_distill` added (original contribution)

3. **Training Utilities** (`utils/helpers.py`)
   - Adapted from `utils.py` in the DyT repository
   - MIT License (Copyright 2025 Jiachen Zhu)
   - Modifications: private attribute naming (`_window`, `_total`, `_count`,
     `_fmt`), type annotations throughout, rewritten `MetricLogger.__getattr__`
     and `update`, rewritten `log_every` with f-strings, rewritten
     `init_distributed_mode` with `os.environ.update`, new `_suppress_non_master_prints`,
     new `WandbLogger` implementation; `TensorboardLogger`, `load_state_dict`,
     `save_model`, `auto_load_model`, `cosine_scheduler` removed as unused

4. **DyT Baseline Implementation** (`baselines/dyt.py`)
   - Copied from `dynamic_tanh.py` in original DyT repository
   - Included as a baseline for comparison
   - Minimal modifications for documentation
   - Original copyright: Meta Platforms, Inc. and affiliates

### Citation

If you use this work, please cite both this project and the original DyT paper:

```bibtex
@inproceedings{zhu2025dyt,
  title={Transformers without Normalization},
  author={Zhu, Jiachen and Chen, Xinlei and He, Kaiming and LeCun, Yann and Liu, Zhuang},
  booktitle={CVPR},
  year={2025}
}
```

## timm (PyTorch Image Models)

**Source**: https://github.com/huggingface/pytorch-image-models  
**License**: Apache License 2.0  
**Author**: Ross Wightman

We use timm extensively for:
- Model architectures (ViT, ConvNeXt, etc.)
- Data augmentation utilities
- Training utilities (ModelEma, accuracy metrics)

## JAX

**Source**: https://github.com/google/jax  
**License**: Apache License 2.0  
**Authors**: Google

JAX is used as the foundation for the Kozax genetic programming library.

## License Compatibility

All dependencies are licensed under permissive licenses (MIT, Apache 2.0) that allow for modification and redistribution with attribution. This project maintains license compatibility by:

1. Preserving original copyright notices in adapted code
2. Clearly documenting which components were adapted vs. written from scratch
3. Providing proper attribution in both code comments and this document
4. Licensing our original contributions under CC BY-NC-ND 4.0

## Kozax

**Source**: https://github.com/kc-ml2/kozax  
**License**: MIT License  
**Install**: `pip install kozax`

Kozax is an external JAX-based genetic programming library used to run GP evolution in this project. It is **not** an original contribution of this work.

## Our Contributions

The following components are original contributions of this project:

1. **GP Evolution Pipeline** - LayerNorm I/O extraction, fitness functions, operators
2. **GP-to-PyTorch Generation** - Code generation for evolved layers
3. **Finetuning Methodology** - Novel finetuning strategies for GP-evolved models

These components are licensed under CC BY-NC-ND 4.0 (Copyright 2026 Kieran Carrigg) — see LICENSE file.

## Third-Party Licenses

### MIT License (DyT Components)

```
MIT License

Copyright (c) 2025 Jiachen Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

**Last Updated**: April 25, 2026
