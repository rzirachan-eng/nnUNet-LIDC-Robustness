# Medical AI Robustness: Pulmonary Nodule Segmentation

# 医学AI鲁棒性：肺结节分割

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![nnU-Net V2](https://img.shields.io/badge/nnU--Net-V2.5.2-green.svg)](https://github.com/MIC-DKFZ/nnUNet)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

> **Robustness evaluation of nnU-Net V2 for pulmonary nodule segmentation under varying CT acquisition parameters.**
>
> **nnU-Net V2 在不同 CT 采集参数下肺结节分割鲁棒性评估。**

---

## Overview / 概述

This project systematically evaluates how CT acquisition parameters — **slice thickness** and **reconstruction kernel** — affect AI-based pulmonary nodule segmentation performance using nnU-Net V2 on the LIDC-IDRI dataset.

本项目系统评估 CT 采集参数——**层厚**和**重建核**——如何影响基于 nnU-Net V2 和 LIDC-IDRI 数据集的 AI 肺结节分割性能。

| Group / 分组 | Description / 描述 | Cases / 病例 | Mean Dice | Conclusion / 结论 |
|:---|:---|:---|:---|:---|
| Sharp_Kernel | 1.25–2.5mm + sharp kernel | 10 | **0.813** | **Best performance / 最优** |
| Smooth_Kernel | 1.25–2.5mm + soft kernel | 47 | 0.660 | Moderate & variable / 中高但波动 |
| Thin_Slice | ≤1.25mm slices | 34 | 0.465 | High variance / 高方差 |
| Thick_Slice | >2.5mm slices | 4 | 0.384 | Low statistics / 样本不足 |

**Key Finding / 关键发现:** Orthogonal two-track analysis reveals **Sharp_Kernel significantly outperforms Smooth_Kernel** (Dice: p=0.047, effect size r=+0.404, medium), while slice thickness alone shows no significant effect (Thin vs Thick: p=0.490). Reconstruction kernel choice — not slice thickness — is the dominant factor driving AI segmentation accuracy.

---

## Project Structure / 项目结构

```
Medical_AI_Robustness/
├── 1_Raw_Data/               # Raw DICOM sequences (not in release)
├── 2_Stratified_Data/        # Stratified NIfTI volumes (not in release)
├── 4_AI_Outputs/             # Inference predictions (not in release)
├── 5_Evaluation_Metrics/     # Evaluation results & charts
├── nnUNet_data/              # nnU-Net intermediate data (not in release)
├── scripts/                  # Pipeline scripts
│   ├── step1_sort_and_convert.py    # DICOM → NIfTI conversion
│   ├── step2_generate_labels.py     # Label generation via pylidc
│   ├── step3_prepare_nnunet.py      # nnU-Net dataset assembly
│   ├── step4_run_inference.py       # Batch inference
│   ├── step5_final_evaluation.py    # Evaluation & orthogonal statistics
│   └── run_training_api.py          # Training via Python API
├── evaluation/               # Evaluation outputs
│   ├── evaluation_results.csv       # Per-case Dice & HD95
│   ├── statistical_results.csv      # Orthogonal two-track tests
│   ├── dice_comparison_boxplot.png  # 4-group Dice boxplot
│   ├── hd95_comparison_boxplot.png  # 4-group HD95 boxplot
│   ├── dice_orthogonal_tracks.png   # Two-track Dice with p-values
│   └── hd95_orthogonal_tracks.png   # Two-track HD95 with p-values
├── Project_Final_Report.md   # Full bilingual project report
└── README.md                 # This file
```

---

## Requirements / 环境依赖

| Package / 包 | Version / 版本 | Purpose / 用途 |
|:---|:---|:---|
| Python | 3.9+ | Runtime |
| PyTorch | 2.6.0+cu124 | Deep learning framework |
| nnU-Net V2 | 2.5.2 | Segmentation framework |
| nibabel | ≥5.0 | NIfTI I/O |
| pydicom | ≥2.4 | DICOM I/O |
| pylidc | ≥0.2.8 | LIDC annotation database |
| medpy | ≥0.5 | Dice & HD95 metrics |
| scipy | ≥1.10 | Statistical tests |
| seaborn | ≥0.12 | Visualization |
| matplotlib | ≥3.7 | Plotting |
| tqdm | ≥4.65 | Progress bars |
| numpy | ≥1.24 | Array operations |

**Install / 安装:**

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install nnunetv2 nibabel pydicom medpy scipy seaborn matplotlib tqdm

# LIDC annotation tools (requires ~800MB download on first run)
pip install pylidc
```

---

## Reproduction Guide / 复现指南

### Step 1: Data Acquisition / 数据获取

Download LIDC-IDRI from [TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/). Place DICOM sequences in `1_Raw_Data/` with LIDC-IDRI patient ID subdirectories.

从 [TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/) 下载 LIDC-IDRI。将 DICOM 序列放入 `1_Raw_Data/`，按 LIDC-IDRI 患者 ID 组织子目录。

### Step 2: DICOM Sorting & NIfTI Conversion / DICOM 分拣与转换

```bash
python scripts/step1_sort_and_convert.py
```

Output: `2_Stratified_Data/{Thin_Slice,Thick_Slice,Smooth_Kernel,Sharp_Kernel}/images/LIDC_CASE_XXX_0000.nii.gz`

### Step 3: Label Generation / 标签生成

```bash
# Requires pylidc (will auto-download ~800MB annotation DB on first run)
python scripts/step2_generate_labels.py
```

Output: `2_Stratified_Data/{Group}/labels/LIDC_CASE_XXX.nii.gz`

### Step 4: nnU-Net Preparation / nnU-Net 数据准备

```bash
python scripts/step3_prepare_nnunet.py
nnUNetv2_plan_and_preprocess 800 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess 800 -planner nnUNetPlannerResEncM
```

### Step 5: Training / 训练

```bash
# Via CLI (recommended for 8GB VRAM)
nnUNetv2_train 800 2d 0 --npz

# Or via Python API (supports checkpoint resume)
python scripts/run_training_api.py
```

**Hardware note:** RTX 4060 8GB requires batch_size=4 for 2D training. 3D full-resolution is not feasible on this GPU.

### Step 6: Inference / 推理

```bash
python scripts/step4_run_inference.py --fold 0 --chk checkpoint_best.pth
```

Output: `4_AI_Outputs/{Group}_Pred/LIDC_CASE_XXX.nii.gz`

### Step 7: Evaluation / 评估

```bash
python scripts/step5_final_evaluation.py
```

Output:
- `evaluation/evaluation_results.csv` — per-case Dice & HD95
- `evaluation/statistical_results.csv` — pairwise comparisons
- `evaluation/dice_comparison_boxplot.png` — Dice boxplot
- `evaluation/hd95_comparison_boxplot.png` — HD95 boxplot

---

## Results Summary / 结果摘要

### Overall Metrics / 总体指标

| Group | N | Dice Mean ± SD | HD95 Median (mm) |
|:---|:---|:---|:---|
| Sharp_Kernel | 10 | 0.813 ± 0.128 | 3.92 |
| Smooth_Kernel | 47 | 0.660 ± 0.235 | 69.10 |
| Thin_Slice | 34 | 0.465 ± 0.323 | 131.61 |
| Thick_Slice | 4 | 0.384 ± 0.363 | 120.25 |

### Orthogonal Two-Track Analysis / 正交双轨分析

| Track | Comparison | Metric | p-value | Effect Size | Significant |
|:---|:---|:---|:---|:---|:---|
| A: Slice Thickness | Thin vs Thick | Dice | 0.490 | −0.221 (small) | No |
| A: Slice Thickness | Thin vs Thick | HD95 | 0.668 | +0.140 (small) | No |
| B: Kernel | Smooth vs Sharp | Dice | **0.047** | +0.404 (medium) | **Yes** |
| B: Kernel | Smooth vs Sharp | HD95 | 0.087 | −0.349 (medium) | No |

**Conclusion:** Sharp_Kernel significantly outperforms Smooth_Kernel in Dice (p=0.047). Slice thickness alone has no significant effect.

### Best Predicted Cases / 最佳预测病例

- **CASE_055** (Smooth_Kernel): Dice = **0.955**
- **CASE_085** (Smooth_Kernel): Dice = **0.942**
- **CASE_051** (Smooth_Kernel): Dice = **0.936**
- **CASE_004** (Smooth_Kernel): Dice = **0.935**
- **CASE_075** (Sharp_Kernel):  Dice = **0.919**

### Training Summary / 训练摘要

| Metric | Value |
|:---|:---|
| Framework | nnU-Net V2 2D (fold 0) |
| Training cases | 76 |
| Validation cases | 19 |
| Best Pseudo Dice (validation) | **0.7775** @ Epoch 981 |
| Final Pseudo Dice (validation) | 0.8156 @ Epoch 999 |
| Training time | ~118 min (epochs 900–999) |
| GPU | RTX 4060 Laptop 8GB |

---

## Citation / 引用

If you use this work, please cite both this repository and the underlying tools:

```bibtex
@misc{medical_ai_robustness_2026,
  title   = {Medical AI Robustness: nnU-Net Pulmonary Nodule Segmentation Under CT Acquisition Variability},
  author  = {Medical AI Robustness Project},
  year    = {2026},
  url     = {https://github.com/your-org/Medical_AI_Robustness}
}

@article{isensee2024nnu,
  title   = {nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation},
  author  = {Isensee, Fabian and Wald, Tassilo and Ulrich, Constantin and Baumgartner, Michael and Roy, Saikat and Maier-Hein, Klaus and Jaeger, Paul F},
  journal = {arXiv preprint arXiv:2404.09556},
  year    = {2024}
}

@article{armato2011lung,
  title   = {The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A Completed Reference Database of Lung Nodules on CT Scans},
  author  = {Armato III, Samuel G and McLennan, Geoffrey and Bidaut, Luc and McNitt-Gray, Michael F and Meyer, Charles R and Reeves, Anthony P and others},
  journal = {Medical Physics},
  volume  = {38},
  number  = {2},
  pages   = {915--931},
  year    = {2011}
}
```

---

## License / 许可证

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact / 联系方式

For questions or collaborations, please open an issue on this repository.

如有问题或合作意向，请在此仓库提交 Issue。

---

*Project completed June 2026. See `Project_Final_Report.md` for the full bilingual report.*
*项目完成于 2026 年 6 月。完整中英双语报告见 `Project_Final_Report.md`。*
