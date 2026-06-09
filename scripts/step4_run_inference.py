"""
step4_run_inference.py
======================
使用训练好的 nnU-Net V2 模型对 2_Stratified_Data 中所有病例进行批量推理，
结果保存至 4_AI_Outputs，按分层组组织。

前提条件:
  1. 已完成训练: nnUNetv2_train 800 2d all (或至少 fold 0)
  2. 模型权重位于: nnUNet_data/results/Dataset800_LIDC/nnUNetTrainer__nnUNetPlans__2d/

输出结构:
  4_AI_Outputs/
    Thin_Slice_Pred/    LIDC_CASE_XXX.nii.gz
    Thick_Slice_Pred/   LIDC_CASE_XXX.nii.gz
    Smooth_Kernel_Pred/ LIDC_CASE_XXX.nii.gz
    Sharp_Kernel_Pred/  LIDC_CASE_XXX.nii.gz

硬件适配 (RTX 4060 Laptop 7GB):
  - 使用 2D 配置 (3D fullres 会 OOM)
  - --disable_tta (禁用测试时增强，省显存)
  - 单进程推理，避免并行显存竞争

用法:
  # 使用 fold 0 推理 (快速测试)
  python step4_run_inference.py --fold 0

  # 使用所有 fold 集成推理 (最佳精度)
  python step4_run_inference.py --fold all

  # 仅处理特定分组
  python step4_run_inference.py --fold 0 --groups Thin_Slice
"""

import os
import sys
import subprocess
import shutil
import argparse
import logging
from pathlib import Path

# ==========================================
# 配置
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STRATIFIED_DIR = os.path.join(PROJECT_ROOT, "2_Stratified_Data")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "4_AI_Outputs")
NNUNET_BASE = os.path.join(PROJECT_ROOT, "nnUNet_data")

DATASET_ID = "800"
DATASET_NAME = "Dataset800_LIDC"
CONFIG = "2d"  # RTX 4060 8GB 只能用 2D

# 分组名 → 输出子目录映射
GROUP_OUTPUT_MAP = {
    "Thin_Slice": "Thin_Slice_Pred",
    "Thick_Slice": "Thick_Slice_Pred",
    "Smooth_Kernel": "Smooth_Kernel_Pred",
    "Sharp_Kernel": "Sharp_Kernel_Pred",
}
GROUPS = list(GROUP_OUTPUT_MAP.keys())

# ==========================================
# 日志
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("step4_run_inference.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def setup_env():
    """设置 nnU-Net 环境变量"""
    raw = os.path.join(NNUNET_BASE, "raw")
    preprocessed = os.path.join(NNUNET_BASE, "preprocessed")
    results = os.path.join(NNUNET_BASE, "results")

    os.environ["nnUNet_raw"] = raw
    os.environ["nnUNet_preprocessed"] = preprocessed
    os.environ["nnUNet_results"] = results

    # 验证目录存在
    for name, path in [("raw", raw), ("preprocessed", preprocessed), ("results", results)]:
        if not os.path.isdir(path):
            log.error(f"nnU-Net {name} 目录不存在: {path}")
            return False
    return True


def check_model_exists(fold="0", chk="checkpoint_best.pth"):
    """检查训练好的模型是否存在 (支持指定 checkpoint)"""
    trainer = "nnUNetTrainer__nnUNetPlans__"
    model_dir = os.path.join(
        NNUNET_BASE, "results", DATASET_NAME, f"{trainer}{CONFIG}"
    )

    if fold == "all":
        missing = []
        for f in range(5):
            fold_dir = os.path.join(model_dir, f"fold_{f}")
            if not _find_checkpoint(fold_dir, chk):
                missing.append(str(f))
        if missing:
            log.error(f"缺少 fold(s): {missing}")
            log.error(f"模型目录: {model_dir}")
            log.error("请先运行训练: nnUNetv2_train 800 2d all")
            return False
        return True
    else:
        fold_dir = os.path.join(model_dir, f"fold_{fold}")
        ckpt = _find_checkpoint(fold_dir, chk)
        if ckpt:
            log.info(f"  找到 checkpoint: {ckpt}")
            return True
        log.error(f"未找到模型 checkpoint '{chk}': {fold_dir}")
        log.error(f"请先运行: nnUNetv2_train 800 2d {fold}")
        return False


def _find_checkpoint(fold_dir, preferred_chk="checkpoint_best.pth"):
    """查找 checkpoint 文件，优先使用指定 checkpoint"""
    if not os.path.isdir(fold_dir):
        return None
    # 先检查指定的 checkpoint
    ckpt_path = os.path.join(fold_dir, preferred_chk)
    if os.path.exists(ckpt_path):
        return preferred_chk
    # 回退查找其他 checkpoint
    for ckpt_name in ["checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"]:
        ckpt_path = os.path.join(fold_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            return ckpt_name
    return None


def run_inference_for_group(group, fold="0", chk=None):
    """
    对单个分层组运行推理。
    将 images/ 下所有 _0000.nii.gz 作为输入，
    预测结果保存到 4_AI_Outputs/{Group}_Pred/
    """
    group_img_dir = os.path.join(STRATIFIED_DIR, group, "images")
    if not os.path.isdir(group_img_dir):
        log.warning(f"  {group}: images 目录不存在，跳过")
        return 0

    # 检查是否有图像文件
    image_files = sorted([
        f for f in os.listdir(group_img_dir)
        if f.endswith("_0000.nii.gz")
    ])
    if not image_files:
        log.warning(f"  {group}: 无 _0000.nii.gz 文件，跳过")
        return 0

    # 输出目录
    output_dir = os.path.join(OUTPUT_BASE, f"{group}_Pred")
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"  {group}: {len(image_files)} 个病例 → {output_dir}")

    # 构建 nnUNetv2_predict 命令
    # RTX 4060 7GB 优化参数:
    #   --disable_tta      禁用测试时增强 (节省 ~8x 显存和时间)
    #   -npp 1             单进程预处理
    #   -nps 1             单进程分割
    #   --verbose          显示进度
    cmd = [
        "nnUNetv2_predict",
        "-i", group_img_dir,
        "-o", output_dir,
        "-d", DATASET_ID,
        "-c", CONFIG,
        "-f", str(fold),
        "--disable_tta",
        "-npp", "1",
        "-nps", "1",
        "--verbose",
    ]
    if chk:
        cmd.insert(11, "-chk")
        cmd.insert(12, chk)
        log.info(f"  使用 checkpoint: {chk}")

    log.info(f"  执行: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2小时超时
            env=os.environ.copy(),
        )

        if result.returncode != 0:
            log.error(f"  {group} 推理失败!")
            log.error(f"  STDERR: {result.stderr[-500:]}")
            return 0

        # 统计输出
        pred_files = [
            f for f in os.listdir(output_dir)
            if f.endswith(".nii.gz")
        ]
        log.info(f"  {group}: 生成 {len(pred_files)} 个预测文件")
        return len(pred_files)

    except subprocess.TimeoutExpired:
        log.error(f"  {group} 推理超时!")
        return 0
    except FileNotFoundError:
        log.error("  nnUNetv2_predict 命令未找到!")
        log.error("  请确保 nnU-Net V2 已安装且 PATH 中包含其 Scripts 目录")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="nnU-Net V2 批量推理 —— LIDC-IDRI 肺结节分割"
    )
    parser.add_argument(
        "--fold", type=str, default="0",
        help="使用哪个 fold 进行推理 (默认: 0, 可选: all, 0-4)"
    )
    parser.add_argument(
        "--groups", type=str, nargs="+", default=None,
        help="指定要处理的分组 (默认: 全部四组)"
    )
    parser.add_argument(
        "--chk", type=str, default="checkpoint_best.pth",
        choices=["checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_latest.pth"],
        help="指定使用的 checkpoint 权重文件 (默认: checkpoint_best.pth)"
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  nnU-Net V2 批量推理")
    log.info(f"  Dataset: {DATASET_NAME} (ID={DATASET_ID})")
    log.info(f"  Config:  {CONFIG}")
    log.info(f"  Fold:    {args.fold}")
    log.info(f"  Checkpoint: {args.chk}")
    log.info("=" * 60)

    # 1. 环境检查
    if not setup_env():
        sys.exit(1)

    # 2. 模型检查
    log.info("\n[Step 1] 检查模型权重...")
    if not check_model_exists(args.fold, args.chk):
        sys.exit(1)
    log.info("  模型权重 ✓")

    # 3. 选择分组
    groups = args.groups if args.groups else GROUPS
    log.info(f"\n[Step 2] 推理分组: {groups}")

    # 4. 逐组推理
    total_preds = 0
    for group in groups:
        log.info(f"\n--- {group} ---")
        n = run_inference_for_group(group, args.fold, args.chk)
        total_preds += n

    # 5. 汇总
    log.info("\n" + "=" * 60)
    log.info(f"  推理完成! 总计生成 {total_preds} 个预测文件")
    log.info(f"  输出目录: {OUTPUT_BASE}")
    for group in groups:
        pred_dir = os.path.join(OUTPUT_BASE, GROUP_OUTPUT_MAP.get(group, f"{group}_Pred"))
        if os.path.isdir(pred_dir):
            count = len([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])
            log.info(f"    {GROUP_OUTPUT_MAP.get(group, group)}: {count} files")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
