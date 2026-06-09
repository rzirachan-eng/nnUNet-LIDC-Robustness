"""
step3_prepare_nnunet.py
=======================
将 2_Stratified_Data 中有标签的病例汇聚为 nnU-Net V2 标准数据集，
并设置环境变量，输出 plan / train / inference 指令。

输出结构:
  nnUNet_raw/Dataset800_LIDC/
    imagesTr/    LIDC_CASE_XXX_0000.nii.gz
    labelsTr/    LIDC_CASE_XXX.nii.gz
    dataset.json

环境要求:
  - PyTorch 2.6.0+cu124  （已安装）
  - nnU-Net V2 2.5.2      （已安装）
  - RTX 4060 Laptop (7 GB) → 推荐 2D 训练

用法:
  1. python step3_prepare_nnunet.py          # 首次准备数据
  2. nnUNetv2_plan_and_preprocess 800 -planner nnUNetPlannerResEncM
  3. nnUNetv2_train 800 2d all              # 训练 2D (适合 7GB VRAM)
  4. python step4_run_inference.py           # 推理
"""

import os
import sys
import json
import shutil
import logging
from collections import defaultdict

# ==========================================
# 用户配置
# ==========================================
stratified_dir = "./2_Stratified_Data"
nnunet_base = "./nnUNet_data"  # nnU-Net 数据根目录

# 四个分层组
GROUPS = ["Thin_Slice", "Thick_Slice", "Smooth_Kernel", "Sharp_Kernel"]

# ==========================================
# 日志
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("step3_prepare_nnunet.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def setup_env():
    """设置 nnU-Net 环境变量"""
    base = os.path.abspath(nnunet_base)
    raw = os.path.join(base, "raw")
    preprocessed = os.path.join(base, "preprocessed")
    results = os.path.join(base, "results")

    os.makedirs(raw, exist_ok=True)
    os.makedirs(preprocessed, exist_ok=True)
    os.makedirs(results, exist_ok=True)

    os.environ["nnUNet_raw"] = raw
    os.environ["nnUNet_preprocessed"] = preprocessed
    os.environ["nnUNet_results"] = results

    log.info(f"nnUNet_raw:         {raw}")
    log.info(f"nnUNet_preprocessed: {preprocessed}")
    log.info(f"nnUNet_results:      {results}")

    return raw, preprocessed, results


def assemble_dataset(raw_dir, dataset_name="Dataset800_LIDC"):
    """
    从 2_Stratified_Data 各分层组中扫描有标签的病例，
    汇聚到 nnU-Net raw 数据集。

    Returns:
        dict: 数据集统计信息
    """
    dataset_dir = os.path.join(raw_dir, dataset_name)
    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")

    # 清除已有数据
    if os.path.exists(dataset_dir):
        log.warning(f"删除已有数据集: {dataset_dir}")
        shutil.rmtree(dataset_dir, ignore_errors=True)

    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)

    case_idx = 1
    stats = {"total_images": 0, "total_labels": 0, "by_group": defaultdict(int)}

    for group in GROUPS:
        group_img_dir = os.path.join(stratified_dir, group, "images")
        group_lbl_dir = os.path.join(stratified_dir, group, "labels")

        if not os.path.isdir(group_lbl_dir):
            log.warning(f"  {group}: labels 目录不存在，跳过")
            continue

        # 找到同时有图像和标签的病例
        label_files = sorted([
            f for f in os.listdir(group_lbl_dir)
            if f.endswith(".nii.gz")
        ])

        for lbl_file in label_files:
            # 标签文件名: LIDC_CASE_XXX.nii.gz
            base_name = lbl_file.replace(".nii.gz", "")
            img_file = f"{base_name}_0000.nii.gz"

            img_src = os.path.join(group_img_dir, img_file)
            lbl_src = os.path.join(group_lbl_dir, lbl_file)

            if not os.path.exists(img_src):
                log.debug(f"  ⏭️ {base_name}: 图像缺失, 跳过")
                continue

            # 复制到 nnU-Net 数据集 (使用自增编号)
            new_name = f"LIDC_{case_idx:03d}"
            shutil.copy2(img_src, os.path.join(images_tr, f"{new_name}_0000.nii.gz"))
            shutil.copy2(lbl_src, os.path.join(labels_tr, f"{new_name}.nii.gz"))

            stats["total_images"] += 1
            stats["total_labels"] += 1
            stats["by_group"][group] += 1
            case_idx += 1

    # 生成 dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0,
            "nodule": 1,
        },
        "numTraining": stats["total_labels"],
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }

    json_path = os.path.join(dataset_dir, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)

    log.info(f"dataset.json → {json_path}")
    return stats


def main():
    log.info("=" * 60)
    log.info("  nnU-Net V2 数据准备")
    log.info("=" * 60)

    # 1. 环境变量
    raw_dir, preprocessed_dir, results_dir = setup_env()

    # 2. 汇聚数据集
    log.info("\n[Phase 1] 汇聚有标签病例到 nnU-Net 格式...")
    stats = assemble_dataset(raw_dir)

    log.info(f"  总病例数: {stats['total_labels']}")
    for g in GROUPS:
        if g in stats["by_group"]:
            log.info(f"    {g}: {stats['by_group'][g]}")

    # 3. 输出下一步指令
    dataset_id = "800"
    log.info("\n" + "=" * 60)
    log.info("  后续步骤")
    log.info("=" * 60)
    log.info(f"""
  ── 第 1 步：验证数据集完整性 ──
  nnUNetv2_plan_and_preprocess {dataset_id} --verify_dataset_integrity

  ── 第 2 步：Plan & Preprocess ──
  nnUNetv2_plan_and_preprocess {dataset_id} -planner nnUNetPlannerResEncM

  ── 第 3 步：训练 (根据 VRAM 选择) ──

    推荐 (RTX 4060 7GB, 安全):  2D 训练
    nnUNetv2_train {dataset_id} 2d all

    如果 VRAM 充足 (>12GB):  3D 全分辨率
    nnUNetv2_train {dataset_id} 3d_fullres all

    中等 VRAM (8-12GB):  3D 低分辨率
    nnUNetv2_train {dataset_id} 3d_lowres all

  ── 第 4 步：推理 ──
  python step4_run_inference.py
  或直接使用命令行:
  nnUNetv2_predict -i <input_folder> -o <output_folder> -d {dataset_id} -c 2d
""")

    # 4. 写入环境变量文件供后续脚本使用
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnunet_env.bat")
    with open(env_file, "w") as f:
        f.write(f"@echo off\n")
        f.write(f"set nnUNet_raw={raw_dir}\n")
        f.write(f"set nnUNet_preprocessed={preprocessed_dir}\n")
        f.write(f"set nnUNet_results={results_dir}\n")
    log.info(f"  环境变量批处理文件: {env_file}")

    log.info("\n" + "=" * 60)
    log.info("  数据准备完成！请执行上面的命令继续。")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
