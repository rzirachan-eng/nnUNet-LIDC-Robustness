"""
step1_sort_and_convert.py — 优化版
=====================================
功能：将 1_Raw_Data 中的 LIDC-IDRI DICOM 序列按放射学参数分拣，
      并转换为 nnU-Net 兼容的 NIfTI 格式。

优化要点 (v2.0):
  1. 序列筛选：Modality==CT + 最低文件数阈值 50，过滤 Annotation/XML 文件夹
  2. 健壮性：缺失 Tag 安全回退，单文件损坏不中断全局流程
  3. 分层逻辑：基于 LIDC-IDRI 实际 Tag 分布优化 Thin/Thick/Sharp/Smooth 分类
  4. 并行加速：多进程 DICOM→NIfTI 转换，大幅提升吞吐量
  5. nnU-Net 规范：严格 _0000.nii.gz 命名 + 扁平文件结构
"""

import os
import sys
import logging
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
import nibabel as nib
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 用户配置
# ==========================================
raw_data_dir = "./1_Raw_Data"
sorted_data_dir = "./2_Stratified_Data"

# 并行工作进程数（留 1 核给系统）
NUM_WORKERS = max(1, cpu_count() - 1)

# DICOM 序列最低文件数（CT 序列通常 ≥80 张；XML 标注文件夹只有几个文件）
MIN_DCM_FILES = 50

# ==========================================
# 日志配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("step1_sort_and_convert.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)


# ==========================================
# 放射学分类规则（基于 LIDC-IDRI 实际 Tag 分布）
# ==========================================

# 层厚阈值
THIN_THRESHOLD = 1.25   # ≤ 此值为薄层
THICK_THRESHOLD = 2.5   # > 此值为厚层

# 锐化重建核关键词（LIDC-IDRI 中出现的：B70f, LUNG, FC10 等）
SHARP_KERNELS = {
    "b70", "b75", "b80",   # Siemens 锐化
    "lung",                 # GE 肺窗核 (锐化)
    "fc10", "fc30",         # 东芝/佳能 锐化
    "bone", "edge",         # 通用锐化关键词
    "70", "80",             # 数字后缀匹配
}

# 平滑重建核关键词（LIDC-IDRI 中出现的：B30f, B31s, FC01, STANDARD 等）
SMOOTH_KERNELS = {
    "b20", "b25", "b30", "b31",  # Siemens 平滑
    "b10", "b35",                  # Siemens 软/中等平滑
    "fc01", "fc03",                # 东芝/佳能 平滑
    "standard",                    # GE 标准核 (平滑)
    "soft",                        # 通用平滑
    "20", "30", "31",              # 数字后缀匹配
}


def classify_series(thickness, kernel):
    """
    根据层厚和重建核对 DICOM 序列进行放射学分类。
    
    策略：
      优先按层厚分 Thin / Thick。
      对于中间层厚 (1.25 < T ≤ 2.5)，使用重建核分 Sharp / Smooth。
      缺失参数或无法匹配时返回 None（跳过）。
    
    Returns:
        str | None: "Thin_Slice", "Thick_Slice", "Sharp_Kernel", "Smooth_Kernel", 或 None
    """
    if thickness is not None:
        try:
            t = float(thickness)
            if t <= THIN_THRESHOLD:
                return "Thin_Slice"
            if t > THICK_THRESHOLD:
                return "Thick_Slice"
        except (TypeError, ValueError):
            pass

    # 中间层厚 → 按重建核分类
    if kernel is not None:
        kernel_lower = str(kernel).lower().replace(" ", "")
        for sharp_kw in SHARP_KERNELS:
            if sharp_kw in kernel_lower:
                return "Sharp_Kernel"
        for smooth_kw in SMOOTH_KERNELS:
            if smooth_kw in kernel_lower:
                return "Smooth_Kernel"

    return None  # 不可分类 → 跳过


def dicom_series_to_nifti(dicom_dir, output_path):
    """
    将单个 DICOM 序列目录转换为 NIfTI (.nii.gz) 文件。
    
    实现步骤：
      1. 读取所有 .dcm 并排序（按 SliceLocation / ImagePositionPatient）
      2. 提取像素数据并应用 RescaleSlope/Intercept
      3. 构建 3D 仿射矩阵
      4. 用 nibabel 保存为压缩 NIfTI
    """
    # 1. 读取所有 DICOM 切片
    dcm_files = sorted([
        f for f in os.listdir(dicom_dir)
        if f.lower().endswith('.dcm')
    ])
    if not dcm_files:
        raise ValueError(f"目录 {dicom_dir} 中没有 .dcm 文件")

    slices = []
    for fname in dcm_files:
        filepath = os.path.join(dicom_dir, fname)
        try:
            ds = pydicom.dcmread(filepath, force=True)
            # 跳过非图像文件 (如 RTSTRUCT, SEG 等)
            if not hasattr(ds, 'pixel_array'):
                continue
            # 跳过非轴向切片 (localizer/scout)
            if hasattr(ds, 'ImageType') and 'LOCALIZER' in str(ds.ImageType).upper():
                continue
            slices.append(ds)
        except (InvalidDicomError, Exception):
            continue

    if len(slices) < 2:
        raise ValueError(f"有效切片不足: {len(slices)}")

    # 2. 按空间位置排序
    def _slice_position(ds):
        """提取切片在患者坐标系中的 z 位置"""
        if hasattr(ds, 'SliceLocation'):
            return float(ds.SliceLocation)
        if hasattr(ds, 'ImagePositionPatient'):
            ipp = ds.ImagePositionPatient
            if hasattr(ds, 'ImageOrientationPatient'):
                iop = ds.ImageOrientationPatient
                row_cos = np.array(iop[:3], dtype=np.float64)
                col_cos = np.array(iop[3:6], dtype=np.float64)
                normal = np.cross(row_cos, col_cos)
                return float(np.dot(np.array(ipp, dtype=np.float64), normal))
            return float(ipp[2])
        if hasattr(ds, 'InstanceNumber'):
            return float(ds.InstanceNumber)
        return 0.0

    slices.sort(key=_slice_position)

    # 3. 构建 3D 体积
    ref_ds = slices[0]
    rows = int(ref_ds.Rows)
    cols = int(ref_ds.Columns)
    num_slices = len(slices)

    # 检查切片尺寸一致性，跳过不一致的
    consistent_slices = []
    for ds in slices:
        if int(ds.Rows) == rows and int(ds.Columns) == cols:
            consistent_slices.append(ds)
    slices = consistent_slices
    num_slices = len(slices)

    volume = np.zeros((cols, rows, num_slices), dtype=np.float32)
    for i, ds in enumerate(slices):
        px = ds.pixel_array.astype(np.float32)
        # 应用 Rescale
        slope = float(getattr(ds, 'RescaleSlope', 1) or 1)
        intercept = float(getattr(ds, 'RescaleIntercept', 0) or 0)
        px = px * slope + intercept
        volume[:, :, i] = px

    # 4. 计算仿射矩阵
    # 像素间距
    px_spacing = getattr(ref_ds, 'PixelSpacing', [1.0, 1.0])
    dx, dy = float(px_spacing[0]), float(px_spacing[1])

    # 层间距
    if hasattr(ref_ds, 'SliceThickness') and len(slices) > 1:
        dz = float(ref_ds.SliceThickness)
    else:
        dz = abs(_slice_position(slices[1]) - _slice_position(slices[0])) if len(slices) > 1 else dx
    if dz <= 0:
        dz = abs(_slice_position(slices[-1]) - _slice_position(slices[0])) / (num_slices - 1) if num_slices > 1 else dx
    if dz <= 0:
        dz = dx

    # 方向余弦
    if hasattr(ref_ds, 'ImageOrientationPatient'):
        iop = np.array(ref_ds.ImageOrientationPatient, dtype=np.float64).reshape(2, 3)
        row_cos = iop[0]
        col_cos = iop[1]
        normal = np.cross(row_cos, col_cos)
    else:
        row_cos = np.array([1.0, 0.0, 0.0])
        col_cos = np.array([0.0, 1.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])

    # 原点
    if hasattr(ref_ds, 'ImagePositionPatient'):
        origin = np.array(ref_ds.ImagePositionPatient, dtype=np.float64)
    else:
        origin = np.array([0.0, 0.0, 0.0])

    # 构建 4x4 仿射矩阵 (RAS 坐标系)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = row_cos * dx
    affine[:3, 1] = col_cos * dy
    affine[:3, 2] = normal * dz
    affine[:3, 3] = origin

    # 5. 保存为 .nii.gz
    nifti_img = nib.Nifti1Image(volume, affine)
    nib.save(nifti_img, output_path)


def scan_patient(patient_id, raw_data_dir):
    """
    扫描单个患者目录，提取 DICOM 序列信息和放射学参数。
    
    Returns:
        dict | None: {
            "patient_id": str,
            "series_path": str,
            "thickness": float|None,
            "kernel": str|None,
            "modality": str,
            "num_files": int,
        }
    """
    patient_path = os.path.join(raw_data_dir, patient_id)
    if not os.path.isdir(patient_path):
        return None

    # 遍历子目录，找到合法的 CT 序列
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
        if len(dcm_files) < MIN_DCM_FILES:
            continue  # 跳过 XML/Annotation 等小型文件夹

        # 快速验证：读第一个 DICOM 检查 Modality
        first_dcm = os.path.join(root, dcm_files[0])
        try:
            ds = pydicom.dcmread(first_dcm, stop_before_pixels=True)
        except (InvalidDicomError, Exception):
            continue

        modality = str(ds.get('Modality', '')).upper()
        if modality not in ('CT', 'MR'):
            continue  # 非影像序列，跳过

        # 提取放射学参数
        thickness = None
        kernel = None
        try:
            if 'SliceThickness' in ds:
                thickness = float(ds.SliceThickness)
        except (TypeError, ValueError, AttributeError):
            pass
        try:
            if 'ConvolutionKernel' in ds:
                kernel = str(ds.ConvolutionKernel)
        except (TypeError, ValueError, AttributeError):
            pass

        return {
            "patient_id": patient_id,
            "series_path": root,
            "thickness": thickness,
            "kernel": kernel,
            "modality": modality,
            "num_files": len(dcm_files),
        }

    return None  # 未找到合法序列


def convert_single_case(args):
    """
    单病例转换任务（供多进程调用）。
    独立完成：分类 → 转换 NIfTI。
    
    Returns:
        dict: {"case_id": int, "group": str, "status": "ok"|"skip"|"error", "message": str}
    """
    case_id, series_info, sorted_data_dir = args
    target_series_path = series_info["series_path"]
    thickness = series_info["thickness"]
    kernel = series_info["kernel"]
    patient_id = series_info["patient_id"]

    result = {
        "case_id": case_id,
        "patient_id": patient_id,
        "group": "",
        "status": "skip",
        "message": "",
    }

    try:
        # ---- 放射学分类 ----
        selected_group = classify_series(thickness, kernel)
        if selected_group is None:
            result["message"] = (
                f"参数不典型 → 跳过 (T={thickness}, K={kernel})"
            )
            return result

        result["group"] = selected_group

        # ---- nnU-Net 兼容命名 ----
        output_name = f"LIDC_CASE_{case_id:03d}"
        output_dir = os.path.join(sorted_data_dir, selected_group, "images")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_name}_0000.nii.gz")

        # ---- DICOM → NIfTI（纯 pydicom + nibabel 实现）----
        dicom_series_to_nifti(target_series_path, output_path)

        result["status"] = "ok"
        result["message"] = (
            f"→ {selected_group}/{output_name}_0000.nii.gz "
            f"({series_info['num_files']} slices, T={thickness}mm, K={kernel})"
        )

    except Exception as e:
        result["status"] = "error"
        result["message"] = f"转换失败: {e}"
        log.error(f"[CASE_{case_id:03d}] {patient_id} 出错:\n{traceback.format_exc()}")

    return result


# ==========================================
# 主流程
# ==========================================
def main():
    log.info("=" * 60)
    log.info("  LIDC-IDRI 放射学分拣与转换 (优化版 v2.0)")
    log.info(f"  并行工作进程: {NUM_WORKERS}")
    log.info("=" * 60)

    # ---- 1. 创建目标文件夹 ----
    groups = ["Thin_Slice", "Thick_Slice", "Smooth_Kernel", "Sharp_Kernel"]
    for g in groups:
        os.makedirs(os.path.join(sorted_data_dir, g), exist_ok=True)

    # ---- 2. 扫描所有患者，提取 DICOM 序列信息 ----
    log.info("[Phase 1] 扫描 DICOM 序列...")
    patient_ids = sorted(
        f for f in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, f))
        and not f.startswith('.') and not f.startswith('_')
    )
    log.info(f"  发现 {len(patient_ids)} 个患者目录")

    valid_series = []  # 合法序列列表
    skip_reasons = defaultdict(int)

    for pid in tqdm(patient_ids, desc="扫描中", unit="patient"):
        info = scan_patient(pid, raw_data_dir)
        if info is None:
            skip_reasons["未找到合法CT序列"] += 1
            continue
        valid_series.append(info)

    log.info(f"  合法 CT 序列: {len(valid_series)}")
    for reason, count in skip_reasons.items():
        log.info(f"  跳过 ({reason}): {count}")

    # ---- 3. 多进程 DICOM → NIfTI 转换 ----
    log.info(f"\n[Phase 2] 并行转换 NIfTI (workers={NUM_WORKERS})...")

    # 按 case_id 编号
    tasks = [
        (i + 1, series, sorted_data_dir)
        for i, series in enumerate(valid_series)
    ]

    stats = {"ok": 0, "skip": 0, "error": 0}
    group_counts = defaultdict(int)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(convert_single_case, task): task[0] for task in tasks}
        with tqdm(total=len(futures), desc="转换中", unit="case") as pbar:
            for future in as_completed(futures):
                result = future.result()
                stats[result["status"]] += 1
                if result["status"] == "ok":
                    group_counts[result["group"]] += 1

                # 简要输出
                status_icon = {"ok": "✅", "skip": "⏭️", "error": "❌"}
                log.info(
                    f"{status_icon.get(result['status'], '?')} "
                    f"[{result['case_id']:03d}] {result['message']}"
                )
                pbar.update(1)

    # ---- 4. 汇总报告 ----
    log.info("\n" + "=" * 60)
    log.info("  转换完成!")
    log.info(f"  成功: {stats['ok']}  跳过: {stats['skip']}  错误: {stats['error']}")
    log.info("-" * 40)
    for g in groups:
        count = group_counts.get(g, 0)
        log.info(f"  {g}: {count} cases")
    log.info(f"\n  输出目录: {os.path.abspath(sorted_data_dir)}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()