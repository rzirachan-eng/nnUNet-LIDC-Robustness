"""
step2_generate_labels.py
=========================
基于 pylidc 标注数据库，生成 LIDC-IDRI 肺部结节多数共识分割标签，
并与 CT 图像配对输出为 nnU-Net 兼容格式。

功能流程：
  1. 连接 pylidc 标注数据库（首次运行自动下载 ~800MB）
  2. 匹配本地 1_Raw_Data 中已下载的 DICOM 数据
  3. 按放射学参数（层厚/重建核）分类到 4 个分层组
  4. 提取 3D CT 体积 + 生成多数共识 (consensus) 结节掩码
  5. 保存到 2_Stratified_Data/{Group}/images/ 和 labels/ 子目录

依赖: pylidc, nibabel, numpy, tqdm
用法: python step2_generate_labels.py
"""

import os
import sys
import logging
import traceback
from collections import defaultdict

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
import nibabel as nib
from tqdm import tqdm

# ==========================================
# 用户配置
# ==========================================
raw_data_dir = "./1_Raw_Data"
output_dir = "./2_Stratified_Data"

# 共识水平：4 位医生中至少 clevel*4 位标注才算结节区域
# 0.5 = 至少 2/4 认同；0.75 = 至少 3/4 认同
CONSENSUS_LEVEL = 0.5

# ==========================================
# 日志配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("step2_generate_labels.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ==========================================
# 依赖检查
# ==========================================
def check_dependencies():
    """检查必要库是否安装，缺失时给出安装指引"""
    missing = []
    for mod_name, pkg_name in [
        ("pylidc", "pylidc"),
        ("nibabel", "nibabel"),
        ("numpy", "numpy"),
        ("pydicom", "pydicom"),
    ]:
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(f"  pip install {pkg_name}")

    if missing:
        log.error("缺少必要依赖，请运行以下命令安装：")
        for m in missing:
            log.error(m)
        log.error(
            "\n注意：pylidc 首次运行时会自动下载 ~800MB 标注数据库，"
            "请确保网络通畅。若编译问题请先 pip install greenlet==2.0.2"
        )
        sys.exit(1)
    log.info("依赖检查通过")


# ==========================================
# 放射学分类（与 step1 保持一致）
# ==========================================
THIN_THRESHOLD = 1.25
THICK_THRESHOLD = 2.5

SHARP_KERNELS = {
    "b70", "b75", "b80", "lung", "fc10", "fc30",
    "bone", "edge", "70", "80",
}
SMOOTH_KERNELS = {
    "b20", "b25", "b30", "b31", "b10", "b35",
    "fc01", "fc03", "standard", "soft", "20", "30", "31",
}


def classify_scan(thickness, kernel):
    """
    与 step1 一致的放射学分类逻辑。
    Returns: str | None
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

    if kernel is not None:
        kernel_lower = str(kernel).lower().replace(" ", "")
        for kw in SHARP_KERNELS:
            if kw in kernel_lower:
                return "Sharp_Kernel"
        for kw in SMOOTH_KERNELS:
            if kw in kernel_lower:
                return "Smooth_Kernel"

    return None


def get_scan_kernel(patient_id, raw_data_dir):
    """
    从本地 DICOM 文件读取 ConvolutionKernel。
    pylidc 的 Scan 对象不直接暴露 kernel，需要从 DICOM 头提取。

    Returns:
        str | None
    """
    patient_path = os.path.join(raw_data_dir, patient_id)
    if not os.path.isdir(patient_path):
        return None

    for root, dirs, files in os.walk(patient_path):
        for f in sorted(files):
            if f.lower().endswith('.dcm'):
                try:
                    ds = pydicom.dcmread(
                        os.path.join(root, f), stop_before_pixels=True
                    )
                    kernel = ds.get('ConvolutionKernel', None)
                    if kernel is not None:
                        return str(kernel)
                except (InvalidDicomError, Exception):
                    continue
    return None


# ==========================================
# 主流程
# ==========================================
def main():
    check_dependencies()

    # 延迟导入 pylidc（确保依赖检查在前）
    import pylidc
    from pylidc.utils import consensus

    log.info("=" * 60)
    log.info("  LIDC-IDRI 标签生成器 (pylidc + consensus)")
    log.info(f"  共识水平: clevel={CONSENSUS_LEVEL}")
    log.info("=" * 60)

    # ---- 1. 初始化 pylidc 数据库 ----
    log.info("[Phase 1] 连接 pylidc 标注数据库...")
    log.info("  (首次运行会自动下载 LIDC-IDRI 标注数据库，约 800MB，请稍候)")
    try:
        all_scans = pylidc.query(pylidc.Scan).all()
        log.info(f"  pylidc 数据库中共有 {len(all_scans)} 个扫描序列")
    except Exception as e:
        log.error(f"pylidc 数据库初始化失败: {e}")
        log.error("请检查网络连接。pylidc 需要从 TCIA 下载标注数据库。")
        sys.exit(1)

    # ---- 2. 匹配本地已下载的患者 ----
    log.info("\n[Phase 2] 匹配本地 DICOM 数据...")
    raw_abs = os.path.abspath(raw_data_dir)
    downloaded_pids = set()
    for item in os.listdir(raw_abs):
        item_path = os.path.join(raw_abs, item)
        if os.path.isdir(item_path) and not item.startswith('.') and not item.startswith('_'):
            downloaded_pids.add(item)

    log.info(f"  本地患者目录: {len(downloaded_pids)}")
    log.info(f"  pylidc 数据库记录: {len(all_scans)}")

    matched_scans = []
    unmatched_pids = downloaded_pids.copy()

    for scan in all_scans:
        if scan.patient_id in downloaded_pids:
            matched_scans.append(scan)
            unmatched_pids.discard(scan.patient_id)

    log.info(f"  成功匹配: {len(matched_scans)} 个扫描")
    if unmatched_pids:
        log.warning(f"  未匹配的本地目录 ({len(unmatched_pids)}): {sorted(unmatched_pids)[:5]}...")

    if not matched_scans:
        log.error("没有匹配到任何患者！请确认 1_Raw_Data 中的 PatientID 与 LIDC-IDRI 一致")
        sys.exit(1)

    # ---- 3. 创建输出目录 ----
    groups = ["Thin_Slice", "Thick_Slice", "Smooth_Kernel", "Sharp_Kernel"]
    for g in groups:
        os.makedirs(os.path.join(output_dir, g, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, g, "labels"), exist_ok=True)

    # ---- 4. 逐病例生成图像 + 标签 ----
    log.info(f"\n[Phase 3] 生成图像与标签 (共识水平={CONSENSUS_LEVEL})...")

    stats = {"ok": 0, "skip_classify": 0, "skip_nolabel": 0, "error": 0}
    group_counts = defaultdict(int)
    case_idx = 1

    for scan in tqdm(matched_scans, desc="处理中", unit="case"):
        pid = scan.patient_id

        try:
            # 提取放射学参数
            thickness = scan.slice_thickness
            # pylidc Scan 对象不直接暴露 kernel，从本地 DICOM 读取
            kernel = get_scan_kernel(pid, raw_abs)

            # 分类
            selected_group = classify_scan(thickness, kernel)
            if selected_group is None:
                stats["skip_classify"] += 1
                log.debug(f"  ⏭️ {pid}: 参数不典型 (T={thickness}, K={kernel})")
                continue

            # 获取结节标注
            nods = scan.cluster_annotations()
            if not nods or len(nods) == 0:
                stats["skip_nolabel"] += 1
                log.debug(f"  ⏭️ {pid}: 无结节标注")
                continue

            # ---- 提取 3D CT 体积 ----
            vol = scan.to_volume(verbose=False)
            if vol is None or vol.size == 0:
                stats["error"] += 1
                log.warning(f"  ❌ {pid}: to_volume() 返回空")
                continue

            # ---- 生成共识掩码 ----
            mask = np.zeros(vol.shape, dtype=np.uint8)
            for nodule_cluster in nods:
                try:
                    cmask, cbbox, _ = consensus(
                        nodule_cluster, clevel=CONSENSUS_LEVEL
                    )
                    # 将结节局部掩码嵌入全图对应位置
                    if cmask is not None and cbbox is not None:
                        slices_tuple = tuple(
                            slice(bb.start, bb.stop) for bb in cbbox
                        )
                        mask[slices_tuple] = np.where(
                            cmask, 1, mask[slices_tuple]
                        )
                except Exception as nod_err:
                    log.debug(f"  {pid}: 单个结节 consensus 失败: {nod_err}")

            # 检查掩码是否为空
            if mask.sum() == 0:
                stats["skip_nolabel"] += 1
                log.debug(f"  ⏭️ {pid}: consensus 掩码全空")
                continue

            # ---- 构建仿射矩阵 ----
            # pylidc 的 to_volume() 默认输出为 (z, y, x)
            # nnU-Net 可以通过 affine 对角元获取 voxel spacing
            px_sp = scan.pixel_spacing if scan.pixel_spacing else 1.0
            dz_val = (
                thickness if thickness and thickness > 0
                else scan.slice_spacing
            )
            if not dz_val or dz_val <= 0:
                dz_val = 2.5

            affine = np.diag([px_sp, px_sp, dz_val, 1.0]).astype(np.float64)

            # ---- 保存图像和标签 ----
            case_name = f"LIDC_CASE_{case_idx:03d}"
            group_img_dir = os.path.join(output_dir, selected_group, "images")
            group_lbl_dir = os.path.join(output_dir, selected_group, "labels")

            img_path = os.path.join(group_img_dir, f"{case_name}_0000.nii.gz")
            lbl_path = os.path.join(group_lbl_dir, f"{case_name}.nii.gz")

            # NIfTI 保存
            img_nii = nib.Nifti1Image(vol, affine)
            lbl_nii = nib.Nifti1Image(mask, affine)

            nib.save(img_nii, img_path)
            nib.save(lbl_nii, lbl_path)

            stats["ok"] += 1
            group_counts[selected_group] += 1
            case_idx += 1

            log.debug(
                f"  ✅ {pid} → {selected_group}/{case_name} "
                f"(vol={vol.shape}, nodules={len(nods)}, "
                f"mask_voxels={int(mask.sum())})"
            )

        except Exception as e:
            stats["error"] += 1
            log.error(f"  ❌ {pid} 处理失败:\n{traceback.format_exc()}")

    # ---- 5. 汇总报告 ----
    log.info("\n" + "=" * 60)
    log.info("  标签生成完成!")
    log.info(f"  成功: {stats['ok']}")
    log.info(f"  跳过 (不可分类): {stats['skip_classify']}")
    log.info(f"  跳过 (无结节/无标签): {stats['skip_nolabel']}")
    log.info(f"  错误: {stats['error']}")
    log.info("-" * 40)
    for g in groups:
        img_count = len([
            f for f in os.listdir(os.path.join(output_dir, g, "images"))
            if f.endswith(".nii.gz")
        ]) if os.path.isdir(os.path.join(output_dir, g, "images")) else 0
        lbl_count = len([
            f for f in os.listdir(os.path.join(output_dir, g, "labels"))
            if f.endswith(".nii.gz")
        ]) if os.path.isdir(os.path.join(output_dir, g, "labels")) else 0
        log.info(f"  {g}: {img_count} images, {lbl_count} labels")
    log.info(f"\n  输出根目录: {os.path.abspath(output_dir)}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
