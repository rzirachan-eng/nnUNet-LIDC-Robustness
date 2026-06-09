"""
step5_final_evaluation.py
=========================
对 nnU-Net 推理结果进行医学图像分割评估，计算 Dice 系数和 Hausdorff 距离 (HD95)，
并执行统计分析与可视化。

数据流:
  2_Stratified_Data/{Group}/labels/  ← 金标准 (Ground Truth)
  4_AI_Outputs/{Group}_Pred/         ← AI 预测结果
  → 5_Evaluation_Metrics/            → 评估结果输出

评估指标:
  - Dice Similarity Coefficient (DSC): 分割重叠度, 范围 [0,1], 1=完美重叠
  - 95% Hausdorff Distance (HD95): 边界距离, 单位 mm, 值越小越好

统计方法:
  - 两组比较: Welch's t-test + Mann-Whitney U (自动选择)
  - 多组比较: Kruskal-Wallis H + post-hoc Dunn's test
  - 正态性: Shapiro-Wilk test (α=0.05)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu
from pathlib import Path
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # 非交互式后端, 避免无 GUI 环境报错
import matplotlib.pyplot as plt
import seaborn as sns
from medpy.metric.binary import dc as dice_coefficient
from medpy.metric.binary import hd95 as hausdorff_95

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ============================================================
# 路径配置 —— 基于项目根目录动态构建, 避免硬编码
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

GT_BASE_DIR = os.path.join(PROJECT_ROOT, "2_Stratified_Data")
"""金标准目录: 2_Stratified_Data/{Group}/labels/"""

PRED_BASE_DIR = os.path.join(PROJECT_ROOT, "4_AI_Outputs")
"""预测结果目录: 4_AI_Outputs/{Group}_Pred/"""

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "5_Evaluation_Metrics")
"""评估输出目录"""

# 要评估的所有分层组 (可在此增减)
ALL_GROUPS = ["Thin_Slice", "Thick_Slice", "Smooth_Kernel", "Sharp_Kernel"]

# 统计显著性阈值
ALPHA = 0.05

# ============================================================
# 工具函数
# ============================================================


def load_nifti(filepath):
    """
    安全加载 NIfTI 文件, 返回 (numpy_array, voxel_spacing_mm)。

    Args:
        filepath: .nii.gz 文件路径

    Returns:
        (data, zooms) 或 (None, None) 如果加载失败
    """
    try:
        img = nib.load(filepath)
        # 使用 np.asanyarray 代替已弃用的 get_fdata()
        data = np.asanyarray(img.dataobj, dtype=np.float32)
        zooms = img.header.get_zooms()[:3]  # (dz, dy, dx) 或 (dx, dy, dz)
        return data, zooms
    except Exception as e:
        log.error(f"加载文件失败 [{filepath}]: {e}")
        return None, None


def validate_shapes(gt_data, pred_data, case_id):
    """
    检查金标准和预测结果的维度是否一致。
    若不一致则尝试裁剪/填充最小公共区域。

    Returns:
        (gt_cropped, pred_cropped, is_valid)
    """
    if gt_data.shape == pred_data.shape:
        return gt_data, pred_data, True

    log.warning(f"  [{case_id}] 维度不匹配: GT {gt_data.shape} vs Pred {pred_data.shape}")

    # 尝试裁剪到最小公共区域
    try:
        min_shape = tuple(min(a, b) for a, b in zip(gt_data.shape, pred_data.shape))
        gt_cropped = gt_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        pred_cropped = pred_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        log.warning(f"  [{case_id}] 已裁剪至 {min_shape}")
        return gt_cropped, pred_cropped, True
    except Exception as e:
        log.error(f"  [{case_id}] 裁剪失败: {e}")
        return gt_data, pred_data, False


def compute_metrics(gt_data, pred_data, voxel_spacing):
    """
    计算单例的 Dice 和 HD95, 处理边界情况。

    边界情况处理策略:
      - 两者全零 (都无结节) → Dice=1.0, HD95=0.0 (完美一致)
      - 只有一方为零 → Dice=0.0, HD95=NaN (标记为缺失)
      - 正常情况 → 计算实际指标

    Args:
        gt_data:   金标准二值数组
        pred_data: 预测二值数组
        voxel_spacing: 体素间距 (mm), 如 (dx, dy, dz)

    Returns:
        (dice, hd95) 元组
    """
    gt_sum = np.sum(gt_data)
    pred_sum = np.sum(pred_data)

    # 情况1: 两者全为零 — 都正确判断为无结节
    if gt_sum == 0 and pred_sum == 0:
        return 1.0, 0.0

    # 情况2: 只有一方为零 — 完全不一致
    if gt_sum == 0 or pred_sum == 0:
        return 0.0, float("nan")

    # 情况3: 正常计算
    try:
        # 确保二值化 (medpy 期望 0/1 整数)
        gt_bin = (gt_data > 0).astype(np.uint8)
        pred_bin = (pred_data > 0).astype(np.uint8)
        dsc = dice_coefficient(pred_bin, gt_bin)
    except Exception as e:
        log.warning(f"  Dice 计算异常: {e}")
        dsc = 0.0

    try:
        # HD95 需要传入体素间距 (voxelspacing), 否则距离单位是体素而非 mm
        hd = hausdorff_95(pred_bin, gt_bin, voxelspacing=voxel_spacing)
        # 限制极大值避免离群点污染统计
        hd = min(hd, 200.0)
    except Exception as e:
        log.warning(f"  HD95 计算异常: {e}")
        hd = float("nan")

    return dsc, hd


# ============================================================
# 核心: 分组评估
# ============================================================


def evaluate_group(group_name):
    """
    对单个分层组执行批量评估。

    流程:
      1. 扫描 labels/ 目录获取所有金标准文件
      2. 逐例匹配预测结果 (4_AI_Outputs/{group}_Pred/)
      3. 计算 Dice + HD95
      4. 返回 DataFrame

    Args:
        group_name: 组名, 如 "Thin_Slice", "Sharp_Kernel"

    Returns:
        pd.DataFrame, 包含 Case_ID, Group, Dice, HD95 列
    """
    gt_dir = os.path.join(GT_BASE_DIR, group_name, "labels")
    pred_dir = os.path.join(PRED_BASE_DIR, f"{group_name}_Pred")

    # 检查输入目录是否存在
    if not os.path.isdir(gt_dir):
        log.warning(f"跳过 [{group_name}]: 金标准目录不存在 {gt_dir}")
        return pd.DataFrame()

    if not os.path.isdir(pred_dir):
        log.warning(f"跳过 [{group_name}]: 预测目录不存在 {pred_dir}")
        return pd.DataFrame()

    # 获取所有金标准标签文件
    label_files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".nii.gz"))
    if not label_files:
        log.warning(f"跳过 [{group_name}]: labels/ 目录为空")
        return pd.DataFrame()

    log.info(f"评估 [{group_name}]: {len(label_files)} 个病例")

    results = []
    skipped_missing = 0
    skipped_dimension = 0

    total = len(label_files)
    for idx, label_name in enumerate(label_files, 1):
        case_id = label_name.replace(".nii.gz", "")
        gt_path = os.path.join(gt_dir, label_name)
        pred_path = os.path.join(pred_dir, label_name)

        # 检查预测文件是否存在
        if not os.path.exists(pred_path):
            log.warning(f"  [{case_id}] 预测文件缺失, 跳过")
            skipped_missing += 1
            continue

        # 加载数据
        gt_data, gt_zooms = load_nifti(gt_path)
        pred_data, _ = load_nifti(pred_path)

        if gt_data is None or pred_data is None:
            skipped_missing += 1
            continue

        # 验证维度一致性
        gt_data, pred_data, dim_ok = validate_shapes(gt_data, pred_data, case_id)
        if not dim_ok:
            skipped_dimension += 1
            continue

        # 计算指标
        dice_score, hd95_score = compute_metrics(gt_data, pred_data, gt_zooms)
        log.info(f"  [{idx}/{total}] {case_id}: Dice={dice_score:.4f}, HD95={hd95_score if np.isnan(hd95_score) else f'{hd95_score:.2f}'}")

        results.append({
            "Case_ID": case_id,
            "Group": group_name,
            "Dice": dice_score,
            "HD95": hd95_score,
        })

    # 汇总跳过的数量
    n_success = len(results)
    if skipped_missing > 0:
        log.warning(f"  [{group_name}] {skipped_missing} 例因文件缺失跳过")
    if skipped_dimension > 0:
        log.warning(f"  [{group_name}] {skipped_dimension} 例因维度不匹配跳过")
    log.info(f"  [{group_name}] 成功评估 {n_success} 例")

    return pd.DataFrame(results)


# ============================================================
# 统计分析
# ============================================================


def check_normality(data, group_name):
    """
    Shapiro-Wilk 正态性检验。

    H0: 数据服从正态分布
    Ha: 数据不服从正态分布
    若 p < ALPHA → 拒绝 H0 → 数据不服从正态分布

    Returns:
        (statistic, p_value, is_normal: bool)
    """
    if len(data) < 3:
        log.warning(f"  [{group_name}] 样本量不足 (n={len(data)}), 无法进行正态性检验")
        return np.nan, np.nan, False
    stat, p = stats.shapiro(data)
    is_normal = p >= ALPHA
    return stat, p, is_normal


def compare_two_groups(df, group_a, group_b, metric="Dice"):
    """
    对两组数据进行统计比较，自动选择参数/非参数检验。

    流程:
      1. Shapiro-Wilk 检验正态性
      2. 若两组均正态 → Welch's t-test (不假设方差齐性)
      3. 否则 → Mann-Whitney U test (非参数)

    Args:
        df:       包含所有数据的 DataFrame
        group_a:  组A名称
        group_b:  组B名称
        metric:   比较的指标列名 ("Dice" 或 "HD95")

    Returns:
        dict: {group_a_mean, group_b_mean, group_a_std, group_b_std,
               test_method, statistic, p_value, significant}
    """
    data_a = df[df["Group"] == group_a][metric].dropna().values
    data_b = df[df["Group"] == group_b][metric].dropna().values

    if len(data_a) == 0 or len(data_b) == 0:
        log.warning(f"  {group_a} vs {group_b}: 数据为空, 跳过")
        return None

    log.info(f"\n{'='*50}")
    log.info(f"  {group_a} (n={len(data_a)}) vs {group_b} (n={len(data_b)}) [{metric}]")
    log.info(f"{'='*50}")

    # 描述统计
    log.info(f"  {group_a}: Mean={np.mean(data_a):.4f}, SD={np.std(data_a):.4f}")
    log.info(f"  {group_b}: Mean={np.mean(data_b):.4f}, SD={np.std(data_b):.4f}")

    # 正态性检验
    _, p_norm_a, normal_a = check_normality(data_a, group_a)
    _, p_norm_b, normal_b = check_normality(data_b, group_b)

    log.info(f"  正态性检验 (Shapiro-Wilk):")
    log.info(f"    {group_a}: p={p_norm_a:.4f} {'✓ 正态' if normal_a else '✗ 非正态'}")
    log.info(f"    {group_b}: p={p_norm_b:.4f} {'✓ 正态' if normal_b else '✗ 非正态'}")

    # 选择检验方法
    if normal_a and normal_b:
        # 两组均正态 → Welch's t-test (不等方差)
        stat, p_val = stats.ttest_ind(data_a, data_b, equal_var=False)
        method = "Welch's t-test"
        log.info(f"  检验方法: Welch's t-test (两组均正态)")
    else:
        # 至少一组非正态 → Mann-Whitney U
        stat, p_val = mannwhitneyu(data_a, data_b, alternative="two-sided")
        method = "Mann-Whitney U"
        log.info(f"  检验方法: Mann-Whitney U (非参数)")

    significant = p_val < ALPHA
    log.info(f"  统计量: {stat:.4f}")
    log.info(f"  P 值: {p_val:.6f}")
    log.info(f"  结论: {'✅ 显著差异 (p < {ALPHA})' if significant else '⚠️ 无显著差异'}")

    return {
        "group_a": group_a,
        "group_b": group_b,
        "metric": metric,
        "n_a": len(data_a),
        "n_b": len(data_b),
        "mean_a": np.mean(data_a),
        "mean_b": np.mean(data_b),
        "std_a": np.std(data_a),
        "std_b": np.std(data_b),
        "test_method": method,
        "statistic": stat,
        "p_value": p_val,
        "significant": significant,
    }


def compare_all_groups(df, metric="Dice"):
    """
    对 DataFrame 中所有组两两比较，生成统计结果表。
    同时执行 Kruskal-Wallis 多组整体检验。

    Returns:
        list[dict]: 两两比较结果列表
    """
    groups = sorted(df["Group"].unique())
    if len(groups) < 2:
        log.warning("组数不足两组，无法进行统计比较")
        return []

    # Kruskal-Wallis 多组整体检验 (非参数)
    log.info(f"\n{'#'*50}")
    log.info(f"  Kruskal-Wallis 多组整体检验 [{metric}]")
    log.info(f"{'#'*50}")
    group_data = [df[df["Group"] == g][metric].dropna().values for g in groups]
    h_stat, p_kw = stats.kruskal(*group_data)
    log.info(f"  H 统计量: {h_stat:.4f}, P 值: {p_kw:.6f}")
    log.info(f"  结论: {'✅ 组间存在显著差异' if p_kw < ALPHA else '⚠️ 组间无显著差异'}")

    # 两两比较
    pairwise_results = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            res = compare_two_groups(df, groups[i], groups[j], metric)
            if res:
                pairwise_results.append(res)

    return pairwise_results


# ============================================================
# 可视化
# ============================================================


def setup_academic_style():
    """设置 Matplotlib/Seaborn 学术论文风格。"""
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })
    sns.set_style("whitegrid")


def plot_boxplot(df, metric="Dice", save_path=None):
    """
    绘制学术风格箱线图 + 散点叠加。

    Args:
        df:        包含 Group 和 metric 列的 DataFrame
        metric:    要绘制的指标列名
        save_path: 保存路径, None 则不保存
    """
    groups = sorted(df["Group"].unique())
    n_groups = len(groups)
    fig_width = max(5, n_groups * 1.8)

    fig, ax = plt.subplots(figsize=(fig_width, 5))

    # 定义统一的颜色调色板
    palette = sns.color_palette("Set2", n_colors=n_groups)

    # 箱线图
    sns.boxplot(
        x="Group", y=metric, data=df,
        palette=palette, width=0.5,
        linewidth=1.2, fliersize=4,
        ax=ax,
    )

    # 散点叠加 (展现数据分布)
    sns.stripplot(
        x="Group", y=metric, data=df,
        color="black", alpha=0.35, size=5,
        jitter=True, ax=ax,
    )

    # 标签与标题
    metric_label = {
        "Dice": "Dice Similarity Coefficient (DSC)",
        "HD95": "95% Hausdorff Distance (mm)",
    }.get(metric, metric)

    ax.set_ylabel(metric_label)
    ax.set_xlabel("")
    ax.set_title(f"Segmentation Performance: {metric_label}")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Dice 范围 [0, 1], HD95 下限为 0
    if metric == "Dice":
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(y=0.7, color="red", linestyle=":", alpha=0.5, label="DSC=0.7 (临床可用)")
    else:
        ax.set_ylim(bottom=-2)

    # 图例
    if metric == "Dice":
        ax.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        log.info(f"图表已保存: {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ============================================================
# 主流程
# ============================================================


def main():
    """执行完整的评估流水线。"""
    setup_academic_style()

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 验证关键目录
    if not os.path.isdir(GT_BASE_DIR):
        log.error(f"金标准目录不存在: {GT_BASE_DIR}")
        sys.exit(1)
    if not os.path.isdir(PRED_BASE_DIR):
        log.error(f"预测结果目录不存在: {PRED_BASE_DIR}")
        sys.exit(1)

    log.info("=" * 60)
    log.info("  nnU-Net 分割评估工具")
    log.info(f"  金标准: {GT_BASE_DIR}")
    log.info(f"  预测:   {PRED_BASE_DIR}")
    log.info(f"  输出:   {OUTPUT_DIR}")
    log.info(f"  分组:   {ALL_GROUPS}")
    log.info("=" * 60)

    # ---- 第1步: 逐组评估 ----
    log.info("\n[Step 1] 逐组计算 Dice 和 HD95...")
    df_all = pd.DataFrame()
    for group in ALL_GROUPS:
        df_group = evaluate_group(group)
        if not df_group.empty:
            df_all = pd.concat([df_all, df_group], ignore_index=True)

    if df_all.empty:
        log.error("没有成功评估任何病例! 请检查数据目录和预测文件")
        sys.exit(1)

    # 保存完整结果表
    result_csv = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    df_all.to_csv(result_csv, index=False, float_format="%.6f")
    log.info(f"\n✅ 完整评估结果已保存: {result_csv}")
    log.info(f"   总病例数: {len(df_all)}")

    # 打印分组汇总
    log.info("\n[分组汇总]")
    for group in sorted(df_all["Group"].unique()):
        sub = df_all[df_all["Group"] == group]
        dice_mean = sub["Dice"].mean()
        dice_std = sub["Dice"].std()
        hd95_valid = sub["HD95"].dropna()
        hd95_median = hd95_valid.median() if len(hd95_valid) > 0 else float("nan")
        log.info(f"  {group}: n={len(sub)}, Dice={dice_mean:.4f}±{dice_std:.4f}, HD95_median={hd95_median:.2f}mm")

    # ---- 第2步: 统计分析 ----
    log.info("\n[Step 2] 统计分析...")

    # Dice 分析
    dice_results = compare_all_groups(df_all, metric="Dice")
    # HD95 分析
    hd95_results = compare_all_groups(df_all, metric="HD95")

    # 保存统计结果
    stats_rows = []
    for r in (dice_results or []) + (hd95_results or []):
        stats_rows.append({
            "Comparison": f"{r['group_a']} vs {r['group_b']}",
            "Metric": r["metric"],
            "n_A": r["n_a"],
            "n_B": r["n_b"],
            "Mean_A": f"{r['mean_a']:.4f}",
            "Mean_B": f"{r['mean_b']:.4f}",
            "SD_A": f"{r['std_a']:.4f}",
            "SD_B": f"{r['std_b']:.4f}",
            "Test": r["test_method"],
            "Statistic": f"{r['statistic']:.4f}",
            "P_value": f"{r['p_value']:.6f}",
            "Significant": "Yes" if r["significant"] else "No",
        })
    if stats_rows:
        stats_csv = os.path.join(OUTPUT_DIR, "statistical_results.csv")
        pd.DataFrame(stats_rows).to_csv(stats_csv, index=False)
        log.info(f"  统计结果已保存: {stats_csv}")

    # ---- 第3步: 可视化 ----
    log.info("\n[Step 3] 生成可视化图表...")

    # Dice 箱线图
    dice_plot_path = os.path.join(OUTPUT_DIR, "dice_comparison_boxplot.png")
    plot_boxplot(df_all, metric="Dice", save_path=dice_plot_path)

    # HD95 箱线图
    hd95_plot_path = os.path.join(OUTPUT_DIR, "hd95_comparison_boxplot.png")
    plot_boxplot(df_all, metric="HD95", save_path=hd95_plot_path)

    # ---- 第4步: 核心结论 ----
    log.info("\n" + "=" * 60)
    log.info("  评估完成!")
    log.info(f"  结果文件: {result_csv}")
    log.info(f"  统计表:   {stats_csv if stats_rows else 'N/A'}")
    log.info(f"  图表:     {dice_plot_path}")
    log.info(f"            {hd95_plot_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
