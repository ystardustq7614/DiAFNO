#!/usr/bin/env python3
"""模块职责：生成交接报告用的紧凑图表（与具体数据文件无关，图中数值
为写作时刻固化的快照/结论）。

不负责：不读取任何 NPZ/数据集 —— 图内数字全部硬编码在函数体中
（来自归档审计与实验结论）；不依赖任何正式 PRE 模块。

关键约束：
- 输出 plots/05_handoff_overview.png 与 plots/06_legacy_failure.png
  （OUT 目录在导入时创建，同名文件直接覆盖，无拒绝覆盖保护）；
- 06_legacy_failure 是"历史失败基线"：标题已注明不得当作修复后模型的
  结果展示（u/v 比值与验证相对 L2 曲线均为归档数值）；
- 中文字体依赖本机 Microsoft YaHei / SimHei 等，缺失时由 matplotlib
  回退到其他字体，中文可能显示为方框。

依赖关系：matplotlib / numpy（仅作图，不读数据）。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


OUT = Path(__file__).resolve().parents[1] / "plots"
OUT.mkdir(exist_ok=True)


def overview():
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    fig.suptitle("PRE 海流预测项目交接总览", fontsize=16, weight="bold")

    ax = axes[0, 0]
    ax.axis("off")
    ax.text(0.02, 0.94, "数据概况", fontsize=13, weight="bold", va="top")
    ax.text(
        0.02,
        0.80,
        "10,591 个日平均场 × 30 个 sigma 层 × 400 × 441\n"
        "时间范围：1994-01-01 至 2022-12-30\n"
        "rho 网格：400×441；原生 u：400×440；原生 v：399×441\n"
        "有效海洋网格：69.9%；mask：1=海洋，0=陆地\n"
        "区域：112.315–115.678°E，20.896–23.028°N\n"
        "中位网格距 dx/dy：758 m / 407 m（曲线、非等距网格）",
        fontsize=11,
        va="top",
        linespacing=1.55,
    )

    ax = axes[0, 1]
    widths = np.array([79.32, 10.34, 10.34])
    colors = ["#4472C4", "#ED7D31", "#70AD47"]
    labels = ["训练集\n8,401 天 / 8,394 窗口", "验证集\n1,095 天\n1,088 窗口", "测试集\n1,095 天\n1,088 窗口"]
    left = 0.0
    for width, color, label in zip(widths, colors, labels):
        ax.barh(0, width, left=left, height=0.55, color=color)
        fontsize = 8.5 if width > 20 else 7.2
        ax.text(left + width / 2, 0, label, ha="center", va="center", color="white", fontsize=fontsize, weight="bold")
        left += width
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.65, 0.65)
    ax.set_xlabel("10,591 天占比（按时间顺序切分；滑动窗口不跨集合边界）")
    ax.set_title("1994–2016 训练集 | 2017–2019 验证集 | 2020–2022 测试集")
    ax.grid(axis="x", alpha=0.2)

    ax = axes[1, 0]
    ax.axis("off")
    ax.set_title("单步训练的张量流", pad=12)
    day_colors = ["#5B9BD5", "#A5A5A5"] * 7
    for i in range(14):
        x = 0.02 + i * 0.045
        ax.add_patch(plt.Rectangle((x, 0.62), 0.04, 0.17, color=day_colors[i]))
        ax.text(x + 0.02, 0.705, f"{i}", ha="center", va="center", fontsize=8, color="white")
    ax.text(0.02, 0.86, "条件输入：14 通道 = [u(d0), v(d0), …, u(d6), v(d6)]", fontsize=10)
    ax.text(0.68, 0.69, "+", fontsize=20, ha="center", va="center")
    for i, (label, color) in enumerate([("u*", "#C55A11"), ("v*", "#FFC000")]):
        x = 0.72 + i * 0.07
        ax.add_patch(plt.Rectangle((x, 0.62), 0.06, 0.17, color=color))
        ax.text(x + 0.03, 0.705, label, ha="center", va="center", fontsize=9, color="white")
    ax.text(0.72, 0.86, "带噪的下一日目标", fontsize=10)
    ax.annotate("IAFNO 输入层：16 通道", xy=(0.48, 0.46), xytext=(0.48, 0.57), ha="center", arrowprops={"arrowstyle": "->"})
    ax.text(0.33, 0.31, "输出：2 通道 [u(下一日), v(下一日)]", fontsize=11, weight="bold")
    ax.text(0.02, 0.09, "Mask 不作为输入通道，仅用于计算损失和评估指标。", fontsize=10, color="#C00000")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0.02, 0.94, "当前证据与实验状态", fontsize=13, weight="bold", va="top")
    rows = [
        ("完成", "已核对原始/处理后数据维度、mask 含义及网格结构", "#2E7D32"),
        ("完成", "本地 PRE 与旧版冒烟测试通过（CUDA 专用分支跳过）", "#2E7D32"),
        ("失败", "旧版表层实验失败：第 1 天 RMSE 为持续性基线的 2.7–3.1 倍", "#C62828"),
        ("完成", "代码已修复 sigma_data 尺度错误（0.0856 → 0.1712）", "#2E7D32"),
        ("失败", "SD2 重训与评估已归档：day-1/15-day 仍败于 persistence", "#C62828"),
        ("暂停", "Full-3D 按 surface No-Go 规则暂停，尚未执行", "#EF6C00"),
    ]
    for i, (mark, text, color) in enumerate(rows):
        y = 0.80 - i * 0.13
        ax.text(0.03, y, mark, color=color, fontsize=10.5, weight="bold", va="center")
        ax.text(0.13, y, text, fontsize=10.5, va="center")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "05_handoff_overview.png", dpi=160)
    plt.close(fig)


def legacy_result():
    lead = np.array([1, 5, 10, 15])
    u_ratio = np.array([0.377 / 0.139, 0.628 / 0.258, 0.488 / 0.280, 0.531 / 0.267])
    v_ratio = np.array([0.279 / 0.090, 0.214 / 0.140, 0.232 / 0.146, 0.242 / 0.146])
    val_epoch = np.array([1, 2, 3, 4, 10])
    val_rel = np.array([1.956, 1.577, 1.567, 2.193, 2.393])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(val_epoch, val_rel, "o-", color="#C00000", label="reported checkpoints")
    axes[0].axhline(1, color="black", ls="--", lw=1, label="zero-field threshold")
    axes[0].set(title="Legacy validation (wrong sigma_data scale)", xlabel="epoch", ylabel="masked relative L2")
    axes[0].set_xticks(val_epoch)
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(lead, u_ratio, "o-", label="u", color="#4472C4")
    axes[1].plot(lead, v_ratio, "s-", label="v", color="#ED7D31")
    axes[1].axhline(1, color="black", ls="--", lw=1, label="persistence")
    axes[1].set(title="Legacy test RMSE / persistence RMSE", xlabel="lead day", ylabel="ratio (<1 is better)")
    axes[1].set_xticks(lead)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle("Historical failure baseline — do not present as the fixed model's final result", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "06_legacy_failure.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    overview()
    legacy_result()
    print(f"saved handoff figures to {OUT}")
