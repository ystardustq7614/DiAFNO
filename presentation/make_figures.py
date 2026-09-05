#!/usr/bin/env python3
"""模块职责：组会汇报图表生成 —— presentation/figures/ 下 7 张 PNG
（300 dpi）。

数据来源全部为本地归档快照 checkpoints/PRE/<run_tag>/ 下的 eval NPZ
（test h15 / val h1）与 leadtime_diag NPZ，取数目录见 F1_FILES、
ABL_FILES、LAYER_FILES、DIAG_FILES 各常量；非 NPZ 的数字集中在
COND_BARS 等常量并逐条注明出处（实验 05 RESULTS）。

不负责：只读归档 —— 不重算评估、不重建模型、不触碰训练产物；唯一写盘
是 presentation/figures/ 下 7 张 PNG（目录已存在则直接覆盖同名文件，
无拒绝覆盖保护）；不 import 任何正式 PRE 模块。

关键约束（防选错 checkpoint/NPZ）：
- pooled 口径与 pre_evaluate / RESULTS.md 一致：
  pooled = sqrt(Σ rmse²·count / Σcount)；per_lead_ratio 把 (L, 2, Z) 的
  rmse_model / rmse_persistence 沿 (u/v, Z) 轴池化成 (L,)，再逐 lead 相除；
- 每张图的关键数值都与 RESULTS.md 期望值逐项对照（check()/assert），
  不一致即 AssertionError —— 断言失败时优先怀疑选错了 checkpoint/NPZ；
- F6 诊断图只接受键名修复后的 diag NPZ（必须含 m_rmse_u 等 m_/p_ 前缀
  键；旧键 NPZ 是坏档，加载时直接 assert 拒绝）；
- F5 的 Middle 层优先采用正式 Ep4 test NPZ，否则回退 Ep2 探索性档并显式
  标注（LAYER_EXPECT 同步切换）。

运行（repo 根目录）：
    python presentation/make_figures.py
"""
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "PRE")
OUT = os.path.join(ROOT, "presentation", "figures")
os.makedirs(OUT, exist_ok=True)

SURF_RES = os.path.join(CKPT, "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES")
SURF_MS5 = os.path.join(CKPT, "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES_MS5")
SURF_MS10 = os.path.join(CKPT, "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES_MS10")
SURF_SD2 = os.path.join(CKPT, "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2")
MID_RES = os.path.join(CKPT, "middle_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES")
MID_MS5 = os.path.join(CKPT, "middle_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES_MS5")
BOT_RES = os.path.join(CKPT, "bottom_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES")
BOT_MS5 = os.path.join(CKPT, "bottom_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES_MS5")


def one(pattern, base):
    """返回唯一匹配的归档 NPZ 路径；匹配数 != 1 即 AssertionError（防选错档）。"""
    hits = sorted(glob.glob(os.path.join(base, pattern)))
    assert len(hits) == 1, f"expected exactly 1 match for {pattern} in {base}, got {hits}"
    return hits[0]


def optional_one(pattern, base):
    """返回至多一个匹配；0 个匹配返回 None（用于可选的正式档回退）。"""
    hits = sorted(glob.glob(os.path.join(base, pattern)))
    assert len(hits) <= 1, f"expected at most 1 match for {pattern} in {base}, got {hits}"
    return hits[0] if hits else None


def load(path):
    """读取归档 NPZ；allow_pickle=True 是因为存有字符串元数据。"""
    return np.load(path, allow_pickle=True)


def pooled(rmse, count):
    """pooled = sqrt(Σ rmse²·count / Σcount)：与 RESULTS.md 同口径的标量池化。"""
    rmse = np.asarray(rmse, float)
    count = np.asarray(count, float)
    return float(np.sqrt((rmse ** 2 * count).sum() / count.sum()))


def per_lead_ratio(d):
    """(L,2,Z) -> (L,)：把 rmse_model / rmse_persistence 沿 (u/v, Z) 轴
    池化成逐 lead 的 (L,)，再逐 lead 相除得 model/persistence 比值。"""
    m, p, n = d["rmse_model"], d["rmse_persistence"], d["valid_count"]
    lm = np.sqrt((m ** 2 * n).sum(axis=(1, 2)) / n.sum(axis=(1, 2)))
    lp = np.sqrt((p ** 2 * n).sum(axis=(1, 2)) / n.sum(axis=(1, 2)))
    return lm / lp


def overall_ratio(d):
    """15 天 overall 比值：两个全池化标量（model/persistence）之比。"""
    m, p, n = d["rmse_model"], d["rmse_persistence"], d["valid_count"]
    return pooled(m, n) / pooled(p, n)


def day1_ratio(d):
    """day-1 比值：多 lead NPZ 取 per_lead_ratio 首项；单 lead NPZ 退化为 overall_ratio。"""
    return per_lead_ratio(d)[0] if d["rmse_model"].ndim == 3 and d["rmse_model"].shape[0] > 1 \
        else overall_ratio(d)


def check(name, got, want, tol=5e-3):
    """与 RESULTS.md 期望值对照（默认容差 5e-3）；超差即 AssertionError。"""
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: computed {got:.4f} vs RESULTS {want}")
    assert ok, f"{name} mismatch: {got} != {want}"


# ---- F1 / F4 数据：4 个 test h15 NPZ ----
F1_FILES = {
    "单步 residual (Ep10)": one("eval_test_h15_*_ckptEp10.npz", SURF_RES),
    "MS5 (Ep4)": one("eval_test_h15_*_ckptEp4*.npz", SURF_MS5),
    "MS10 (Ep2)": one("eval_test_h15_*_ckptEp2*.npz", SURF_MS10),
    "SD2 diffusion (Ep3)": one("eval_test_h15_*.npz", SURF_SD2),
}
# 期望值（docs/experiments/07、10、04 RESULTS.md test 表）
F1_EXPECT = {"单步 residual (Ep10)": 1.018, "MS5 (Ep4)": 0.871,
             "MS10 (Ep2)": 0.838, "SD2 diffusion (Ep3)": 1.640}
F4_D1_EXPECT = {"单步 residual (Ep10)": 0.833, "MS5 (Ep4)": 0.843,
                "MS10 (Ep2)": 0.833, "SD2 diffusion (Ep3)": 2.201}
STYLES = {"单步 residual (Ep10)": dict(color="#888888", ls="-"),
          "MS5 (Ep4)": dict(color="#1f77b4", ls="-"),
          "MS10 (Ep2)": dict(color="#d62728", ls="-", lw=2.5),
          "SD2 diffusion (Ep3)": dict(color="#d62728", ls="--", alpha=0.65)}


def fig_p20_lead_ratio():
    d = {k: load(v) for k, v in F1_FILES.items()}
    leads = np.arange(1, 16)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for name, dd in d.items():
        r = per_lead_ratio(dd)
        st = STYLES[name]
        ax.plot(leads, r, marker="o" if not st["ls"].startswith("--") else "s",
                ms=4, label=f"{name}  overall={overall_ratio(dd):.3f}", **st)
        check(f"F1 overall {name}", overall_ratio(dd), F1_EXPECT[name])
    ax.axhline(1.0, color="k", lw=1.2, ls=":")
    ax.text(14.6, 1.01, "ratio = 1", fontsize=9)
    ax.set_xlabel("lead day")
    ax.set_ylabel("model / persistence  RMSE ratio")
    ax.set_title("Surface 15-day 自回归 rollout：detached multi-step 消除长期退化\n"
                 "(test, 154 窗口, native m/s；虚线 = direct diffusion 失败参照)", fontsize=11)
    ax.set_xticks(leads)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fp = os.path.join(OUT, "fig_p20_lead_ratio.png")
    fig.savefig(fp, dpi=300)
    plt.close(fig)
    return fp


def fig_p19_overall_bars():
    d = {k: load(v) for k, v in F1_FILES.items()}
    names = list(F1_FILES)
    d1 = [per_lead_ratio(load(F1_FILES[n]))[0] for n in names]
    ov = [overall_ratio(load(F1_FILES[n])) for n in names]
    for n, v in zip(names, d1):
        check(f"F4 day-1 {n}", v, F4_D1_EXPECT[n])
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar(x - w / 2, d1, w, label="Day-1 ratio", color="#9ecae1")
    b2 = ax.bar(x + w / 2, ov, w, label="15-day overall ratio", color="#de6f57")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
    ax.axhline(1.0, color="k", lw=1.2, ls=":")
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylabel("model / persistence RMSE ratio")
    ax.set_title("Surface 主结果（test）：day-1 与 15-day overall\n"
                 "MS10 Ep2：overall 0.838，较 persistence 降低 16.2%", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fp = os.path.join(OUT, "fig_p19_overall_bars.png")
    fig.savefig(fp, dpi=300)
    plt.close(fig)
    return fp


# ---- F2 常量：实验 05 RESULTS.md（同 156 val 窗口、同 seed） ----
COND_BARS = [
    ("Linear probe", 0.1177),
    ("Persistence", 0.1293),
    ("Diffusion\n+ true cond", 0.2584),
    ("错配条件\n(另一窗口)", 0.3408),
    ("Zero cond", 0.4775),
    ("Reversed cond", 0.5655),
]
COLORS_F2 = ["#2ca02c", "#7f7f7f", "#d62728", "#ff9896", "#c44e52", "#931313"]


def fig_p13_condition_signal():
    vals = [v for _, v in COND_BARS]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(range(len(vals)), vals, color=COLORS_F2)
    ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=2)
    ax.axhline(0.1293, color="k", ls=":", lw=1.2)
    ax.set_xticks(range(len(vals)), [k for k, _ in COND_BARS], fontsize=9)
    ax.set_ylabel("Day-1 native RMSE (m/s)")
    ax.set_title("任务有信号、condition 通路完好，但 diffusion 未学成可靠预测器\n"
                 "(实验 05：156 个 validation 窗口、同 seed)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.63)
    fig.tight_layout()
    fp = os.path.join(OUT, "fig_p13_condition_signal.png")
    fig.savefig(fp, dpi=300)
    plt.close(fig)
    return fp


# ---- F3：实验 03，SD2 目录 val h1 NPZ 复算 ----
ABL_FILES = [
    ("Ep2\nchurn=0 E=1", one("eval_val_h1_ch0_e1_s123_ckptEp2.npz", SURF_SD2), 2.312),
    ("Ep3\nchurn=0 E=1", one("eval_val_h1_ch0_e1_s123_ckptEp3.npz", SURF_SD2), 1.998),
    ("Ep4\nchurn=0 E=1", one("eval_val_h1_ch0_e1_s123_ckptEp4.npz", SURF_SD2), 2.555),
    ("Ep3\nchurn=80 E=1", one("eval_val_h1_ch80_e1_s123_ckptEp3.npz", SURF_SD2), 2.500),
    ("Ep3\nchurn=0 E=4", one("eval_val_h1_ch0_e4_s123_ckptEp3.npz", SURF_SD2), 1.911),
    ("Ep3\nsigma_max=3", one("eval_val_h1_*_sigmax3.npz", SURF_SD2), 2.204),
]


def fig_p12_sampler_ablation():
    vals, pers = [], None
    for name, f, want in ABL_FILES:
        d = load(f)
        got = day1_ratio(d)
        check(f"F3 {name!r}", got, want, tol=1e-2)
        vals.append(got)
        pers = pooled(d["rmse_persistence"], d["valid_count"])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(range(len(vals)), vals,
                  color=["#4c72b0"] * 3 + ["#dd8452", "#55a868", "#c44e52"])
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
    ax.axhline(1.0, color="k", ls=":", lw=1.2)
    ax.text(-0.45, 1.04, "persistence", fontsize=9, ha="left")
    ax.set_xticks(range(len(vals)), [k for k, _, _ in ABL_FILES], fontsize=8.5)
    ax.set_ylabel("Day-1 RMSE / persistence")
    ax.set_title(f"Sampler / checkpoint / ensemble 消融（SD2 diffusion，val day-1，"
                 f"persistence {pers:.4f} m/s）\n最好配置仍为 persistence 的 1.91 倍 —— "
                 "sampler 救不回条件模型", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(OUT, "fig_p12_sampler_ablation.png")
    fig.savefig(fp, dpi=300)
    plt.close(fig)
    return fp


# ---- F5：分层 test h15（优先采用 middle Ep4 正式 test，否则回退 Ep2 探索性） ----
MIDDLE_EP4_TEST = optional_one("eval_test_h15_*_ckptEp4*.npz", MID_MS5)
MIDDLE_MS5_TEST = MIDDLE_EP4_TEST or one("eval_test_h15_*_ckptEp2*.npz", MID_MS5)
LAYER_FILES = {
    "Surface": (one("eval_test_h15_*_ckptEp10.npz", SURF_RES),
                one("eval_test_h15_*_ckptEp4*.npz", SURF_MS5)),
    "Middle": (one("eval_test_h15_*_ckptEp10*.npz", MID_RES),
               MIDDLE_MS5_TEST),
    "Bottom": (one("eval_test_h15_*_ckptEp10*.npz", BOT_RES),
               one("eval_test_h15_*_ckptEp5*.npz", BOT_MS5)),
}
LAYER_EXPECT = {  # (单步, MS5) — docs/experiments/07、10、11 RESULTS.md
    "Surface": (1.018, 0.871),
    "Middle": (1.183, None if MIDDLE_EP4_TEST else 0.830),
    "Bottom": (0.930, 0.813),
}


def check_middle_ep4_test(d):
    """校验 Middle 正式 Ep4 test NPZ 的元数据指纹：split/preset/rollout/
    objective/checkpoint 任一不符即 AssertionError（防回退档/选错档）。"""
    scalar = lambda key: np.asarray(d[key]).reshape(-1)[0].item()
    assert scalar("split") == "test"
    assert scalar("preset") == "middle_smoke"
    assert scalar("rollout_days") == 15
    assert scalar("objective") == "persistence_residual"
    assert os.path.basename(scalar("checkpoint_path")) == "Ep4.pth"
    assert d["rmse_model"].shape[0] == 15


def fig_p22_layers():
    single, ms5 = [], []
    for layer, (f1, f2) in LAYER_FILES.items():
        d2 = load(f2)
        if layer == "Middle" and MIDDLE_EP4_TEST:
            check_middle_ep4_test(d2)
        s, m = overall_ratio(load(f1)), overall_ratio(d2)
        check(f"F5 {layer} 单步", s, LAYER_EXPECT[layer][0], tol=8e-3)
        if LAYER_EXPECT[layer][1] is None:
            print(f"  [PASS] F5 Middle MS5: formal Ep4 test computed {m:.4f}")
        else:
            check(f"F5 {layer} MS5", m, LAYER_EXPECT[layer][1], tol=8e-3)
        single.append(s)
        ms5.append(m)
    x = np.arange(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, single, w, label="单步 residual（test h15）", color="#9ecae1")
    b2 = ax.bar(x + w / 2, ms5, w, label="MS5（test h15）", color="#de6f57")
    ax.bar_label(b1, fmt="%.3f", fontsize=9, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=9, padding=2)
    middle_note = "Ep4 正式" if MIDDLE_EP4_TEST else "Ep2 探索性\n(正式 Ep4 待 test)"
    ax.text(1 + w / 2, ms5[1] + 0.055, middle_note, ha="center", fontsize=8,
            color="#225ea8" if MIDDLE_EP4_TEST else "#7f2704")
    ax.axhline(1.0, color="k", ls=":", lw=1.2)
    ax.set_xticks(x, ["Surface (d=29)", "Middle (d=14)", "Bottom (d=0)"])
    ax.set_ylabel("test 15-day overall ratio")
    middle_mark = "" if MIDDLE_EP4_TEST else "*"
    ax.set_title("垂向代表层：detached multi-step 的修复跨深度成立\n"
                 f"（单步→MS5：{single[0]:.3f}/{single[1]:.3f}/{single[2]:.3f} → "
                 f"{ms5[0]:.3f}/{ms5[1]:.3f}{middle_mark}/{ms5[2]:.3f}）", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fp = os.path.join(OUT, "fig_p22_layers.png")
    fig.savefig(fp, dpi=300)
    plt.close(fig)
    return fp


# ---- F6：结构诊断（仅用修复后 m_/p_ key 的 diag NPZ；surface 单步为坏档不用） ----
DIAG_FILES = {"MS5 Ep4": one("leadtime_diag_ckptEp4.npz", SURF_MS5),
              "MS10 Ep2": one("leadtime_diag_ckptEp2.npz", SURF_MS10)}


def fig_p24_diagnostics():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for name, f in DIAG_FILES.items():
        d = load(f)
        assert "m_rmse_u" in d.files, f"{f} 是 key 修复前坏档，禁止使用"
        lead = d["lead"]
        split = str(d["split"])
        style = STYLES["MS5 (Ep4)"] if "MS5" in name else STYLES["MS10 (Ep2)"]
        axes[0].plot(lead, d["m_rmse_u"] / d["p_rmse_u"], label=name, **style)
        axes[1].plot(lead, d["m_corr_mean_u"], label=f"{name} (模型)", **style)
        axes[1].plot(lead, d["p_corr_mean_u"], ls="--", color=style["color"], alpha=0.55)
        axes[2].plot(lead, d["m_var_ratio_u"], label=name, **style)
    axes[0].axhline(1.0, color="k", ls=":", lw=1)
    axes[0].set_title("u RMSE ratio（model/persistence）")
    axes[1].set_title("u 空间相关（实线模型 / 虚线 persistence）")
    axes[2].axhline(1.0, color="k", ls=":", lw=1)
    axes[2].set_title("u variance ratio（1=与真值同方差）")
    for ax in axes:
        ax.set_xlabel("lead day")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    var15 = load(DIAG_FILES["MS10 Ep2"])["m_var_ratio_u"][-1]
    fig.suptitle(f"长期退化三缺陷：d15 ratio 回升、corr 反超、方差塌缩（var_ratio≈{var15:.2f}@d15）"
                 f"　—— 诊断 split={split}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fp = os.path.join(OUT, "fig_p24_diagnostics.png")
    fig.savefig(fp, dpi=300)
    plt.close(fig)
    return fp


# ---- F7：MS10 test 归档 v 分量三联图拼版（u 面板未归档，v 可作任务示意） ----
MAP_DIR = os.path.join(SURF_MS10, "figures_h15_ch0_e1_s123_rf0_ckptEp2_test15ep2")


def fig_p07_map_panel():
    rows = [("Day 1", "d01_s00_v.png"), ("Day 15", "d15_s00_v.png")]
    imgs = [(lab, plt.imread(os.path.join(MAP_DIR, f))) for lab, f in rows]
    fig, axes = plt.subplots(len(imgs), 1, figsize=(11, 4.4 * len(imgs)),
                             constrained_layout=True)
    for ax, (lab, im) in zip(np.atleast_1d(axes), imgs):
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(lab, fontsize=13)
    fig.suptitle("MS10 Ep2（test）：真值 / 模型预测 / 误差（pred−truth）—— v 分量，"
                 "surface 层（归档三联图，原图见 run 目录）", fontsize=12)
    fp = os.path.join(OUT, "fig_p07_forecast_maps.png")
    fig.savefig(fp, dpi=300)
    plt.close(fig)
    return fp


def main():
    outs = {}
    for name, fn in [("F1 P20 主图", fig_p20_lead_ratio),
                     ("F2 P13 条件诊断", fig_p13_condition_signal),
                     ("F3 P12 采样消融", fig_p12_sampler_ablation),
                     ("F4 P19 主结果", fig_p19_overall_bars),
                     ("F5 P22 分层", fig_p22_layers),
                     ("F6 P24 诊断三联", fig_p24_diagnostics),
                     ("F7 P7 场拼版", fig_p07_map_panel)]:
        print(f"[{name}]")
        outs[name] = fn()
    print("\n生成完成：")
    for k, v in outs.items():
        print(f"  {k}: {os.path.relpath(v, ROOT)}")


if __name__ == "__main__":
    main()
