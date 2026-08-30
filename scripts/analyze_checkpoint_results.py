#!/usr/bin/env python3
"""Generate evidence plots for the PRE surface-smoke checkpoints (read-only)."""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "checkpoints" / "PRE"
OLD = PRE / "surface_smoke_BS4_EMD180_I4_E4_S32_C7"
NEW = PRE / "surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2"
OUT = ROOT / "plots"

OLD_TEST = OLD / "eval_test.npz"
NEW_TEST = NEW / "eval_test_h15_ch0_e1_s123_ckptEp3.npz"
COND_PROBE = PRE / "diag_noGo_20260828" / "results" / "probe_sample_conds_full.npz"
COND_LOG = PRE / "diag_noGo_20260828" / "results" / "probe_sample_conds_full.log"
LINEAR_LOG = PRE / "diag_noGo_20260828" / "results" / "probe_linear.log"


def pooled_by_lead(z, key):
    values = z[key].astype(float)
    count = z["valid_count"].astype(float)
    if key.startswith("rmse_"):
        return np.sqrt((values**2 * count).sum(axis=(1, 2)) / count.sum(axis=(1, 2)))
    return (values * count).sum(axis=(1, 2)) / count.sum(axis=(1, 2))


def pooled_overall(z, key):
    values = z[key].astype(float)
    count = z["valid_count"].astype(float)
    if key.startswith("rmse_"):
        return float(np.sqrt((values**2 * count).sum() / count.sum()))
    return float((values * count).sum() / count.sum())


def one_day(path):
    with np.load(path) as z:
        return (
            pooled_by_lead(z, "rmse_model")[0],
            pooled_by_lead(z, "rmse_persistence")[0],
            pooled_by_lead(z, "rmse_zero")[0],
        )


def first_number(pattern, text):
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"pattern not found: {pattern}")
    return float(match.group(1))


def result_overview(old_test, new_test):
    old_loss = np.loadtxt(OLD / "loss.dat").reshape(-1, 3)
    new_loss = np.loadtxt(NEW / "loss.dat").reshape(-1, 3)
    lead = np.arange(1, len(new_test["rmse_model"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("PRE surface_smoke SD2 — second smoke-test outcome", fontsize=16, weight="bold")

    ax = axes[0, 0]
    ax.semilogy(np.arange(1, len(old_loss) + 1), old_loss[:, 1], "o--", color="#A5A5A5", label="legacy train loss")
    ax.semilogy(np.arange(1, len(new_loss) + 1), new_loss[:, 1], "o-", color="#4472C4", label="SD2 train loss")
    ax.set(xlabel="epoch", ylabel="train denoising loss (log scale)", title="Training loss falls, validation does not")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, len(new_loss) + 1), new_loss[:, 2], "s-", color="#C00000", label="SD2 val relative L2")
    ax2.axhline(1, color="black", ls=":", lw=1)
    ax2.set_ylabel("sampled val masked relative L2")
    ax2.annotate("best = Ep3", (3, new_loss[2, 2]), xytext=(3.35, 1.35), arrowprops={"arrowstyle": "->"})
    lines = ax.get_lines() + ax2.get_lines()[:1]
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    cases = [
        ("Ep2", NEW / "eval_val_h1_ch0_e1_s123_ckptEp2.npz"),
        ("Ep3", NEW / "eval_val_h1_ch0_e1_s123_ckptEp3.npz"),
        ("Ep4", NEW / "eval_val_h1_ch0_e1_s123_ckptEp4.npz"),
        ("Ep3\nchurn80", NEW / "eval_val_h1_ch80_e1_s123_ckptEp3.npz"),
        ("Ep3\nensemble4", NEW / "eval_val_h1_ch0_e4_s123_ckptEp3.npz"),
        ("Ep3\nsigma_max3", NEW / "eval_val_h1_ch0_e1_s123_ckptEp3_sigmax3.npz"),
    ]
    vals = [one_day(path)[0] for _, path in cases]
    _, pers, zero = one_day(cases[0][1])
    bars = ax.bar([name for name, _ in cases], vals, color=["#A5A5A5", "#4472C4", "#A5A5A5", "#ED7D31", "#70AD47", "#FFC000"])
    ax.axhline(pers, color="black", ls="--", label=f"persistence {pers:.3f}")
    ax.axhline(zero, color="#7030A0", ls=":", label=f"zero field {zero:.3f}")
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value / pers:.2f}×", ha="center", fontsize=8)
    ax.set(title="Day-1 validation ablations", ylabel="native masked RMSE [m/s]")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    for z, key, label, color, style in (
        (old_test, "rmse_model", "legacy (old sigma, churn80)", "#A5A5A5", "--"),
        (new_test, "rmse_model", "SD2 (churn0)", "#C00000", "-"),
        (new_test, "rmse_persistence", "persistence", "#4472C4", "-"),
        (new_test, "rmse_zero", "zero field", "#7030A0", ":"),
    ):
        ax.plot(lead, pooled_by_lead(z, key), marker="o" if key == "rmse_model" else None, ls=style, color=color, label=label)
    ax.set(title="15-day test rollout", xlabel="lead day", ylabel="pooled native RMSE [m/s]", xticks=[1, 3, 5, 7, 10, 12, 15])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    model = new_test["rmse_model"][:, :, 0]
    persistence = new_test["rmse_persistence"][:, :, 0]
    ax.plot(lead, model[:, 0] / persistence[:, 0], "o-", label="u", color="#4472C4")
    ax.plot(lead, model[:, 1] / persistence[:, 1], "s-", label="v", color="#ED7D31")
    ax.plot(lead, pooled_by_lead(new_test, "rmse_model") / pooled_by_lead(new_test, "rmse_persistence"), "^-", label="pooled", color="#C00000")
    ax.axhline(1, color="black", ls="--", label="must be <1")
    ax.set(title="No lead day beats persistence", xlabel="lead day", ylabel="model / persistence RMSE", xticks=[1, 3, 5, 7, 10, 12, 15])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUT / "07_sd2_result_overview.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def diagnosis_figure():
    probe = np.load(COND_PROBE, allow_pickle=True)
    probe_log = COND_LOG.read_text(encoding="utf-8", errors="replace")
    linear_log = LINEAR_LOG.read_text(encoding="utf-8", errors="replace")
    ridge = first_number(r"ridge lambda=0\s*: pooled native RMSE = ([0-9.]+)", linear_log)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle("Why SD2 still fails — diagnostic evidence", fontsize=16, weight="bold")

    ax = axes[0, 0]
    names = ["linear\ncondition-only", "persistence", "diffusion\ntrue condition", "zero field"]
    values = [ridge, probe["rmse_pers"][0], probe["rmse_a"][0], probe["rmse_zero"][0]]
    bars = ax.bar(names, values, color=["#70AD47", "#4472C4", "#C00000", "#7030A0"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(title="The task has learnable condition→target signal", ylabel="day-1 val RMSE [m/s]")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    names = ["true", "wrong window", "zero condition", "reversed days"]
    values = [probe["rmse_a"][0], probe["rmse_c"][0], probe["rmse_b"][0], probe["rmse_d"][0]]
    bars = ax.bar(names, values, color=["#4472C4", "#FFC000", "#A5A5A5", "#ED7D31"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.axhline(probe["rmse_pers"][0], color="black", ls="--", label="persistence")
    ax.set(title="Condition path works, but is too weak", ylabel="day-1 val RMSE [m/s]")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    corr_names = ["true", "wrong window", "zero condition", "reversed days", "persistence"]
    corr_tags = ["a", "c", "b", "d", "pers"]
    corr = [first_number(rf"^\s*{tag}\s*:.*corr\(pred,truth\)=([-0-9.]+)", probe_log) for tag in corr_tags]
    bars = ax.bar(corr_names, corr, color=["#4472C4", "#FFC000", "#A5A5A5", "#ED7D31", "#70AD47"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Predicted spatial pattern is weakly aligned", ylabel="mean spatial correlation with truth", ylim=(-0.2, 1.0))
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    open_rmse = first_number(r"^\s*a\s*: RMSE open-ocean=([0-9.]+)", probe_log)
    coast_rmse = first_number(r"^\s*a\s*: RMSE open-ocean=[0-9.]+\s+coastal-band=([0-9.]+)", probe_log)
    bars = ax.bar(["open ocean", "coastal band"], [open_rmse, coast_rmse], color=["#5B9BD5", "#ED7D31"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.text(0.5, max(open_rmse, coast_rmse) * 0.72, f"coast/open = {coast_rmse / open_rmse:.2f}×", ha="center", weight="bold")
    ax.set(title="Coastal artefacts amplify the failure", ylabel="rho-grid RMSE [m/s]")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUT / "08_sd2_diagnosis.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    probe.close()
    return path


def main():
    OUT.mkdir(exist_ok=True)
    with np.load(OLD_TEST) as old_test, np.load(NEW_TEST) as new_test:
        assert np.array_equal(old_test["window_start_indices"], new_test["window_start_indices"])
        assert np.allclose(old_test["rmse_persistence"], new_test["rmse_persistence"])
        overview = result_overview(old_test, new_test)
        print(f"day-1: old={pooled_by_lead(old_test, 'rmse_model')[0]:.4f}, SD2={pooled_by_lead(new_test, 'rmse_model')[0]:.4f}, persistence={pooled_by_lead(new_test, 'rmse_persistence')[0]:.4f}")
        print(f"15-day overall: SD2={pooled_overall(new_test, 'rmse_model'):.4f}, persistence={pooled_overall(new_test, 'rmse_persistence'):.4f}, zero={pooled_overall(new_test, 'rmse_zero'):.4f}")
    diagnosis = diagnosis_figure()
    print(f"saved {overview}")
    print(f"saved {diagnosis}")


if __name__ == "__main__":
    main()
