#!/usr/bin/env python3
"""超参数敏感性分析结果可视化 (Revision Experiment 1).

从 run_hyperparam_sensitivity.py 的输出目录读取汇总结果,
生成论文可用的图表:
  - 4 个参数 × 2 指标 (MAP + PPL) 的折线图 (带误差棒)
  - 一张综合对比表 (LaTeX 格式)

用法:
  python scripts/revision/plot_sensitivity.py
  python scripts/revision/plot_sensitivity.py --input_dir artifacts/revision_sensitivity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("⚠️  matplotlib 未安装, 将仅输出文本表格")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_summaries(input_dir: Path) -> Dict:
    """加载全局汇总文件."""
    path = input_dir / "all_summaries.json"
    if not path.exists():
        # 尝试从各子目录分别加载
        summaries = {}
        for sub in input_dir.iterdir():
            if sub.is_dir():
                summary_path = sub / "summary_table.json"
                if summary_path.exists():
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summaries[sub.name] = json.load(f)
        return summaries

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _plot_sensitivity_grid(summaries: Dict, output_dir: Path):
    """生成 2×2 网格图: 每个参数一个子图, 左轴 MAP, 右轴 PPL."""
    if not HAS_MPL:
        return

    param_order = ["lambda_know", "beta_pair", "lambda_graph", "gamma_l1"]
    available = [p for p in param_order if p in summaries]

    if not available:
        print("❌ 无可用参数数据")
        return

    n_params = len(available)
    n_cols = 2
    n_rows = (n_params + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, param_name in enumerate(available):
        row, col = divmod(idx, n_cols)
        ax1 = axes[row][col]

        summary = summaries[param_name]
        results = summary["results"]

        values = []
        map_means, map_stds = [], []
        ppl_herb_means, ppl_herb_stds = [], []
        ppl_sym_means, ppl_sym_stds = [], []

        for r in results:
            if r["n_completed"] == 0:
                continue
            val = r["value"]
            m = r.get("mean", {})
            s = r.get("std", {})

            values.append(val)
            map_means.append(m.get("valid_map_avg", np.nan))
            map_stds.append(s.get("valid_map_avg", 0))

            # 优先使用 EM PPL, 否则回退到普通 PPL
            herb_ppl = m.get("herb_pred_ppl_em_prob", m.get("herb_pred_ppl_prob", m.get("herb_pred_ppl", np.nan)))
            herb_ppl_s = s.get("herb_pred_ppl_em_prob", s.get("herb_pred_ppl_prob", s.get("herb_pred_ppl", 0)))
            ppl_herb_means.append(herb_ppl)
            ppl_herb_stds.append(herb_ppl_s)

            sym_ppl = m.get("symptom_pred_ppl_em_prob", m.get("symptom_pred_ppl_prob", m.get("symptom_pred_ppl", np.nan)))
            sym_ppl_s = s.get("symptom_pred_ppl_em_prob", s.get("symptom_pred_ppl_prob", s.get("symptom_pred_ppl", 0)))
            ppl_sym_means.append(sym_ppl)
            ppl_sym_stds.append(sym_ppl_s)

        values = np.array(values)
        map_means = np.array(map_means)
        map_stds = np.array(map_stds)
        ppl_herb_means = np.array(ppl_herb_means)
        ppl_herb_stds = np.array(ppl_herb_stds)
        ppl_sym_means = np.array(ppl_sym_means)
        ppl_sym_stds = np.array(ppl_sym_stds)

        # X 轴使用 log scale (0 值用特殊标记)
        # 为了 log scale, 用序号作为 x, 但标签显示真实值
        x = np.arange(len(values))

        # 左轴: MAP (蓝)
        color_map = "#2563eb"
        ax1.errorbar(x, map_means, yerr=map_stds, color=color_map,
                     marker='o', markersize=6, linewidth=2, capsize=3,
                     label="MAP$_{avg}$")
        ax1.set_ylabel("MAP$_{avg}$", color=color_map, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color_map)

        # 右轴: PPL (红/橙)
        ax2 = ax1.twinx()
        color_herb = "#dc2626"
        color_sym = "#ea580c"
        ax2.errorbar(x, ppl_herb_means, yerr=ppl_herb_stds, color=color_herb,
                     marker='s', markersize=5, linewidth=1.5, capsize=3,
                     linestyle='--', label="Herb PPL")
        ax2.errorbar(x, ppl_sym_means, yerr=ppl_sym_stds, color=color_sym,
                     marker='^', markersize=5, linewidth=1.5, capsize=3,
                     linestyle=':', label="Sym PPL")
        ax2.set_ylabel("Perplexity", color=color_herb, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color_herb)

        # X 轴标签
        xlabels = [f"{v:.4g}" if v > 0 else "0" for v in values]
        ax1.set_xticks(x)
        ax1.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=9)
        ax1.set_xlabel(summary["display_name"], fontsize=11)

        # 在 value=0 处画虚线 (基线)
        if 0.0 in values:
            baseline_idx = list(values).index(0.0)
            baseline_map = map_means[baseline_idx]
            ax1.axhline(y=baseline_map, color=color_map, linestyle='--', alpha=0.3, linewidth=1)

        # 图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='upper left', fontsize=9, framealpha=0.8)

        ax1.set_title(summary["display_name"], fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.2)

    # 隐藏空白子图
    for idx in range(len(available), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()

    # 保存
    fig_path = output_dir / "sensitivity_grid.pdf"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    fig_path_png = output_dir / "sensitivity_grid.png"
    fig.savefig(fig_path_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 图表已保存: {fig_path}")
    print(f"📊 图表已保存: {fig_path_png}")


def _generate_latex_table(summaries: Dict, output_dir: Path):
    """生成 LaTeX 格式的汇总表."""
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Hyperparameter sensitivity analysis. "
                 r"Each regularization term is added individually to the reconstruction-only baseline. "
                 r"Values are mean $\pm$ std over 3 seeds (K=30).}")
    lines.append(r"\label{tab:sensitivity}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{ll|ccc|cc}")
    lines.append(r"\toprule")
    lines.append(r"Parameter & Value & MAP$_\text{avg}$ & MAP$_\text{s2h}$ & MAP$_\text{h2s}$ "
                 r"& Herb PPL$_\text{EM}$ & Sym PPL$_\text{EM}$ \\")
    lines.append(r"\midrule")

    param_order = ["lambda_know", "beta_pair", "lambda_graph", "gamma_l1"]

    for param_name in param_order:
        if param_name not in summaries:
            continue
        summary = summaries[param_name]
        display = summary["display_name"]
        results = summary["results"]

        first_row = True
        for r in results:
            if r["n_completed"] == 0:
                continue
            m = r.get("mean", {})
            s = r.get("std", {})

            def _fmt(key, prec=4):
                if key in m and np.isfinite(m[key]):
                    return f"{m[key]:.{prec}f}" + r"$\pm$" + f"{s.get(key, 0):.{prec}f}"
                return "---"

            val_str = r["value_str"]
            if r["value"] == 0.0:
                val_str = r"0 (baseline)"

            # 左列只在第一行显示参数名
            left = display if first_row else ""
            first_row = False

            lines.append(
                f"  {left} & {val_str} & "
                f"{_fmt('valid_map_avg')} & {_fmt('valid_map_sym2herb')} & {_fmt('valid_map_herb2sym')} & "
                f"{_fmt('herb_pred_ppl_em_prob', 1)} & {_fmt('symptom_pred_ppl_em_prob', 1)} \\\\"
            )
        lines.append(r"\midrule")

    # 去掉最后一个 \midrule, 替换为 \bottomrule
    if lines and lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    tex_content = "\n".join(lines)
    tex_path = output_dir / "sensitivity_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"📋 LaTeX 表格已保存: {tex_path}")

    # 也打印到控制台
    print("\n" + tex_content)


def _print_text_summary(summaries: Dict):
    """打印纯文本汇总."""
    param_order = ["lambda_know", "beta_pair", "lambda_graph", "gamma_l1"]

    for param_name in param_order:
        if param_name not in summaries:
            continue
        summary = summaries[param_name]
        print(f"\n{'=' * 90}")
        print(f"  {summary['display_name']}")
        print(f"{'=' * 90}")

        header = f"{'Value':>12} | {'MAP_avg':>14} | {'s2h_MAP':>14} | {'h2s_MAP':>14} | {'herb_PPL':>14} | {'sym_PPL':>14} | {'n':>3}"
        print(header)
        print("-" * len(header))

        for r in summary["results"]:
            if r["n_completed"] == 0:
                continue
            m = r.get("mean", {})
            s = r.get("std", {})

            def _f(key):
                if key in m and np.isfinite(m[key]):
                    return f"{m[key]:.4f}±{s.get(key, 0):.4f}"
                return "N/A"

            print(f"{r['value_str']:>12} | {_f('valid_map_avg'):>14} | "
                  f"{_f('valid_map_sym2herb'):>14} | {_f('valid_map_herb2sym'):>14} | "
                  f"{_f('herb_pred_ppl_em_prob'):>14} | {_f('symptom_pred_ppl_em_prob'):>14} | "
                  f"{r['n_completed']:>3}")

        # 找最优值
        best_map_val, best_map_score = None, -1
        for r in summary["results"]:
            m = r.get("mean", {})
            map_val = m.get("valid_map_avg", -1)
            if np.isfinite(map_val) and map_val > best_map_score:
                best_map_score = map_val
                best_map_val = r["value"]

        baseline_map = None
        for r in summary["results"]:
            if r["value"] == 0.0 and r["n_completed"] > 0:
                baseline_map = r.get("mean", {}).get("valid_map_avg")
                break

        if best_map_val is not None and baseline_map is not None:
            delta = best_map_score - baseline_map
            pct = delta / baseline_map * 100 if baseline_map > 0 else 0
            print(f"\n  ★ 最优 MAP@{best_map_val}: {best_map_score:.4f} "
                  f"(vs baseline {baseline_map:.4f}, Δ={delta:+.4f}, {pct:+.2f}%)")


def parse_args():
    p = argparse.ArgumentParser(description="超参数敏感性分析可视化")
    p.add_argument("--input_dir", default="paper_results/sensitivity",
                    help="输入目录 (run_hyperparam_sensitivity 的输出)")
    p.add_argument("--output_dir", default=None,
                    help="输出目录 (默认与 input_dir 相同)")
    return p.parse_args()


def main():
    args = parse_args()

    input_dir = (PROJECT_ROOT / args.input_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if args.output_dir else input_dir

    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        print(f"   请先运行: python scripts/revision/run_hyperparam_sensitivity.py")
        sys.exit(1)

    summaries = _load_summaries(input_dir)
    if not summaries:
        print(f"❌ 未找到实验结果")
        sys.exit(1)

    print(f"📂 加载了 {len(summaries)} 个参数组的结果")

    # 文本汇总
    _print_text_summary(summaries)

    # 图表
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_sensitivity_grid(summaries, output_dir)

    # LaTeX 表格
    _generate_latex_table(summaries, output_dir)

    print(f"\n✅ 可视化完成, 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
