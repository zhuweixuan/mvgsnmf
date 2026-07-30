#!/usr/bin/env python3
"""PPL vs MAP 机制分析 (Revision Experiment: Scatter Correlation).

分析困惑度(PPL)与主推指标(MAP)的内部关联, 
通过数据图表证明在何种情况下两者一致，何种情况下背离，
以加强机制层面的讨论深度。
"""

import json
import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def parse_args():
    p = argparse.ArgumentParser(description="Plot PPL vs MAP scatter correlation")
    p.add_argument("--json", default="paper_results/unified_tables/unified_metrics_all.json", help="统一评估JSON")
    p.add_argument("--output_dir", default="paper_results/figures", help="图表输出目录")
    return p.parse_args()


def plot_correlation(x_data, y_data, labels, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(8, 6))
    
    # 因为 PPL 可能差别巨大（NMF有的大到几万），我们最好将 PPL 取对数，或者过滤异常值
    x_arr = np.array(x_data)
    y_arr = np.array(y_data)
    
    # 过滤掉无法评估的极值 (PPL > 1000) - NMF 裸矩阵会发散
    valid = x_arr < 1000
    x_valid = x_arr[valid]
    y_valid = y_arr[valid]
    labels_valid = [labels[i] for i in range(len(labels)) if valid[i]]
    
    if len(x_valid) < 3:
        print(f"[{title}] ⚠️ 有效数据点太少 ({len(x_valid)}), 跳过作图。")
        return
        
    plt.scatter(x_valid, y_valid, alpha=0.7, edgecolors='w', s=80)
    
    # Add annotations (只取一部分避免重叠)
    for i, label in enumerate(labels_valid):
        # 简单避免重叠：每个 label 只随机展示部分，或者如果是关键模型则标出
        if 'ptm' in label or 'active' in label or 'recon_only' in label.replace('recon_only_ph_ps_nonneg', 'recon_abl'):
            # 简化名字
            short_lbl = label.replace("active_graph_h__graph_s__l1__pair__know_hs", "Full_Model")
            short_lbl = short_lbl.replace("active_graph_h__graph_s", "+Graphs")
            short_lbl = short_lbl.replace("recon_only_mvgsnmf", "ReconBase")
            plt.annotate(short_lbl, (x_valid[i], y_valid[i]), fontsize=8, alpha=0.7)

    # 计算相关系数
    pearson_corr, p_val = pearsonr(x_valid, y_valid)
    spearman_corr, sp_p_val = spearmanr(x_valid, y_valid)
    
    # Fit regression line
    m, b = np.polyfit(x_valid, y_valid, 1)
    plt.plot(x_valid, m*x_valid + b, color='red', linestyle='--', alpha=0.5)
    
    plt.title(f"{title}\nPearson Correlation: {pearson_corr:.3f} (p={p_val:.3e})", fontsize=11)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 已保存图表: {save_path.name}")
    print(f"   Pearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}")


def main():
    args = parse_args()
    json_path = (PROJECT_ROOT / args.json).resolve()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"❌ 找不到数据文件: {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 我们关注 K=30 的各种模型
    target_k = "30"
    
    model_labels = []
    
    sym2herb_map = []
    herb2sym_map = []
    
    sym_ppl = []  # 用 PTM 原版 和 NMF_em
    herb_ppl = []
    
    for model_name, k_dict in data.items():
        if target_k not in k_dict:
            continue
        d = k_dict[target_k]
        
        # 必须都存在 P@5/MAP 才取（PTM 只有 P@5, 我们这里如果有 MAP 用 MAP, 没有用 P@5 替代）
        # 为了公平, 我们可以用 sym2herb_P@5 vs PPL
        if "sym2herb_P@5" not in d or "Topic_Coherence" not in d:
            continue
            
        sym2herb_metric = d.get("sym2herb_MAP", d.get("sym2herb_P@5"))["mean"]
        herb2sym_metric = d.get("herb2sym_MAP", d.get("herb2sym_P@5"))["mean"]
        
        # PPL 取决于是PTM还是NMF
        if "sym_pred_PPL_em" in d:
            s_ppl = d["sym_pred_PPL_em"]["mean"]
            h_ppl = d["herb_pred_PPL_em"]["mean"]
        else:
            s_ppl = d["sym_pred_PPL"]["mean"]
            h_ppl = d["herb_pred_PPL"]["mean"]
            
        model_labels.append(model_name)
        sym2herb_map.append(sym2herb_metric)
        herb2sym_map.append(herb2sym_metric)
        sym_ppl.append(s_ppl)
        herb_ppl.append(h_ppl)

    # 生成 Symp 预测 Herb 方向的 Correlation 图
    plot_correlation(
        x_data=sym_ppl, 
        y_data=sym2herb_map,
        labels=model_labels,
        title="Mechanism Analysis: Symptom PPL vs Symptom->Herb Prescribing (K=30)",
        xlabel="Symptom Perplexity (Lower is better modeling)",
        ylabel="Prescription Metric (MAP / P@5)",
        save_path=out_dir / "scatter_sym_ppl_vs_metric.png"
    )

    # 生成 Herb 预测 Symp 方向的 Correlation 图
    plot_correlation(
        x_data=herb_ppl, 
        y_data=herb2sym_map,
        labels=model_labels,
        title="Mechanism Analysis: Herb PPL vs Herb->Symptom Reverse Prediction (K=30)",
        xlabel="Herb Perplexity (Lower is better modeling)",
        ylabel="Reverse Prediction Metric (MAP / P@5)",
        save_path=out_dir / "scatter_herb_ppl_vs_metric.png"
    )


if __name__ == "__main__":
    main()
