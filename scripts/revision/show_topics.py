#!/usr/bin/env python3
"""展示具有代表性的中医主题 (Revision Experiment: Topic Display).

读取指定的最佳模型 factors, 调用 explain_topic 输出具有代表性的主题。
展现模型在推荐的同时, 学到的中间层 (潜在主题) 如何具备中药配伍的解释力。
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gsnmf.data_loader import load_all
from gsnmf.split import split_data
from gsnmf.recommender import explain_topic
from gsnmf.schemas import ModelFactors


def parse_args():
    p = argparse.ArgumentParser(description="Show Representative Topics")
    p.add_argument("--config", default="config/paper_full.yaml", help="基础配置")
    p.add_argument("--factors", required=True, help="factors.npz 路径")
    p.add_argument("--top_k", type=int, default=5, help="展示前几个代表性主题")
    p.add_argument("--top_items", type=int, default=10, help="每个主题展示的 top herbs/symptoms")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = (PROJECT_ROOT / args.config).resolve()
    factors_path = (PROJECT_ROOT / args.factors).resolve()

    if not factors_path.exists():
        print(f"❌ 找不到模型文件: {factors_path}")
        sys.exit(1)

    print(f"加载配置: {cfg_path.name}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 我们需要加载数据的字典，以获取名字
    # cfg 里会写明 data_root 和 files
    print("加载数据集 (获取药材症状名词)...")
    all_data = load_all(
        cfg["data_root"],
        cfg.get("files", {}),
        load_dosage=bool(
            cfg.get("loss_switches", {}).get("pd", False)
        ),
    )
    split_res = split_data(all_data, cfg)
    herb_names = split_res.meta.herb_names
    symptom_names = split_res.meta.symptom_names

    print(f"加载因子: {factors_path}")
    npz = np.load(factors_path)
    factors = ModelFactors(
        G_p=npz['G_p'] if 'G_p' in npz else npz['G_p_train'],
        H_h=npz['H_h'],
        H_s=npz['H_s'],
        D_h=npz.get('D_h', np.zeros_like(npz['H_h'])),
    )
    
    K_total = factors.H_h.shape[1]
    
    # 基于某种启发式选取"最有代表性"的主题展示
    # 比如: 选药材使用频率分布较不均匀的，或者处方使用较多的
    # 此处最简单的：根据每个主题在处方中的平均表达程度降序，选出前 K 个最主要主题
    # 或者计算 topic coherence
    
    gp_mean = factors.G_p.mean(axis=0)
    top_topics_idx = np.argsort(gp_mean)[-args.top_k:][::-1]

    print("\n" + "="*80)
    print(f"  代表性主题展示 (Top {args.top_k} from Total {K_total})")
    print("="*80)

    for i, t_idx in enumerate(top_topics_idx):
        info = explain_topic(t_idx, factors, herb_names, symptom_names, top_n=args.top_items)
        
        # 组装展示文本
        top_h = [f"{h}({w:.3f})" for h, w in info["top_herbs"]]
        top_s = [f"{s}({w:.3f})" for s, w in info["top_symptoms"]]
        
        print(f"\n👉 主题 {t_idx} (使用度={gp_mean[t_idx]:.4f}):")
        print(f"  方剂核心症状: {', '.join([s for s, w in info['top_symptoms'][:5]])}")
        print(f"  靶向主治药材: {', '.join([h for h, w in info['top_herbs'][:5]])}")
        print(f"  -- 完整 Symptoms ({args.top_items}): {', '.join(top_s)}")
        print(f"  -- 完整 Herbs ({args.top_items})   : {', '.join(top_h)}")

    print("\n提示: 主题输出已按在此数据集中的表达频率排降序。如有需要，可直接提取上述关键词填入论文 Table 中进行医理分析。")

if __name__ == "__main__":
    main()
