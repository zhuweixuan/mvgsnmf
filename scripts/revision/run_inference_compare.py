#!/usr/bin/env python3
"""推断方式对比分析 (Revision Experiment: Inference Method Comparison).

仅针对最佳模型重写加载和评估，
对比基于 非负最小二乘(NNLS) 和 朴素加权求和(DOT) 的推断差距，
论证推荐中正交潜空间的解码必须要解稀疏约束。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gsnmf.data_loader import load_all
from gsnmf.split import split_data
from gsnmf.evaluator import evaluate_all
from gsnmf.schemas import ModelFactors

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/paper_full.yaml")
    p.add_argument("--factors", required=True)
    p.add_argument("--output", default="artifacts/revision_tables/inference_methods.json")
    args = p.parse_args()

    cfg_path = (PROJECT_ROOT / args.config).resolve()
    factors_path = (PROJECT_ROOT / args.factors).resolve()
    
    if not factors_path.exists():
        print(f"❌ 找不到因子文件: {factors_path}")
        return

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("加载数据...")
    all_data = load_all(
        cfg["data_root"],
        cfg.get("files", {}),
        load_dosage=bool(
            cfg.get("loss_switches", {}).get("pd", False)
        ),
    )
    split = split_data(all_data, cfg)

    print(f"加载模型因子: {factors_path}")
    npz = np.load(factors_path)
    model = ModelFactors(
        G_p=np.zeros((1, 1)), # mock since we only need H_h and H_s for valid evaluation
        H_h=npz['H_h'],
        H_s=npz['H_s'],
        D_h=np.zeros_like(npz['H_h']), # mock
        H_pair=np.zeros_like(npz['H_h']) # mock
    )
    # mock C_hh as all zeros to avoid NoneType
    import scipy.sparse as sp
    C_hh = sp.csr_matrix((793, 793))
    model.sw = {"ps": True, "ph": True, "pd": False, "pair": False, "hyper_h": False}

    from gsnmf.ppl_refine import reestimate_Hs_em_vectorized, reestimate_Hh_em_vectorized
    methods = ["nnls", "dot", "ols"]
    results = {}
    
    # 获取训练集频率
    for meth in methods:
        print(f"=====================================")
        print(f"  评估推断方法: {meth.upper()}")
        print(f"=====================================")
        
        print("--- 正在重新重估 PPL 概率分布 (EM Refine) ---")
        X_ph_dense = split.train.X_ph.toarray() if hasattr(split.train.X_ph, "toarray") else split.train.X_ph
        X_ps_dense = split.train.X_ps.toarray() if hasattr(split.train.X_ps, "toarray") else split.train.X_ps
        H_s_ppl = reestimate_Hs_em_vectorized(model.H_h, X_ph_dense, X_ps_dense, infer_method=meth)
        H_h_ppl = reestimate_Hh_em_vectorized(model.H_s, X_ps_dense, X_ph_dense, infer_method=meth)

        
        metrics = evaluate_all(
            model, split, C_hh,
            compute_perplexity=True, # perplexity evaluates H_h / H_s directly
            eval_seed=2025,
            compute_dose=False,
            dirichlet_alpha=cfg.get("model", {}).get("dirichlet_alpha", 0.0),
            infer_method=meth,
            H_s_ppl=H_s_ppl,
            H_h_ppl=H_h_ppl
        )
        
        results[meth] = {
            "valid_map_avg": metrics.get("valid_map_avg", 0.0),
            "s2h_MAP": metrics.get("valid_map_sym2herb", 0.0),
            "h2s_MAP": metrics.get("valid_map_herb2sym", 0.0),
            "s2h_P_10": metrics.get("sym2herb_p@10", 0.0),
            "h2s_P_10": metrics.get("herb2sym_p@10", 0.0),
            "symptom_PPL": metrics.get("symptom_pred_ppl_em_prob", 0.0),
            "herb_PPL": metrics.get("herb_pred_ppl_em_prob", 0.0)
        }
        print(f"[{meth.upper()}] Avg MAP: {results[meth]['valid_map_avg']:.4f}")
        print(f"          S->H MAP: {results[meth]['s2h_MAP']:.4f}")
        print(f"          H->S MAP: {results[meth]['h2s_MAP']:.4f}")
        print(f"          S->H P@10: {results[meth]['s2h_P_10']:.4f}")
        print(f"          H->S P@10: {results[meth]['h2s_P_10']:.4f}")
        print(f"          Sym PPL: {results[meth]['symptom_PPL']:.2f}")
        print(f"          Herb PPL: {results[meth]['herb_PPL']:.2f}")

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 结果已保存至 {out_path}")

if __name__ == "__main__":
    main()
