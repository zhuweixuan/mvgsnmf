#!/usr/bin/env python3
"""TF-IDF 消融实验 (Revision Experiment: TF-IDF Ablation).

针对不同输入表示 (TF-IDF vs TF-only vs Binary) 的对比实验,
验证 TF-IDF 加权对推荐效果和主题提取的必要性。
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gsnmf.evaluator import evaluate_all
from gsnmf.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tfidf_ablation")

REQUIRED_FILES = ["summary.json", "factors.npz", "metrics.json", "config.yaml"]

# =========================================================================
# 参数扫描定义
# =========================================================================

@dataclass
class TfidfConfig:
    name: str               # 显示名称
    dir_name: str           # 目录名称
    enabled: bool
    use_idf: bool


TFIDF_CONFIGS = [
    TfidfConfig("TF-IDF (Baseline)", "tfidf", True, True),
    TfidfConfig("TF only (No IDF)", "tf_only", True, False),
    TfidfConfig("Binary (Count)", "binary", False, False),
]

# 关键指标列表 (包含全额指标)
KEY_METRICS = [
    # MAP
    "valid_map_avg",
    "valid_map_sym2herb",
    "valid_map_herb2sym",
    # Ranking metrics @ 5
    "valid_sym2herb_p@5",
    "valid_herb2sym_p@5",
    # Ranking metrics @ 10
    "valid_sym2herb_p@10",
    "valid_sym2herb_ndcg@10",
    "valid_herb2sym_p@10",
    "valid_herb2sym_ndcg@10",
    # Perplexity (EM-based for fair comparison)
    "herb_pred_ppl_em_prob",
    "symptom_pred_ppl_em_prob",
    "herb_pred_ppl_em",
    "symptom_pred_ppl_em",
    # Coherence
    "topic_coherence",
]


# =========================================================================
# 工具函数
# =========================================================================

def _deep_copy_cfg(cfg: Dict) -> Dict:
    return yaml.safe_load(yaml.safe_dump(cfg, allow_unicode=True))


def _is_completed_run(run_dir: Path) -> bool:
    return run_dir.exists() and run_dir.is_dir() and all(
        (run_dir / f).exists() for f in REQUIRED_FILES
    )


def _write_json_atomic(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(path)


def _build_cfg_for_tfidf(base_cfg: Dict, seed: int, K: int, tfidf_config: TfidfConfig) -> Dict:
    """构建包含指定 tf-idf 设置的 config."""
    cfg = _deep_copy_cfg(base_cfg)
    cfg["seed"] = int(seed)
    cfg["K"] = int(K)
    cfg.setdefault("split", {})["seed"] = int(seed)

    # 保持除 tfidf 之外的所有最优正则参数
    tf_cfg = cfg.setdefault("tfidf", {})
    tf_cfg["enabled"] = tfidf_config.enabled
    tf_cfg["use_idf"] = tfidf_config.use_idf

    cfg["model_name"] = f"tfidf_ablation_{tfidf_config.dir_name}_seed{seed}"
    return cfg


def _run_one(cfg: Dict, run_dir: Path):
    """训练一次并保存到指定目录."""
    trainer = Trainer(cfg)
    trainer.setup()

    def eval_fn(model, split_data, C_hh):
        return evaluate_all(
            model, split_data, C_hh,
            compute_perplexity=True,
            eval_seed=2025,
            compute_dose=False,
            dirichlet_alpha=trainer.dirichlet_alpha,
            tfidf_decouple=trainer.tfidf_decouple,
            hypergraph_bundle=getattr(trainer, "_hyper_bundle", None),
            H_s_ppl=getattr(trainer, "_H_s_ppl", None),
            H_h_ppl=getattr(trainer, "_H_h_ppl", None),
        )

    trainer.train(eval_fn=eval_fn)

    # 移动 artifacts 到目标目录
    src = Path(trainer.run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.move(str(src), str(run_dir))


def _load_summary_metrics(run_dir: Path) -> Optional[Dict]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        metrics = summary.get("test_metrics", {})
        if not metrics:
            metrics = summary
        return metrics
    except Exception:
        return None


def _print_table(all_summaries: Dict):
    print(f"\n{'=' * 150}")
    print(f"  TF-IDF Ablation Summary")
    print(f"{'=' * 150}")

    header_cols = ["Config", "s2h_MAP", "s2h_P@10", "s2h_NDCG@10",
                   "h2s_MAP", "h2s_P@10", "h2s_NDCG@10",
                   "herb_PPL_em", "sym_PPL_em", "Coherence", "n"]
    header = " | ".join(f"{c:>12s}" for c in header_cols)
    print(header)
    print("-" * len(header))

    for config_name, summary in all_summaries.items():
        m = summary.get("mean", {})
        s = summary.get("std", {})
        n = summary.get("n_completed", 0)

        def _fmt(key):
            if key in m:
                return f"{m[key]:.4f}±{s.get(key, 0.0):.4f}"
            return "  N/A"

        row = [
            f"{config_name:>12s}",
            f"{_fmt('valid_map_sym2herb'):>12s}",
            f"{_fmt('valid_sym2herb_p@10'):>12s}",
            f"{_fmt('valid_sym2herb_ndcg@10'):>12s}",
            f"{_fmt('valid_map_herb2sym'):>12s}",
            f"{_fmt('valid_herb2sym_p@10'):>12s}",
            f"{_fmt('valid_herb2sym_ndcg@10'):>12s}",
            f"{_fmt('herb_pred_ppl_em_prob'):>12s}",
            f"{_fmt('symptom_pred_ppl_em_prob'):>12s}",
            f"{_fmt('topic_coherence'):>12s}",
            f"{n:>12d}"
        ]
        print(" | ".join(row))


# =========================================================================
# 主入口
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="TF-IDF Ablation Experiment")
    p.add_argument("--base_config", default="config/best_v4.yaml")
    p.add_argument("--output_root", default="artifacts/revision_tfidf")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--K", type=int, default=30)
    p.add_argument("--resume", action="store_true", help="跳过已完成的 run")
    p.add_argument("--summary_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    cfg_path = (PROJECT_ROOT / args.base_config).resolve()
    out_root = (PROJECT_ROOT / args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    all_summaries = {}
    total_jobs = len(TFIDF_CONFIGS) * len(args.seeds)
    job_idx = 0

    if args.summary_only:
        for tconf in TFIDF_CONFIGS:
            out_dir = out_root / tconf.dir_name
            all_metrics = {k: [] for k in KEY_METRICS}
            n_completed = 0
            for seed in args.seeds:
                run_dir = out_dir / f"seed_{seed}"
                if _is_completed_run(run_dir):
                    metrics = _load_summary_metrics(run_dir)
                    if metrics:
                        n_completed += 1
                        for k in KEY_METRICS:
                            if k in metrics and np.isfinite(metrics[k]):
                                all_metrics[k].append(float(metrics[k]))

            summary = {"n_completed": n_completed, "mean": {}, "std": {}}
            for k in KEY_METRICS:
                if all_metrics[k]:
                    summary["mean"][k] = float(np.mean(all_metrics[k]))
                    summary["std"][k] = float(np.std(all_metrics[k]))
            all_summaries[tconf.dir_name] = summary

        _print_table(all_summaries)
        _write_json_atomic(out_root / "summary.json", all_summaries)
        return

    for tconf in TFIDF_CONFIGS:
        out_dir = out_root / tconf.dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for seed in args.seeds:
            job_idx += 1
            run_dir = out_dir / f"seed_{seed}"

            if args.resume and _is_completed_run(run_dir):
                print(f"[{job_idx}/{total_jobs}] SKIP | config={tconf.dir_name} seed={seed}")
                continue

            print("=" * 80)
            print(f"[{job_idx}/{total_jobs}] RUN | config={tconf.name} | seed={seed} | K={args.K}")
            print(f"  Output: {run_dir}")

            cfg = _build_cfg_for_tfidf(base_cfg, seed=seed, K=args.K, tfidf_config=tconf)
            t0 = time.time()
            try:
                _run_one(cfg, run_dir)
                elapsed = time.time() - t0
                print(f"  ✅ 完成 ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  ❌ 失败 ({elapsed:.1f}s): {e}")

    # ===== 生成汇总 =====
    main_sys_args_backup = sys.argv.copy()
    sys.argv = [sys.argv[0], "--summary_only", "--output_root", args.output_root]
    main()
    sys.argv = main_sys_args_backup


if __name__ == "__main__":
    main()
