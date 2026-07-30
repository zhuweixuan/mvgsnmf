#!/usr/bin/env python3
"""图构造与药对阈值敏感性分析 (Revision Experiment: Structure Sensitivity).

扫描构图超参 KNN_k 和 药对支持度 Min_Support 
证明模型在这些硬编码阈值变化下的表现依然稳定。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 清理缓存，防止不同阈值的图/药对复用相同的 load_all 缓存
from gsnmf.trainer import Trainer
from gsnmf.evaluator import evaluate_all
from scripts.run_ablation_multiseed import build_full_loss_cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("structure_sensitivity")

REQUIRED_FILES = ["summary.json", "factors.npz", "metrics.json", "config.yaml"]

# 关键指标列表
KEY_METRICS = [
    "valid_map_avg",
    "valid_map_sym2herb",
    "valid_map_herb2sym",
    "topic_coherence",
    "herb_pred_ppl_em_prob",
    "symptom_pred_ppl_em_prob"
]

@dataclass
class StructSweep:
    name: str                       
    display_name: str               
    param_keys: List[str]           # 格式: "graph.knn_k" 等
    values: List[float]             


DEFAULT_SWEEPS = [
    StructSweep(
        name="knn_k",
        display_name="Graph KNN Neighbors (k)",
        param_keys=["graph.knn_k"],
        values=[5, 10, 20, 50],
    ),
    StructSweep(
        name="pair_min_support",
        display_name="Pair Min Support Threshold",
        param_keys=["graph.pair_min_support"],
        values=[5, 10, 20, 50],
    ),
]


def _deep_copy_cfg(cfg: Dict) -> Dict:
    return yaml.safe_load(yaml.safe_dump(cfg, allow_unicode=True))


def _set_nested_key(cfg: Dict, key_path: str, value: Any):
    keys = key_path.split('.')
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _apply_sweep(cfg: Dict, sweep: StructSweep, value: float) -> Dict:
    c = _deep_copy_cfg(cfg)
    for key in sweep.param_keys:
        _set_nested_key(c, key, int(value))
    
    # 强制重新触发 data_loader 以应用 threshold
    # 虽然目前 Trainer 有缓存机制, 在循环内最好手动清理
    Trainer._cached_data = None
    Trainer._cached_data_key = None
    
    c["model_name"] = f"struct_sens_{sweep.name}_{int(value)}"
    return c


def _is_completed_run(run_dir: Path, expected_cfg: Dict) -> bool:
    if not (
        run_dir.exists()
        and run_dir.is_dir()
        and all((run_dir / f).exists() for f in REQUIRED_FILES)
    ):
        return False
    try:
        with open(run_dir / "config.yaml", encoding="utf-8") as f:
            saved_cfg = yaml.safe_load(f)
    except Exception:
        return False
    # Do not resume the obsolete reconstruction-only sweep merely because its
    # files exist. A run is reusable only when its complete config matches.
    return saved_cfg == expected_cfg


def _validate_full_profile(cfg: Dict):
    expected_switches = {
        "ph": True,
        "ps": True,
        "pd": False,
        "graph_h": True,
        "graph_s": True,
        "l1": True,
        "contra": False,
        "pair": True,
        "know_hs": True,
        "hyper_h": False,
        "hyper_var": False,
        "hyper_mean": False,
    }
    switches = cfg.get("loss_switches", {})
    bad_switches = {
        key: (switches.get(key), expected)
        for key, expected in expected_switches.items()
        if switches.get(key) is not expected
    }
    expected_weights = {
        "lambda_h": 1.0,
        "lambda_s": 1.0,
        "beta_pair": 0.05,
        "lambda_know": 0.2,
        "gamma_g": 0.1,
        "gamma_h": 0.1,
        "gamma_s": 0.1,
        "gamma_d": 0.01,
    }
    bad_weights = {
        key: (cfg.get(key), expected)
        for key, expected in expected_weights.items()
        if float(cfg.get(key, float("nan"))) != expected
    }
    if bad_switches or bad_weights:
        raise ValueError(
            "Structure sensitivity requires the Full loss profile; "
            f"switch mismatches={bad_switches}, weight mismatches={bad_weights}"
        )


def _write_json_atomic(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(path)


def _run_one(cfg: Dict, run_dir: Path):
    # Keep every source of pseudo-randomness aligned with the matched run seed.
    # Model initialization currently uses NumPy, while this also guards future
    # GPU-side randomized operations without changing the model objective.
    if str(cfg.get("device", "cpu")).lower() == "gpu":
        import cupy as cp

        cp.random.seed(int(cfg.get("seed", 42)))

    trainer = Trainer(cfg)
    trainer.setup()
    logger.info(
        "Runtime profile: backend=%s, pretrain=%s, max_iter=%s, "
        "pair_shape=%s, pair_nnz=%s",
        trainer.model.xp.__name__,
        cfg.get("training", {}).get("pretrain_iters"),
        cfg.get("training", {}).get("max_iter"),
        trainer.split_data.train.X_pair.shape,
        trainer.split_data.train.X_pair.nnz,
    )

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


def _print_sweep_table(sweep_summary: Dict):
    name = sweep_summary["display_name"]
    print(f"\n{'=' * 100}")
    print(f"  {name}")
    print(f"{'=' * 100}")

    header_cols = ["Value", "MAP_avg", "s2h_MAP", "h2s_MAP",
                   "herb_PPL_em", "sym_PPL_em", "coherence", "n"]
    header = " | ".join(f"{c:>12s}" for c in header_cols)
    print(header)
    print("-" * len(header))

    for r in sweep_summary["results"]:
        val_str = r["value_str"]
        n = r["n_completed"]
        m = r.get("mean", {})
        s = r.get("std", {})

        def _fmt(key):
            if key in m:
                return f"{m[key]:.4f}±{s.get(key, 0.0):.4f}"
            return "  N/A"

        row = [
            f"{val_str:>12s}",
            f"{_fmt('valid_map_avg'):>12s}",
            f"{_fmt('valid_map_sym2herb'):>12s}",
            f"{_fmt('valid_map_herb2sym'):>12s}",
            f"{_fmt('herb_pred_ppl_em_prob'):>12s}",
            f"{_fmt('symptom_pred_ppl_em_prob'):>12s}",
            f"{_fmt('topic_coherence'):>12s}",
            f"{n:>12d}",
        ]
        print(" | ".join(row))


def parse_args():
    p = argparse.ArgumentParser("特征图与药对提取阈值鲁棒性分析")
    p.add_argument("--base_config", default="config/best_v4.yaml")
    p.add_argument(
        "--output_root",
        default="artifacts/revision_struct_sens_full_gpu",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    p.add_argument("--K", type=int, default=30)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--pretrain_iters", type=int, default=100)
    p.add_argument(
        "--data_root",
        default=None,
        help="Optional local dataset root overriding the base YAML.",
    )
    p.add_argument(
        "--knowledge_file",
        default=None,
        help="Optional local herb-symptom knowledge file.",
    )
    p.add_argument(
        "--eval_every",
        type=int,
        default=300,
        help="Run the expensive full evaluation every N main iterations.",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--summary_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = (PROJECT_ROOT / args.base_config).resolve()
    out_root = (PROJECT_ROOT / args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    base_cfg = build_full_loss_cfg(raw_cfg)
    base_cfg["device"] = args.device
    # Make the nominal pair threshold explicit in every saved configuration.
    # This also makes the k=10/support=20 crossover a clean repeatability
    # control between the two one-factor-at-a-time sweeps.
    base_cfg.setdefault("graph", {}).setdefault("pair_min_support", 20)
    if args.data_root:
        base_cfg["data_root"] = str(Path(args.data_root).resolve())
    if args.knowledge_file:
        base_cfg.setdefault("files", {})["symptom_herb_knowledge"] = str(
            Path(args.knowledge_file).resolve()
        )
    train_cfg = base_cfg.setdefault("training", {})
    train_cfg["pretrain_iters"] = args.pretrain_iters
    train_cfg["eval_every"] = args.eval_every
    _validate_full_profile(base_cfg)
    logger.info(
        "Reproducibility controls: CUBLAS_WORKSPACE_CONFIG=%s, "
        "PYTHONHASHSEED=%s, seeds=%s",
        os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<unset>"),
        os.environ.get("PYTHONHASHSEED", "<unset>"),
        args.seeds,
    )

    sweeps = DEFAULT_SWEEPS

    if args.summary_only:
        all_summaries = {}
        for sweep in sweeps:
            sweep_dir = out_root / sweep.name
            
            summary = {
                "display_name": sweep.display_name,
                "results": [],
            }
            
            for val in sweep.values:
                val_str = str(val)
                val_metrics = {k: [] for k in KEY_METRICS}
                n_comp = 0
                for seed in args.seeds:
                    run_dir = sweep_dir / f"seed_{seed}" / f"val_{val_str}"
                    metrics = _load_summary_metrics(run_dir)
                    if metrics:
                        n_comp += 1
                        for k in KEY_METRICS:
                            if k in metrics:
                                val_metrics[k].append(float(metrics[k]))
                                
                val_res = {
                    "value_str": val_str,
                    "n_completed": n_comp,
                    "mean": {}, "std": {}
                }
                for k in KEY_METRICS:
                    if val_metrics[k]:
                        val_res["mean"][k] = float(np.mean(val_metrics[k]))
                        val_res["std"][k] = float(np.std(val_metrics[k]))
                summary["results"].append(val_res)
                
            all_summaries[sweep.name] = summary
            _print_sweep_table(summary)
            
        _write_json_atomic(out_root / "summary.json", all_summaries)
        return

    total_jobs = sum(len(s.values) * len(args.seeds) for s in sweeps)
    job_idx = 0

    for sweep in sweeps:
        sweep_dir = out_root / sweep.name
        sweep_dir.mkdir(parents=True, exist_ok=True)

        for seed in args.seeds:
            for val in sweep.values:
                job_idx += 1
                val_str = str(val)
                run_dir = sweep_dir / f"seed_{seed}" / f"val_{val_str}"

                cfg = _apply_sweep(base_cfg, sweep, val)
                cfg["seed"] = seed
                cfg["K"] = args.K
                cfg.setdefault("split", {})["seed"] = seed
                _validate_full_profile(cfg)

                if args.resume and _is_completed_run(run_dir, cfg):
                    print(f"[{job_idx}/{total_jobs}] SKIP | {sweep.name}={val_str} seed={seed}")
                    continue

                print("=" * 80)
                print(f"[{job_idx}/{total_jobs}] RUN | {sweep.name}={val_str} | seed={seed} | K={args.K}")

                t0 = time.time()
                try:
                    _run_one(cfg, run_dir)
                    print(f"  ✅ 完成 ({time.time()-t0:.1f}s)")
                except Exception as e:
                    print(f"  ❌ 失败 ({time.time()-t0:.1f}s): {e}")

    # ===== 生成汇总 =====
    main_sys_args_backup = sys.argv.copy()
    sys.argv = [
        sys.argv[0],
        "--summary_only",
        "--base_config",
        args.base_config,
        "--output_root",
        args.output_root,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--device",
        args.device,
        "--pretrain_iters",
        str(args.pretrain_iters),
        "--eval_every",
        str(args.eval_every),
    ]
    if args.data_root:
        sys.argv.extend(["--data_root", args.data_root])
    if args.knowledge_file:
        sys.argv.extend(["--knowledge_file", args.knowledge_file])
    main()
    sys.argv = main_sys_args_backup


if __name__ == "__main__":
    main()
