#!/usr/bin/env python3
"""超参数敏感性分析实验 (Revision Experiment 1).

审稿人质疑:
  "知识耦合无效"可能是因为 λ_know, β_pair 等参数没调好,
  而非机制上无效。

实验设计:
  以 recon_only (ph+ps) 作为基线, 每次只打开一项正则,
  扫描其权重从极小到极大, 证明即使在最优权重下, 额外增益也有限。

参数组:
  1. lambda_know  (知识耦合)   → 需开启 know_hs
  2. beta_pair    (药对视图)   → 需开启 pair
  3. lambda_graph  (图正则 h+s) → 需开启 graph_h + graph_s
  4. gamma_l1     (L1 稀疏)    → 需开启 l1

输出:
  artifacts/revision_sensitivity/<param_name>/seed_<s>/val_<v>/
  + summary_table.json (汇总 mean±std)

用法:
  python scripts/revision/run_hyperparam_sensitivity.py
  python scripts/revision/run_hyperparam_sensitivity.py --seeds 42 43 44 --resume
  python scripts/revision/run_hyperparam_sensitivity.py --params lambda_know gamma_l1
"""

from __future__ import annotations

import argparse
import json
import logging
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

from gsnmf.evaluator import evaluate_all
from gsnmf.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sensitivity")

REQUIRED_FILES = ["summary.json", "factors.npz", "metrics.json", "config.yaml"]

# =========================================================================
# 参数扫描定义
# =========================================================================

@dataclass
class ParamSweep:
    """一组参数扫描的定义."""
    name: str                       # 参数组名 (用于目录名)
    display_name: str               # 展示名 (用于表头)
    param_keys: List[str]           # 要修改的 cfg key 列表
    loss_switches: Dict[str, bool]  # 额外要打开的 loss switch
    values: List[float]             # 扫描值列表
    # 对于图正则, lambda_h 和 lambda_s 需要同步设置, 用这个系数
    secondary_keys: List[str] = field(default_factory=list)
    secondary_scale: float = 1.0    # secondary = primary * scale


# 定义所有参数扫描组
DEFAULT_SWEEPS: List[ParamSweep] = [
    ParamSweep(
        name="lambda_know",
        display_name="Knowledge Coupling (λ_know)",
        param_keys=["lambda_know"],
        loss_switches={"know_hs": True},
        values=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    ),
    ParamSweep(
        name="beta_pair",
        display_name="Herb Pair View (β_pair)",
        param_keys=["beta_pair"],
        loss_switches={"pair": True},
        values=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
    ),
    ParamSweep(
        name="lambda_graph",
        display_name="Graph Regularization (λ_graph)",
        param_keys=["lambda_h"],
        loss_switches={"graph_h": True, "graph_s": True},
        values=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        secondary_keys=["lambda_s"],
        secondary_scale=1.0,  # λ_s = λ_h
    ),
    ParamSweep(
        name="gamma_l1",
        display_name="L1 Sparsity (γ)",
        param_keys=["gamma_h"],
        loss_switches={"l1": True},
        values=[0.0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        secondary_keys=["gamma_g", "gamma_s", "gamma_d"],
        secondary_scale=1.0,  # gamma_g = gamma_s = gamma_h, gamma_d = gamma_h
    ),
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


def _build_base_recon_cfg(base_cfg: Dict, seed: int, K: int) -> Dict:
    """构建纯重构基线配置 (ph+ps only, 所有正则关闭)."""
    cfg = _deep_copy_cfg(base_cfg)
    cfg["seed"] = int(seed)
    cfg["K"] = int(K)
    cfg.setdefault("split", {})["seed"] = int(seed)

    # 关闭所有正则
    cfg["lambda_h"] = 0.0
    cfg["lambda_s"] = 0.0
    cfg["lambda_hyper"] = 0.0
    cfg["beta_pair"] = 0.0
    cfg["lambda_know"] = 0.0
    cfg["rho"] = 0.0
    cfg["beta"] = 0.0
    cfg["gamma_g"] = 0.0
    cfg["gamma_h"] = 0.0
    cfg["gamma_s"] = 0.0
    cfg["gamma_d"] = 0.0

    sw = cfg.setdefault("loss_switches", {})
    sw.update({
        "ph": True, "ps": True, "pd": False,
        "graph_h": False, "graph_s": False,
        "l1": False, "contra": False, "pair": False,
        "know_hs": False, "hyper_h": False,
        "hyper_var": False, "hyper_mean": False,
    })

    # 使用与 best_v4 相同的训练策略
    tr = cfg.setdefault("training", {})
    tr["disable_early_stop"] = True
    tr["early_stop_metric"] = "valid_map_avg"

    return cfg


def _apply_sweep(cfg: Dict, sweep: ParamSweep, value: float) -> Dict:
    """在基线配置上施加一组参数扫描."""
    c = _deep_copy_cfg(cfg)

    # 设置主参数
    for key in sweep.param_keys:
        c[key] = float(value)

    # 设置关联参数
    for key in sweep.secondary_keys:
        c[key] = float(value * sweep.secondary_scale)

    # L1 的特殊处理: gamma_d 通常设为 gamma_h 的 1/10
    if sweep.name == "gamma_l1":
        c["gamma_d"] = float(value * 0.1)

    # 打开对应的 loss switch
    sw = c.setdefault("loss_switches", {})
    sw.update(sweep.loss_switches)

    # 如果 value == 0, loss switch 仍然打开但权重为 0, 这样保持代码路径一致
    # 但为了效率, value=0 时可以关掉 (等价于基线)
    if value == 0.0:
        for k, v in sweep.loss_switches.items():
            sw[k] = False

    c["model_name"] = f"sensitivity_{sweep.name}_val{value}"

    return c


# =========================================================================
# 单次训练
# =========================================================================

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
    """从 summary.json 中提取关键指标."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        # 取 test_metrics (如果有) 或顶层
        metrics = summary.get("test_metrics", {})
        if not metrics:
            # 可能指标直接在顶层
            metrics = summary
        return metrics
    except Exception:
        return None


# =========================================================================
# 汇总统计
# =========================================================================

# 关键指标列表: 这些指标会被收集到汇总表
KEY_METRICS = [
    "valid_map_sym2herb",
    "valid_map_herb2sym",
    "valid_map_avg",
    "herb_pred_ppl",
    "herb_pred_ppl_prob",
    "symptom_pred_ppl",
    "symptom_pred_ppl_prob",
    "herb_pred_ppl_em",
    "herb_pred_ppl_em_prob",
    "symptom_pred_ppl_em",
    "symptom_pred_ppl_em_prob",
    "topic_coherence",
    "nre_ph",
    "nre_ps",
    "ppl_masked_20_prob",
    "ppl_half_prob",
]


def _generate_sweep_summary(
    sweep: ParamSweep,
    seeds: List[int],
    sweep_dir: Path,
) -> Dict:
    """为一个参数扫描生成汇总统计 (mean ± std)."""
    summary = {
        "param_name": sweep.name,
        "display_name": sweep.display_name,
        "values": sweep.values,
        "seeds": seeds,
        "results": [],
    }

    for val in sweep.values:
        val_str = f"{val:.6g}"
        val_results = {
            "value": val,
            "value_str": val_str,
            "seed_metrics": {},
            "mean": {},
            "std": {},
            "n_completed": 0,
        }

        all_metrics: Dict[str, List[float]] = {k: [] for k in KEY_METRICS}

        for seed in seeds:
            run_dir = sweep_dir / f"seed_{seed}" / f"val_{val_str}"
            metrics = _load_summary_metrics(run_dir)
            if metrics is not None:
                val_results["n_completed"] += 1
                val_results["seed_metrics"][str(seed)] = {
                    k: metrics.get(k) for k in KEY_METRICS if metrics.get(k) is not None
                }
                for k in KEY_METRICS:
                    v = metrics.get(k)
                    if v is not None and np.isfinite(v):
                        all_metrics[k].append(float(v))

        # 计算 mean ± std
        for k in KEY_METRICS:
            vals = all_metrics[k]
            if vals:
                val_results["mean"][k] = float(np.mean(vals))
                val_results["std"][k] = float(np.std(vals))

        summary["results"].append(val_results)

    return summary


def _print_sweep_table(sweep_summary: Dict):
    """打印一个参数扫描的结果表."""
    name = sweep_summary["display_name"]
    print(f"\n{'=' * 100}")
    print(f"  {name}")
    print(f"{'=' * 100}")

    # 表头
    header_cols = ["Value", "MAP_avg", "s2h_MAP", "h2s_MAP",
                   "herb_PPL_em", "sym_PPL_em", "coherence", "n"]
    header = " | ".join(f"{c:>14s}" for c in header_cols)
    print(header)
    print("-" * len(header))

    for r in sweep_summary["results"]:
        val_str = r["value_str"]
        n = r["n_completed"]
        m = r.get("mean", {})
        s = r.get("std", {})

        def _fmt(key):
            if key in m:
                mean = m[key]
                std = s.get(key, 0.0)
                return f"{mean:.4f}±{std:.4f}"
            return "  N/A"

        row = [
            f"{val_str:>14s}",
            f"{_fmt('valid_map_avg'):>14s}",
            f"{_fmt('valid_map_sym2herb'):>14s}",
            f"{_fmt('valid_map_herb2sym'):>14s}",
            f"{_fmt('herb_pred_ppl_em_prob'):>14s}",
            f"{_fmt('symptom_pred_ppl_em_prob'):>14s}",
            f"{_fmt('topic_coherence'):>14s}",
            f"{n:>14d}",
        ]
        print(" | ".join(row))


# =========================================================================
# 主入口
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="超参数敏感性分析实验 (Revision Experiment 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--base_config", default="config/best_v4.yaml",
        help="基础配置文件 (默认 best_v4.yaml)",
    )
    p.add_argument(
        "--output_root", default="artifacts/revision_sensitivity",
        help="输出根目录 (相对项目根目录)",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 43, 44],
        help="随机种子列表 (默认 3 个种子保证统计稳定性)",
    )
    p.add_argument(
        "--K", type=int, default=30,
        help="主题数 K (默认 30, 与 best_v4 一致)",
    )
    p.add_argument(
        "--params", nargs="*", default=None,
        help="只运行指定参数组, 例如: --params lambda_know gamma_l1",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="跳过已完成的 run",
    )
    p.add_argument(
        "--summary_only", action="store_true",
        help="仅生成汇总表, 不运行新实验",
    )
    return p.parse_args()


def main():
    args = parse_args()

    cfg_path = (PROJECT_ROOT / args.base_config).resolve()
    out_root = (PROJECT_ROOT / args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # 筛选要运行的参数组
    sweeps = DEFAULT_SWEEPS
    if args.params:
        sweeps = [s for s in DEFAULT_SWEEPS if s.name in args.params]
        if not sweeps:
            print(f"❌ 未匹配到参数组: {args.params}")
            print(f"   可用: {[s.name for s in DEFAULT_SWEEPS]}")
            sys.exit(1)

    # 计算总任务量
    total_jobs = sum(len(s.values) * len(args.seeds) for s in sweeps)
    done_jobs = 0
    job_idx = 0

    # 预扫描已完成任务
    if args.resume or args.summary_only:
        for sweep in sweeps:
            sweep_dir = out_root / sweep.name
            for seed in args.seeds:
                for val in sweep.values:
                    val_str = f"{val:.6g}"
                    run_dir = sweep_dir / f"seed_{seed}" / f"val_{val_str}"
                    if _is_completed_run(run_dir):
                        done_jobs += 1
        print("=" * 80)
        print(f"[扫描] 已完成 {done_jobs}/{total_jobs}, 待运行 {total_jobs - done_jobs}")

    # 仅汇总模式
    if args.summary_only:
        all_summaries = {}
        for sweep in sweeps:
            sweep_dir = out_root / sweep.name
            summary = _generate_sweep_summary(sweep, args.seeds, sweep_dir)
            all_summaries[sweep.name] = summary
            _print_sweep_table(summary)
            # 保存每个参数组的汇总
            _write_json_atomic(sweep_dir / "summary_table.json", summary)

        # 保存全局汇总
        _write_json_atomic(out_root / "all_summaries.json", all_summaries)
        print(f"\n✅ 汇总已保存至: {out_root}")
        return

    # 保存实验 manifest
    manifest = {
        "experiment": "hyperparameter_sensitivity",
        "description": "审稿修改实验1: 超参数敏感性分析, 证明正则项在最优权重下增益也有限",
        "base_config": str(cfg_path),
        "K": args.K,
        "seeds": args.seeds,
        "sweeps": {
            s.name: {
                "display_name": s.display_name,
                "values": s.values,
                "param_keys": s.param_keys,
                "loss_switches": s.loss_switches,
            }
            for s in sweeps
        },
        "total_jobs": total_jobs,
    }
    _write_json_atomic(out_root / "manifest.json", manifest)

    # ===== 主循环 =====
    for sweep in sweeps:
        sweep_dir = out_root / sweep.name
        sweep_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#' * 80}")
        print(f"# 参数组: {sweep.display_name}")
        print(f"# 扫描值: {sweep.values}")
        print(f"{'#' * 80}")

        for seed in args.seeds:
            for val in sweep.values:
                job_idx += 1
                val_str = f"{val:.6g}"
                run_dir = sweep_dir / f"seed_{seed}" / f"val_{val_str}"

                # 跳过已完成
                if args.resume and _is_completed_run(run_dir):
                    print(f"[{job_idx}/{total_jobs}] SKIP | {sweep.name}={val_str} seed={seed}")
                    continue

                print("=" * 80)
                print(f"[{job_idx}/{total_jobs}] RUN | {sweep.name}={val_str} | seed={seed} | K={args.K}")
                print(f"  Output: {run_dir}")

                # 构建配置: 基线 + 施加单项正则
                cfg = _build_base_recon_cfg(base_cfg, seed=seed, K=args.K)
                cfg = _apply_sweep(cfg, sweep, val)
                cfg["model_name"] = f"sensitivity_{sweep.name}_{val_str}_seed{seed}"

                t0 = time.time()
                try:
                    _run_one(cfg, run_dir)
                    elapsed = time.time() - t0
                    print(f"  ✅ 完成 ({elapsed:.1f}s)")

                    # 即时打印关键指标
                    metrics = _load_summary_metrics(run_dir)
                    if metrics:
                        print(f"  MAP_avg={metrics.get('valid_map_avg', 'N/A'):.4f}"
                              f"  herb_PPL_em={metrics.get('herb_pred_ppl_em_prob', 'N/A')}"
                              f"  sym_PPL_em={metrics.get('symptom_pred_ppl_em_prob', 'N/A')}")
                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  ❌ 失败 ({elapsed:.1f}s): {e}")
                    # 记录错误但继续
                    error_path = run_dir / "error.txt"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    with open(error_path, "w") as f:
                        f.write(str(e))

        # 每个参数组完成后生成汇总
        summary = _generate_sweep_summary(sweep, args.seeds, sweep_dir)
        _write_json_atomic(sweep_dir / "summary_table.json", summary)
        _print_sweep_table(summary)

    # ===== 全局汇总 =====
    print(f"\n{'=' * 80}")
    print("生成全局汇总...")
    all_summaries = {}
    for sweep in sweeps:
        sweep_dir = out_root / sweep.name
        summary = _generate_sweep_summary(sweep, args.seeds, sweep_dir)
        all_summaries[sweep.name] = summary
    _write_json_atomic(out_root / "all_summaries.json", all_summaries)

    print(f"\n✅ 超参数敏感性分析实验完成")
    print(f"   结果目录: {out_root}")
    print(f"   汇总文件: {out_root / 'all_summaries.json'}")
    print(f"\n   可用以下命令仅查看结果:")
    print(f"   python {Path(__file__).relative_to(PROJECT_ROOT)} --summary_only")


if __name__ == "__main__":
    main()
