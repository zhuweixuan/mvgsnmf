#!/usr/bin/env python3
"""多 seed 消融实验脚本。

目标：
1) 从“全量损失”开始训练；
2) 按既定顺序逐步删除损失项；
3) 每个设置跑 5 个 seed；
4) 结果按: 大文件夹/设置名/seed_xxx/K_xx/ 保存。

示例：
python scripts/run_ablation_multiseed.py \
  --base_config config/best_v4.yaml \
  --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

# 允许从 scripts/ 直接运行时导入项目包 gsnmf
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gsnmf.evaluator import evaluate_all
from gsnmf.trainer import Trainer


REQUIRED_FILES = ["summary.json", "factors.npz", "metrics.json", "config.yaml"]


@dataclass
class AblationStage:
    name: str
    remove_losses: List[str]


def _deep_copy_cfg(cfg: Dict) -> Dict:
    # yaml roundtrip 方式做深拷贝，避免手动 copy 深层 dict
    return yaml.safe_load(yaml.safe_dump(cfg, allow_unicode=True))


def build_full_loss_cfg(cfg: Dict) -> Dict:
    """把配置重置到“全量损失”起点（按用户给定权重）。"""
    c = _deep_copy_cfg(cfg)

    # 全量权重
    c["lambda_h"] = 1.0
    c["lambda_s"] = 1.0
    c["beta_pair"] = 0.05
    c["lambda_know"] = 0.2

    # L1 稀疏 (γ=0.1；gamma_d 按项目习惯设为 0.01)
    c["gamma_g"] = 0.1
    c["gamma_h"] = 0.1
    c["gamma_s"] = 0.1
    c["gamma_d"] = 0.01

    # Loss 开关
    sw = c.setdefault("loss_switches", {})
    sw.update(
        {
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
    )

    # 预训练轮数由 CLI 控制，默认由 main() 注入
    train_cfg = c.setdefault("training", {})
    train_cfg.setdefault("pretrain_iters", 0)

    return c


def apply_remove_losses(cfg: Dict, remove_losses: List[str]) -> Dict:
    """在 cfg 上关闭指定损失，并把对应权重置零。"""
    c = _deep_copy_cfg(cfg)
    sw = c.setdefault("loss_switches", {})

    for loss_name in remove_losses:
        if loss_name == "graph_h":
            sw["graph_h"] = False
            c["lambda_h"] = 0.0
        elif loss_name == "graph_s":
            sw["graph_s"] = False
            c["lambda_s"] = 0.0
        elif loss_name == "l1":
            sw["l1"] = False
            c["gamma_g"] = 0.0
            c["gamma_h"] = 0.0
            c["gamma_s"] = 0.0
            c["gamma_d"] = 0.0
        elif loss_name == "pair":
            sw["pair"] = False
            c["beta_pair"] = 0.0
        elif loss_name == "know_hs":
            sw["know_hs"] = False
            c["lambda_know"] = 0.0
        elif loss_name == "contra":
            sw["contra"] = False
            c["rho"] = 0.0
        elif loss_name == "hyper_h":
            sw["hyper_h"] = False
            c["lambda_hyper"] = 0.0
        elif loss_name in ("hyper_var", "hyper_mean"):
            sw[loss_name] = False
            c["lambda_hyper"] = 0.0
        elif loss_name == "pd":
            sw["pd"] = False
            c["beta"] = 0.0
        else:
            raise ValueError(f"Unknown loss name: {loss_name}")

    return c


def current_active_losses(cfg: Dict) -> List[str]:
    """返回当前激活的损失项名（除重构 ph/ps 外）。"""
    sw = cfg.get("loss_switches", {})
    active = []
    for key in ["graph_h", "graph_s", "l1", "pair", "know_hs"]:
        if sw.get(key, False):
            active.append(key)
    return active


def stage_folder_name(cfg: Dict) -> str:
    active = current_active_losses(cfg)
    if not active:
        return "recon_only_ph_ps_nonneg"
    return "active_" + "__".join(active)


def is_completed_run(run_dir: Path) -> bool:
    if not run_dir.exists() or not run_dir.is_dir():
        return False
    return all((run_dir / name).exists() for name in REQUIRED_FILES)


def run_one(cfg: Dict, out_seed_dir: Path):
    trainer = Trainer(cfg)
    trainer.setup()

    def eval_fn(model, split_data, C_hh):
        return evaluate_all(
            model,
            split_data,
            C_hh,
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

    src_run_dir = Path(trainer.run_dir)
    out_seed_dir.parent.mkdir(parents=True, exist_ok=True)

    if out_seed_dir.exists():
        shutil.rmtree(out_seed_dir)
    shutil.move(str(src_run_dir), str(out_seed_dir))


def parse_args():
    p = argparse.ArgumentParser(description="Ablation multi-seed runner")
    p.add_argument("--base_config", default="config/best_v4.yaml", help="基础配置文件")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="seed 列表")
    p.add_argument(
        "--output_root",
        default="artifacts/ablation_multiseed",
        help="消融总输出目录（相对项目根目录）",
    )
    p.add_argument(
        "--drop_order",
        nargs="+",
        default=["know_hs", "pair", "l1", "graph_s", "graph_h"],
        help="逐步移除顺序",
    )
    p.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20, 25, 30, 35, 40],
        help="每个 seed 内扫描的 K 值",
    )
    p.add_argument(
        "--pretrain_iters",
        type=int,
        default=100,
        help="预训练轮数 (默认 100)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="扫描输出目录并跳过已完成的 runs",
    )
    return p.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg_path = (project_root / args.base_config).resolve()
    out_root = (project_root / args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    full_cfg = build_full_loss_cfg(base_cfg)

    # 构建阶段: full -> 逐步删 loss，直到仅重构
    stage_cfgs: List[Dict] = []
    removed: List[str] = []

    cfg0 = _deep_copy_cfg(full_cfg)
    cfg0["model_name"] = "ablation_full_losses"
    stage_cfgs.append(cfg0)

    for loss_name in args.drop_order:
        removed.append(loss_name)
        c = apply_remove_losses(full_cfg, removed)
        c["model_name"] = f"ablation_drop_{'__'.join(removed)}"
        stage_cfgs.append(c)

    manifest = {
        "base_config": str(cfg_path),
        "seeds": args.seeds,
        "k_values": args.k_values,
        "pretrain_iters": args.pretrain_iters,
        "drop_order": args.drop_order,
        "stages": [],
    }

    # 预扫描总任务量与已完成数量
    total_jobs = len(stage_cfgs) * len(args.seeds) * len(args.k_values)
    done_jobs = 0
    if args.resume:
        for stage_idx, cfg in enumerate(stage_cfgs):
            stage_name = f"{stage_idx:02d}_{stage_folder_name(cfg)}"
            stage_dir = out_root / stage_name
            for seed in args.seeds:
                for k in args.k_values:
                    run_dir = stage_dir / f"seed_{seed}" / f"K_{k}"
                    if is_completed_run(run_dir):
                        done_jobs += 1
        print("=" * 80)
        print(f"[Resume scan] completed {done_jobs}/{total_jobs}, pending {total_jobs - done_jobs}")

    job_idx = 0
    for stage_idx, cfg in enumerate(stage_cfgs):
        stage_name = f"{stage_idx:02d}_{stage_folder_name(cfg)}"
        stage_dir = out_root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        active_losses = current_active_losses(cfg)
        stage_info = {
            "stage_dir": str(stage_dir),
            "active_losses": active_losses,
            "model_name": cfg.get("model_name", ""),
            "seeds": [],
        }

        for seed in args.seeds:
            seed_info = {"seed": seed, "k_runs": []}
            for k in args.k_values:
                job_idx += 1
                run_dir = stage_dir / f"seed_{seed}" / f"K_{k}"

                if args.resume and is_completed_run(run_dir):
                    print("-" * 80)
                    print(f"[{job_idx}/{total_jobs}] SKIP completed | stage={stage_name} seed={seed} K={k}")
                    seed_info["k_runs"].append({"K": k, "dir": str(run_dir), "status": "skipped_completed"})
                    continue

                seed_cfg = _deep_copy_cfg(cfg)
                seed_cfg["seed"] = int(seed)
                seed_cfg.setdefault("split", {})["seed"] = int(seed)
                seed_cfg["K"] = int(k)
                seed_cfg.setdefault("training", {})["pretrain_iters"] = int(args.pretrain_iters)

                seed_cfg["model_name"] = f"{cfg.get('model_name', 'ablation')}_seed{seed}_K{k}"

                print("=" * 80)
                print(f"[{job_idx}/{total_jobs}] RUN | stage={stage_name} | seed={seed} | K={k}")
                print(f"Active losses: {active_losses if active_losses else ['ph','ps(only reconstruction)']}")
                print(f"Output: {run_dir}")
                run_one(seed_cfg, run_dir)

                seed_info["k_runs"].append({"K": k, "dir": str(run_dir), "status": "done"})

            stage_info["seeds"].append(seed_info)

        manifest["stages"].append(stage_info)

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n✅ Ablation finished.")
    print(f"Results root: {out_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
