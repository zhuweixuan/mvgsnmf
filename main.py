"""Train or smoke-test the MV-GSNMF model."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml

from gsnmf.evaluator import evaluate_all
from gsnmf.trainer import Trainer


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MV-GSNMF training entry point")
    parser.add_argument(
        "--config",
        default="config/paper_full.yaml",
        help="YAML configuration path (default: config/paper_full.yaml)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=None,
        help="Override the device in the YAML file",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override the data directory in the YAML file",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Use only this many prescriptions",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Override training.max_iter",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short end-to-end check instead of the full experiment",
    )
    return parser.parse_args()


def load_config(path_value: str) -> dict:
    cfg_path = Path(path_value)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with cfg_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    data_root = Path(cfg.get("data_root", "data"))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    cfg["data_root"] = str(data_root.resolve())
    return cfg


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if args.data_root:
        cfg["data_root"] = str(Path(args.data_root).resolve())
    if args.max_iter is not None:
        cfg.setdefault("training", {})["max_iter"] = args.max_iter

    subsample = args.subsample
    if args.smoke:
        subsample = subsample or 128
        training = cfg.setdefault("training", {})
        training.update(
            {
                "max_iter": 2,
                "pretrain_iters": 1,
                "refine_hs_iters": 0,
                "average_last_n_checkpoints": 1,
                "log_every": 1,
                "eval_every": 2,
            }
        )
        cfg.setdefault("ppl_refine", {})["em_iters"] = 0
        cfg["model_name"] = "mvgsnmf_smoke"

    trainer = Trainer(cfg)
    trainer.setup(subsample=subsample)

    def eval_fn(model, split_data, c_hh):
        return evaluate_all(
            model,
            split_data,
            c_hh,
            compute_perplexity=True,
            eval_seed=2025,
            compute_dose=False,
            dirichlet_alpha=trainer.dirichlet_alpha,
            tfidf_decouple=trainer.tfidf_decouple,
            hypergraph_bundle=getattr(trainer, "_hyper_bundle", None),
            H_s_ppl=getattr(trainer, "_H_s_ppl", None),
            H_h_ppl=getattr(trainer, "_H_h_ppl", None),
        )

    trainer.train(eval_fn=None if args.smoke else eval_fn)
    mode = "Smoke test" if args.smoke else "Training"
    print(f"{mode} completed. Outputs: {trainer.run_dir}")


if __name__ == "__main__":
    main()
