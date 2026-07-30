#!/usr/bin/env python3
"""PPL 主指标的多方法、多 seed、多 K 对比实验。

目标
----
- 5 个变体：
  1) recon_only_mvgsnmf          (本项目模型，仅保留 ph+ps 重构)
  2) vanilla_nmf_fro             (Vanilla NMF, Frobenius)
  3) indep_nmf_procrustes_bridge (独立双视图 NMF + Procrustes 对齐)
  4) sparse_nmf_l1               (Sparse NMF, L1/Lasso)
  5) gnmf_graph                  (Graph-regularized NMF)
- seed: 42~46
- K: 5,10,...,40
- 输出目录结构对齐 run_ablation_multiseed.py:
  artifacts/<output_root>/<stage>/seed_<seed>/K_<K>/

说明
----
- 本脚本不做汇总统计；每个 run 目录内都保存完整指标。
- 评估复用 gsnmf.evaluator.evaluate_all（包含 PPL 与其它指标）。
- 可通过 --eval_split 选择在 valid 或 test 上评估。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gsnmf.constraints import build_contraindication_matrix
from gsnmf.data_loader import load_all
from gsnmf.evaluator import evaluate_all
from gsnmf.ppl_refine import reestimate_Hh_em_vectorized, reestimate_Hs_em_vectorized
from gsnmf.schemas import SplitData
from gsnmf.split import split_data
from gsnmf.trainer import Trainer, _tfidf_transform_csr


REQUIRED_FILES = ["summary.json", "factors.npz", "metrics.json", "config.yaml"]


@dataclass
class VariantSpec:
    key: str
    kind: str  # "mvgsnmf" | "nmf" | "nmf_independent" | "nmf_shared_w" | "nmf_shared_w_kl" | "gnmf" | "semi_nmf"
    nmf_params: Optional[Dict] = None
    alpha_ps: Optional[float] = None


class EvalOnlyModel:
    """仅用于复用 evaluator 的轻量模型壳。"""

    def __init__(self, H_h: np.ndarray, H_s: np.ndarray):
        self.H_h = H_h
        self.H_s = H_s
        self.sw = {"ps": True}


def _deep_copy_cfg(cfg: Dict) -> Dict:
    return yaml.safe_load(yaml.safe_dump(cfg, allow_unicode=True))


def _build_recon_only_cfg(base_cfg: Dict, seed: int, k: int) -> Dict:
    cfg = _deep_copy_cfg(base_cfg)
    cfg["model_name"] = "recon_only_mvgsnmf"
    cfg["seed"] = int(seed)
    cfg["K"] = int(k)

    cfg.setdefault("split", {})["seed"] = int(seed)

    # 仅保留双视图重构
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
    sw.update(
        {
            "ph": True,
            "ps": True,
            "pd": False,
            "graph_h": False,
            "graph_s": False,
            "l1": False,
            "contra": False,
            "pair": False,
            "know_hs": False,
            "hyper_h": False,
            "hyper_var": False,
            "hyper_mean": False,
        }
    )

    # 这轮以 PPL 为主指标
    tr = cfg.setdefault("training", {})
    tr["early_stop_metric"] = "symptom_pred_ppl_prob"
    tr["disable_early_stop"] = True

    return cfg


def _stage_name(idx: int, key: str) -> str:
    return f"{idx:02d}_{key}"


def _is_completed_run(run_dir: Path) -> bool:
    return run_dir.exists() and run_dir.is_dir() and all((run_dir / f).exists() for f in REQUIRED_FILES)


def _write_manifest_atomic(path: Path, manifest: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _save_run_artifacts(
    run_dir: Path,
    cfg: Dict,
    factors: Dict[str, np.ndarray],
    metrics: Dict,
    fit_info: Dict,
    eval_split: str,
):
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    np.savez_compressed(run_dir / "factors.npz", **factors)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump([{**fit_info, **metrics}], f, indent=2, ensure_ascii=False, default=float)

    summary = {
        "model_name": cfg.get("model_name", ""),
        "train_seed": cfg.get("seed", None),
        "split_seed": cfg.get("split", {}).get("seed", None),
        "K": cfg.get("K", None),
        "primary_metric": "symptom_pred_ppl_prob",
        "eval_split": eval_split,
        f"{eval_split}_metrics": metrics,
        "fit_info": fit_info,
        "config": cfg,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)


def _select_eval_split(split_data: SplitData, eval_split: str) -> SplitData:
    if eval_split not in ("valid", "test"):
        raise ValueError(f"eval_split must be 'valid' or 'test', got: {eval_split}")
    if eval_split == "valid":
        return split_data
    return SplitData(
        meta=split_data.meta,
        train=split_data.train,
        valid=split_data.test,
        test=split_data.test,
    )


def _dense_bin(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return (np.asarray(X) > 0).astype(np.float64)


def _build_em_ppl_factors(
    H_h: np.ndarray,
    H_s: np.ndarray,
    split_data: SplitData,
    cfg: Dict,
):
    ppl_cfg = cfg.get("ppl_refine", {})
    em_iters = int(ppl_cfg.get("em_iters", 15))
    beta_bar = float(ppl_cfg.get("beta_bar", 0.1))
    alpha = float(cfg.get("inference", {}).get("dirichlet_alpha", 0.01))

    X_ph_train = _dense_bin(split_data.train.X_ph)
    X_ps_train = _dense_bin(split_data.train.X_ps)

    H_s_ppl = reestimate_Hs_em_vectorized(
        H_h=H_h,
        X_ph_train=X_ph_train,
        X_ps_train=X_ps_train,
        n_iters=em_iters,
        beta_bar=beta_bar,
        dirichlet_alpha=alpha,
        H_s_init=H_s,
    )

    H_h_ppl = reestimate_Hh_em_vectorized(
        H_s=H_s,
        X_ps_train=X_ps_train,
        X_ph_train=X_ph_train,
        n_iters=em_iters,
        beta=beta_bar,
        dirichlet_alpha=alpha,
        H_h_init=H_h,
    )
    return H_h_ppl, H_s_ppl


def _apply_tfidf_if_needed(sp_data, cfg: Dict):
    tfidf_cfg = cfg.get("tfidf", {})
    if not tfidf_cfg.get("enabled", False):
        return

    target = tfidf_cfg.get("target", "ph")
    use_idf = bool(tfidf_cfg.get("use_idf", True))
    smooth_idf = bool(tfidf_cfg.get("smooth_idf", True))
    sublinear_tf = bool(tfidf_cfg.get("sublinear_tf", False))
    norm = tfidf_cfg.get("norm", "l2")

    if target in ("ph", "both"):
        sp_data.train.X_ph = _tfidf_transform_csr(sp_data.train.X_ph, use_idf, smooth_idf, sublinear_tf, norm)
        sp_data.valid.X_ph = _tfidf_transform_csr(sp_data.valid.X_ph, use_idf, smooth_idf, sublinear_tf, norm)
        sp_data.test.X_ph = _tfidf_transform_csr(sp_data.test.X_ph, use_idf, smooth_idf, sublinear_tf, norm)

    if target in ("ps", "both"):
        sp_data.train.X_ps = _tfidf_transform_csr(sp_data.train.X_ps, use_idf, smooth_idf, sublinear_tf, norm)
        sp_data.valid.X_ps = _tfidf_transform_csr(sp_data.valid.X_ps, use_idf, smooth_idf, sublinear_tf, norm)
        sp_data.test.X_ps = _tfidf_transform_csr(sp_data.test.X_ps, use_idf, smooth_idf, sublinear_tf, norm)


def _fit_concat_nmf(
    X_ph: np.ndarray,
    X_ps: np.ndarray,
    k: int,
    seed: int,
    nmf_params: Dict,
    alpha_ps: float,
) -> Dict[str, np.ndarray]:
    """在拼接视图 [X_ph | alpha_ps * X_ps] 上做 NMF，并拆回 H_h/H_s。"""
    X_concat = np.hstack([X_ph, alpha_ps * X_ps])

    model = NMF(
        n_components=k,
        init=nmf_params.get("init", "nndsvda"),
        solver=nmf_params.get("solver", "cd"),
        beta_loss=nmf_params.get("beta_loss", "frobenius"),
        alpha_W=float(nmf_params.get("alpha_W", 0.0)),
        alpha_H=float(nmf_params.get("alpha_H", 0.0)),
        l1_ratio=float(nmf_params.get("l1_ratio", 0.0)),
        max_iter=int(nmf_params.get("max_iter", 500)),
        tol=float(nmf_params.get("tol", 1e-4)),
        random_state=int(seed),
        shuffle=bool(nmf_params.get("shuffle", False)),
    )

    W = model.fit_transform(X_concat)  # (P, K)
    H_concat = model.components_       # (K, H+S)

    H = X_ph.shape[1]
    H_h = H_concat[:, :H].T
    H_s_scaled = H_concat[:, H:].T
    H_s = H_s_scaled / max(alpha_ps, 1e-12)

    return {
        "W": W,
        "H_h": H_h,
        "H_s": H_s,
        "D_h": H_h.copy(),
        "reconstruction_err": float(model.reconstruction_err_),
        "n_iter": int(model.n_iter_),
    }


def _fit_independent_nmf(
    X_ph: np.ndarray,
    X_ps: np.ndarray,
    k: int,
    seed: int,
    nmf_params: Dict,
) -> Dict[str, np.ndarray]:
    """分别对 herb/symptom 做两个独立 NMF，再将样本因子做对齐并融合。"""

    common_kwargs = {
        "n_components": k,
        "init": nmf_params.get("init", "nndsvda"),
        "solver": nmf_params.get("solver", "cd"),
        "beta_loss": nmf_params.get("beta_loss", "frobenius"),
        "alpha_W": float(nmf_params.get("alpha_W", 0.0)),
        "alpha_H": float(nmf_params.get("alpha_H", 0.0)),
        "l1_ratio": float(nmf_params.get("l1_ratio", 0.0)),
        "max_iter": int(nmf_params.get("max_iter", 500)),
        "tol": float(nmf_params.get("tol", 1e-4)),
        "shuffle": bool(nmf_params.get("shuffle", False)),
    }

    model_h = NMF(random_state=int(seed), **common_kwargs)
    model_s = NMF(random_state=int(seed) + 997, **common_kwargs)

    W_h = model_h.fit_transform(X_ph)
    H_h = model_h.components_.T

    W_s = model_s.fit_transform(X_ps)
    H_s = model_s.components_.T

    bridge = nmf_params.get("bridge", "none")
    if bridge == "cosine_match":
        C = cosine_similarity(W_h.T, W_s.T)
        match = np.argmax(C, axis=1)
        H_s = H_s[:, match]
        W_s = W_s[:, match]
    elif bridge == "procrustes":
        R, _ = orthogonal_procrustes(W_s, W_h)
        W_s = np.maximum(W_s @ R, 1e-12)
        H_s = np.maximum(H_s @ R, 1e-12)

    merge_mode = nmf_params.get("merge_mode", "avg")
    if merge_mode == "geo":
        W = np.sqrt(np.maximum(W_h, 1e-12) * np.maximum(W_s, 1e-12))
    else:
        W = 0.5 * (W_h + W_s)

    rec_h = np.linalg.norm(X_ph - W_h @ model_h.components_, ord="fro")
    rec_s = np.linalg.norm(X_ps - W_s @ H_s.T, ord="fro")

    return {
        "W": W,
        "W_h": W_h,
        "W_s": W_s,
        "H_h": H_h,
        "H_s": H_s,
        "D_h": H_h.copy(),
        "reconstruction_err": float(rec_h + rec_s),
        "n_iter": int(model_h.n_iter_ + model_s.n_iter_),
    }


def _fit_shared_w_multiview_nmf(
    X_ph: np.ndarray,
    X_ps: np.ndarray,
    k: int,
    seed: int,
    alpha_ps: float,
    max_iter: int = 400,
    tol: float = 1e-4,
) -> Dict[str, np.ndarray]:
    """共享 W 的双视图 NMF（非拼接）：X_ph≈W H_h^T, X_ps≈W H_s^T。"""

    rng = np.random.RandomState(seed)
    P, H = X_ph.shape
    S = X_ps.shape[1]

    W = np.abs(rng.randn(P, k)) * 0.1 + 1e-6
    H_h = np.abs(rng.randn(H, k)) * 0.1 + 1e-6
    H_s = np.abs(rng.randn(S, k)) * 0.1 + 1e-6

    w_ps = max(float(alpha_ps), 1e-8)
    eps = 1e-10
    prev_obj = np.inf
    n_iter_done = 0

    for it in range(max_iter):
        WTW = W.T @ W

        # H_h / H_s updates
        H_h *= (X_ph.T @ W) / np.maximum(H_h @ WTW, eps)
        H_s *= (X_ps.T @ W) / np.maximum(H_s @ WTW, eps)

        H_h = np.maximum(H_h, 1e-12)
        H_s = np.maximum(H_s, 1e-12)

        # W update
        num_w = X_ph @ H_h + w_ps * (X_ps @ H_s)
        den_w = W @ (H_h.T @ H_h + w_ps * (H_s.T @ H_s))
        W *= num_w / np.maximum(den_w, eps)
        W = np.maximum(W, 1e-12)

        # objective for early break
        rec_h = np.linalg.norm(X_ph - W @ H_h.T, ord="fro") ** 2
        rec_s = np.linalg.norm(X_ps - W @ H_s.T, ord="fro") ** 2
        obj = rec_h + w_ps * rec_s
        n_iter_done = it + 1

        if np.isfinite(prev_obj):
            rel = abs(prev_obj - obj) / max(prev_obj, 1e-12)
            if rel < tol:
                break
        prev_obj = obj

    rec = np.linalg.norm(X_ph - W @ H_h.T, ord="fro") + np.linalg.norm(X_ps - W @ H_s.T, ord="fro")

    return {
        "W": W,
        "H_h": H_h,
        "H_s": H_s,
        "D_h": H_h.copy(),
        "reconstruction_err": float(rec),
        "n_iter": int(n_iter_done),
    }


def _fit_shared_w_multiview_kl(
    X_ph: np.ndarray,
    X_ps: np.ndarray,
    k: int,
    seed: int,
    alpha_ps: float,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> Dict[str, np.ndarray]:
    """共享 W 的双视图 KL-NMF：KL(X_ph||W H_h^T)+w*KL(X_ps||W H_s^T)。"""

    rng = np.random.RandomState(seed)
    P, H = X_ph.shape
    S = X_ps.shape[1]

    Xh = np.maximum(X_ph, 1e-12)
    Xs = np.maximum(X_ps, 1e-12)

    W = np.abs(rng.randn(P, k)) * 0.1 + 1e-6
    H_h = np.abs(rng.randn(H, k)) * 0.1 + 1e-6
    H_s = np.abs(rng.randn(S, k)) * 0.1 + 1e-6

    w_ps = max(float(alpha_ps), 1e-8)
    eps = 1e-10
    prev_obj = np.inf
    n_iter_done = 0

    one_h = np.ones_like(Xh)
    one_s = np.ones_like(Xs)

    for it in range(max_iter):
        WHh = np.maximum(W @ H_h.T, eps)
        WHs = np.maximum(W @ H_s.T, eps)

        # KL MU updates for H_h / H_s
        H_h *= ((Xh / WHh).T @ W) / np.maximum(one_h.T @ W, eps)
        H_s *= ((Xs / WHs).T @ W) / np.maximum(one_s.T @ W, eps)
        H_h = np.maximum(H_h, 1e-12)
        H_s = np.maximum(H_s, 1e-12)

        WHh = np.maximum(W @ H_h.T, eps)
        WHs = np.maximum(W @ H_s.T, eps)

        # KL MU update for W (weighted two-view)
        num_w = (Xh / WHh) @ H_h + w_ps * ((Xs / WHs) @ H_s)
        den_w = one_h @ H_h + w_ps * (one_s @ H_s)
        W *= num_w / np.maximum(den_w, eps)
        W = np.maximum(W, 1e-12)

        WHh = np.maximum(W @ H_h.T, eps)
        WHs = np.maximum(W @ H_s.T, eps)

        kl_h = np.sum(Xh * np.log(Xh / WHh) - Xh + WHh)
        kl_s = np.sum(Xs * np.log(Xs / WHs) - Xs + WHs)
        obj = kl_h + w_ps * kl_s
        n_iter_done = it + 1

        if np.isfinite(prev_obj):
            rel = abs(prev_obj - obj) / max(abs(prev_obj), 1e-12)
            if rel < tol:
                break
        prev_obj = obj

    rec = np.linalg.norm(X_ph - W @ H_h.T, ord="fro") + np.linalg.norm(X_ps - W @ H_s.T, ord="fro")

    return {
        "W": W,
        "H_h": H_h,
        "H_s": H_s,
        "D_h": H_h.copy(),
        "reconstruction_err": float(rec),
        "n_iter": int(n_iter_done),
    }


def _build_knn_graph(X: np.ndarray, k: int = 10) -> np.ndarray:
    """基于样本余弦相似构建 KNN 图邻接 (P, P)。"""
    # 归一化
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Xn = X / norms
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, 0.0)

    P = sim.shape[0]
    A = np.zeros_like(sim)
    for i in range(P):
        idx = np.argsort(sim[i])[-k:]
        A[i, idx] = sim[i, idx]

    A = 0.5 * (A + A.T)
    return A


def _fit_concat_gnmf(
    X_ph: np.ndarray,
    X_ps: np.ndarray,
    k: int,
    seed: int,
    alpha_ps: float,
    lambda_g: float = 1e-2,
    knn_k: int = 10,
    max_iter: int = 300,
) -> Dict[str, np.ndarray]:
    """简化 GNMF: min ||X-WH||^2 + lambda * tr(W^T L W) 的 MU 更新。"""
    rng = np.random.RandomState(seed)
    X = np.hstack([X_ph, alpha_ps * X_ps]).astype(np.float64)
    P, D = X.shape

    W = np.abs(rng.randn(P, k)) * 0.1 + 1e-6
    Hm = np.abs(rng.randn(k, D)) * 0.1 + 1e-6

    A = _build_knn_graph(X, k=knn_k)
    d = A.sum(axis=1)
    Dg = np.diag(d)
    L = Dg - A

    eps = 1e-10
    for _ in range(max_iter):
        # H update
        num_h = W.T @ X
        den_h = (W.T @ W @ Hm) + eps
        Hm *= num_h / den_h

        # W update (带图正则)
        num_w = X @ Hm.T + lambda_g * (A @ W)
        den_w = W @ (Hm @ Hm.T) + lambda_g * (Dg @ W) + eps
        W *= num_w / den_w

        W = np.maximum(W, 1e-12)
        Hm = np.maximum(Hm, 1e-12)

    H = X_ph.shape[1]
    H_h = Hm[:, :H].T
    H_s = (Hm[:, H:].T) / max(alpha_ps, 1e-12)

    rec = np.linalg.norm(X - W @ Hm, ord="fro")

    return {
        "W": W,
        "H_h": H_h,
        "H_s": H_s,
        "D_h": H_h.copy(),
        "reconstruction_err": float(rec),
        "n_iter": int(max_iter),
    }


def _fit_concat_semi_nmf(
    X_ph: np.ndarray,
    X_ps: np.ndarray,
    k: int,
    seed: int,
    alpha_ps: float,
    max_iter: int = 300,
) -> Dict[str, np.ndarray]:
    """简化 Semi-NMF: X ≈ W H, 其中 H >=0, W 可有符号。"""
    rng = np.random.RandomState(seed)
    X = np.hstack([X_ph, alpha_ps * X_ps]).astype(np.float64)
    P, D = X.shape

    W = rng.randn(P, k) * 0.1
    Hm = np.abs(rng.randn(k, D)) * 0.1 + 1e-6

    eps = 1e-10
    for _ in range(max_iter):
        # W closed-form least squares
        HHT = Hm @ Hm.T
        W = X @ Hm.T @ np.linalg.pinv(HHT + 1e-8 * np.eye(k))

        # H multiplicative with split positive/negative parts
        WT = W.T
        WTX = WT @ X
        WTW = WT @ W

        WTX_pos = np.maximum(WTX, 0)
        WTX_neg = np.maximum(-WTX, 0)
        WTW_pos = np.maximum(WTW, 0)
        WTW_neg = np.maximum(-WTW, 0)

        num = WTX_pos + WTW_neg @ Hm
        den = WTX_neg + WTW_pos @ Hm + eps
        Hm *= np.sqrt(num / den)
        Hm = np.maximum(Hm, 1e-12)

    H = X_ph.shape[1]
    H_h = Hm[:, :H].T
    H_s = (Hm[:, H:].T) / max(alpha_ps, 1e-12)

    rec = np.linalg.norm(X - W @ Hm, ord="fro")

    return {
        "W": np.maximum(W, 0.0),  # evaluator 不用 W，这里仅为了保存非负版本
        "W_raw": W,
        "H_h": H_h,
        "H_s": H_s,
        "D_h": H_h.copy(),
        "reconstruction_err": float(rec),
        "n_iter": int(max_iter),
    }


def _run_mvgsnmf_recon_only(cfg: Dict, run_dir: Path, eval_split: str):
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

    # 若要求 test 评估，重算并覆盖 summary（避免 evaluator 内部默认 valid 口径）
    if eval_split == "test":
        eval_data = _select_eval_split(trainer.split_data, eval_split)
        metrics = evaluate_all(
            trainer.model,
            eval_data,
            trainer.C_hh,
            compute_perplexity=True,
            eval_seed=2025,
            compute_dose=False,
            dirichlet_alpha=trainer.dirichlet_alpha,
            tfidf_decouple=trainer.tfidf_decouple,
            hypergraph_bundle=getattr(trainer, "_hyper_bundle", None),
            H_s_ppl=getattr(trainer, "_H_s_ppl", None),
            H_h_ppl=getattr(trainer, "_H_h_ppl", None),
        )
        summary_path = Path(trainer.run_dir) / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            summary["eval_split"] = "test"
            summary.pop("test_metrics", None)
            summary["test_metrics"] = metrics
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    src = Path(trainer.run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.move(str(src), str(run_dir))


def _run_baseline_variant(base_cfg: Dict, variant: VariantSpec, seed: int, k: int, run_dir: Path, eval_split: str):
    cfg = _deep_copy_cfg(base_cfg)
    cfg["model_name"] = variant.key
    cfg["seed"] = int(seed)
    cfg["K"] = int(k)
    cfg.setdefault("split", {})["seed"] = int(seed)

    data = load_all(
        data_root=cfg.get("data_root"),
        file_overrides=cfg.get("files"),
        load_dosage=bool(
            cfg.get("loss_switches", {}).get("pd", False)
        ),
    )
    sp_data = split_data(data, cfg.get("split", {}))
    _apply_tfidf_if_needed(sp_data, cfg)

    X_ph_train = sp_data.train.X_ph.toarray().astype(np.float64)
    X_ps_train = sp_data.train.X_ps.toarray().astype(np.float64)
    alpha_ps = float(variant.alpha_ps) if variant.alpha_ps is not None else float(cfg.get("alpha", 1.0))

    if variant.kind == "nmf":
        fit = _fit_concat_nmf(
            X_ph=X_ph_train,
            X_ps=X_ps_train,
            k=k,
            seed=seed,
            nmf_params=variant.nmf_params or {},
            alpha_ps=alpha_ps,
        )
    elif variant.kind == "nmf_independent":
        fit = _fit_independent_nmf(
            X_ph=X_ph_train,
            X_ps=X_ps_train,
            k=k,
            seed=seed,
            nmf_params=variant.nmf_params or {},
        )
    elif variant.kind == "nmf_shared_w":
        p = variant.nmf_params or {}
        fit = _fit_shared_w_multiview_nmf(
            X_ph=X_ph_train,
            X_ps=X_ps_train,
            k=k,
            seed=seed,
            alpha_ps=float(p.get("alpha_ps", alpha_ps)),
            max_iter=int(p.get("max_iter", 400)),
            tol=float(p.get("tol", 1e-4)),
        )
    elif variant.kind == "nmf_shared_w_kl":
        p = variant.nmf_params or {}
        fit = _fit_shared_w_multiview_kl(
            X_ph=X_ph_train,
            X_ps=X_ps_train,
            k=k,
            seed=seed,
            alpha_ps=float(p.get("alpha_ps", alpha_ps)),
            max_iter=int(p.get("max_iter", 500)),
            tol=float(p.get("tol", 1e-4)),
        )
    elif variant.kind == "gnmf":
        p = variant.nmf_params or {}
        fit = _fit_concat_gnmf(
            X_ph=X_ph_train,
            X_ps=X_ps_train,
            k=k,
            seed=seed,
            alpha_ps=alpha_ps,
            lambda_g=float(p.get("lambda_g", 1e-2)),
            knn_k=int(p.get("knn_k", 10)),
            max_iter=int(p.get("max_iter", 300)),
        )
    elif variant.kind == "semi_nmf":
        p = variant.nmf_params or {}
        fit = _fit_concat_semi_nmf(
            X_ph=X_ph_train,
            X_ps=X_ps_train,
            k=k,
            seed=seed,
            alpha_ps=alpha_ps,
            max_iter=int(p.get("max_iter", 300)),
        )
    else:
        raise ValueError(f"Unsupported variant kind: {variant.kind}")

    H_h = fit["H_h"]
    H_s = fit["H_s"]
    H_h_ppl, H_s_ppl = _build_em_ppl_factors(H_h=H_h, H_s=H_s, split_data=sp_data, cfg=cfg)

    model = EvalOnlyModel(H_h=H_h, H_s=H_s)

    C_hh = build_contraindication_matrix(data.herb_mutex, H_h.shape[0])
    eval_data = _select_eval_split(sp_data, eval_split)
    metrics = evaluate_all(
        model=model,
        split_data=eval_data,
        C_hh=C_hh,
        compute_perplexity=True,
        eval_seed=2025,
        compute_dose=False,
        dirichlet_alpha=float(cfg.get("inference", {}).get("dirichlet_alpha", 0.0)),
        tfidf_decouple=bool(cfg.get("inference", {}).get("tfidf_decouple", False)),
        hypergraph_bundle=None,
        H_s_ppl=H_s_ppl,
        H_h_ppl=H_h_ppl,
    )

    factors = {
        "G_p": fit["W"],
        "H_h": H_h,
        "H_s": H_s,
        "D_h": fit["D_h"],
        "H_h_ppl": H_h_ppl,
        "H_s_ppl": H_s_ppl,
    }
    if "W_raw" in fit:
        factors["G_p_raw"] = fit["W_raw"]
    if "W_h" in fit:
        factors["G_p_h"] = fit["W_h"]
    if "W_s" in fit:
        factors["G_p_s"] = fit["W_s"]

    fit_info = {
        "variant": variant.key,
        "reconstruction_err": fit["reconstruction_err"],
        "n_iter": fit["n_iter"],
    }
    _save_run_artifacts(run_dir, cfg, factors, metrics, fit_info, eval_split=eval_split)


def parse_args():
    p = argparse.ArgumentParser(description="PPL主指标：多变体多seed多K实验")
    p.add_argument("--base_config", default="config/best_v4.yaml")
    p.add_argument("--output_root", default="artifacts/ppl_multiseed_compare")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 15, 20, 25, 30, 35, 40])
    p.add_argument("--eval_split", choices=["valid", "test"], default="test", help="评估数据切片")
    p.add_argument("--resume", action="store_true", help="跳过已完成 run")
    p.add_argument("--auto_resume", action="store_true", help="自动从 output_root 下已有结果恢复（含断电恢复）")
    p.add_argument("--overwrite_partial", action="store_true", help="若检测到不完整 run，先删除后重跑")
    return p.parse_args()


def main():
    args = parse_args()

    cfg_path = (PROJECT_ROOT / args.base_config).resolve()
    out_root = (PROJECT_ROOT / args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    variants: List[VariantSpec] = [
        VariantSpec(key="recon_only_mvgsnmf", kind="mvgsnmf"),
        VariantSpec(
            key="vanilla_nmf_fro",
            kind="nmf",
            nmf_params={"solver": "cd", "beta_loss": "frobenius", "max_iter": 500, "tol": 1e-4, "shuffle": True},
        ),
        VariantSpec(
            key="indep_nmf_procrustes_bridge",
            kind="nmf_independent",
            nmf_params={
                "solver": "cd",
                "beta_loss": "frobenius",
                "max_iter": 500,
                "tol": 1e-4,
                "shuffle": True,
                "bridge": "procrustes",
                "merge_mode": "geo",
            },
        ),
        VariantSpec(
            key="sparse_nmf_l1",
            kind="nmf",
            nmf_params={
                "solver": "cd",
                "beta_loss": "frobenius",
                "alpha_W": 1e-3,
                "alpha_H": 1e-3,
                "l1_ratio": 0.5,
                "max_iter": 600,
                "tol": 1e-4,
                "shuffle": True,
            },
        ),
        VariantSpec(
            key="gnmf_graph",
            kind="gnmf",
            nmf_params={"lambda_g": 1e-2, "knn_k": 10, "max_iter": 300},
        ),
    ]

    total_jobs = len(variants) * len(args.seeds) * len(args.k_values)
    done_jobs = 0

    use_resume = bool(args.resume or args.auto_resume)
    manifest_path = out_root / "manifest.json"

    if use_resume:
        for i, v in enumerate(variants):
            stage = out_root / _stage_name(i, v.key)
            for sd in args.seeds:
                for k in args.k_values:
                    run_dir = stage / f"seed_{sd}" / f"K_{k}"
                    if _is_completed_run(run_dir):
                        done_jobs += 1
                    elif args.overwrite_partial and run_dir.exists():
                        shutil.rmtree(run_dir, ignore_errors=True)
        print("=" * 80)
        print(f"[Resume scan] completed {done_jobs}/{total_jobs}, pending {total_jobs - done_jobs}")

    if use_resume and manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}
    else:
        manifest = {}

    manifest.update(
        {
            "base_config": str(cfg_path),
            "output_root": str(out_root),
            "seeds": args.seeds,
            "k_values": args.k_values,
            "eval_split": args.eval_split,
            "primary_metric": "symptom_pred_ppl_prob",
            "variants": [v.key for v in variants],
            "total_jobs": total_jobs,
            "completed_jobs_scan": done_jobs,
            "runs": manifest.get("runs", []),
        }
    )
    _write_manifest_atomic(manifest_path, manifest)

    job = 0
    for i, v in enumerate(variants):
        stage_name = _stage_name(i, v.key)
        stage_dir = out_root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        for sd in args.seeds:
            for k in args.k_values:
                job += 1
                run_dir = stage_dir / f"seed_{sd}" / f"K_{k}"

                if use_resume and _is_completed_run(run_dir):
                    print("-" * 80)
                    print(f"[{job}/{total_jobs}] SKIP completed | variant={v.key} seed={sd} K={k}")
                    manifest["runs"].append({
                        "variant": v.key,
                        "seed": sd,
                        "K": k,
                        "dir": str(run_dir),
                        "status": "skipped_completed",
                    })
                    _write_manifest_atomic(manifest_path, manifest)
                    continue

                print("=" * 80)
                print(f"[{job}/{total_jobs}] RUN | variant={v.key} | seed={sd} | K={k}")
                print(f"Output: {run_dir}")

                run_rec = {
                    "variant": v.key,
                    "seed": sd,
                    "K": k,
                    "dir": str(run_dir),
                    "status": "running",
                }
                manifest["runs"].append(run_rec)
                _write_manifest_atomic(manifest_path, manifest)

                try:
                    if v.kind == "mvgsnmf":
                        cfg = _build_recon_only_cfg(base_cfg, seed=sd, k=k)
                        _run_mvgsnmf_recon_only(cfg, run_dir, eval_split=args.eval_split)
                    else:
                        _run_baseline_variant(base_cfg, v, seed=sd, k=k, run_dir=run_dir, eval_split=args.eval_split)
                    run_rec["status"] = "done"
                except Exception as e:
                    run_rec["status"] = "failed"
                    run_rec["error"] = str(e)
                    _write_manifest_atomic(manifest_path, manifest)
                    raise

                _write_manifest_atomic(manifest_path, manifest)

    manifest["final_done_jobs"] = sum(1 for r in manifest.get("runs", []) if r.get("status") in ("done", "skipped_completed"))
    _write_manifest_atomic(manifest_path, manifest)

    print("\n✅ 全部实验完成")
    print(f"Results root: {out_root}")
    print(f"Manifest: {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
