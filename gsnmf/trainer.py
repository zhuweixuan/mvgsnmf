"""
训练循环、收敛判断、early stopping、run 管理。
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import yaml
from scipy import sparse as sp
from tqdm import trange

from gsnmf.backend import get_backend, to_device, to_numpy
from gsnmf.constraints import build_contraindication_matrix
from gsnmf.data_loader import load_all
from gsnmf.graph_builder import build_herb_graph, build_symptom_graph
from gsnmf.hypergraph_builder import build_herb_hypergraph
from gsnmf.model import MVGSNMTF
from gsnmf.ppl_refine import reestimate_Hh_em_vectorized, reestimate_Hs_em_vectorized
from gsnmf.schemas import (AllData, LossComponents, ModelFactors,
                            SplitData, SplitSlice)
from gsnmf.split import split_data


def _tfidf_transform_csr(
    X: sp.csr_matrix,
    use_idf: bool = True,
    smooth_idf: bool = True,
    sublinear_tf: bool = False,
    norm: str = "l2",
) -> sp.csr_matrix:
    """对稀疏矩阵做 TF-IDF 前处理（不改变 shape）。"""
    X = X.tocsr().astype(np.float64, copy=True)
    P, V = X.shape

    if sublinear_tf and X.nnz > 0:
        X.data = np.log1p(X.data)

    if use_idf:
        # df: 每列非零文档数
        df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float64)
        if smooth_idf:
            idf = np.log((1.0 + P) / (1.0 + df)) + 1.0
        else:
            df = np.maximum(df, 1.0)
            idf = np.log(P / df) + 1.0
        X = X @ sp.diags(idf, offsets=0, shape=(V, V), format="csr")

    if norm in ("l1", "l2"):
        row_sums = np.asarray(X.sum(axis=1)).ravel()
        if norm == "l2":
            row_sums = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        row_sums = np.maximum(row_sums, 1e-12)
        inv = 1.0 / row_sums
        X = sp.diags(inv, offsets=0, shape=(P, P), format="csr") @ X

    return X

logger = logging.getLogger(__name__)


# =========================================================================
# 配置加载
# =========================================================================

def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict:
    """加载 YAML 配置，可选 CLI 覆盖。"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def _deep_update(base: dict, override: dict):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


# =========================================================================
# 稠密化工具
# =========================================================================

def _to_dense(X) -> np.ndarray:
    """将稀疏矩阵转为稠密 ndarray。"""
    if sp.issparse(X):
        return X.toarray().astype(np.float64)
    return np.asarray(X, dtype=np.float64)


# =========================================================================
# Run 管理
# =========================================================================

def _init_run_dir(artifacts_dir: str, cfg: Dict) -> str:
    """创建 artifacts/<run_id>/ 目录并保存配置。"""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(artifacts_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    logger.info("Run 目录: %s", run_dir)
    return run_dir


# =========================================================================
# Trainer
# =========================================================================

class Trainer:
    """MV-GSNMTF 训练器。"""

    # 进程级缓存：避免消融实验中重复从磁盘加载同一份数据
    _cached_data_key: Optional[str] = None
    _cached_data: Optional[AllData] = None

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.model: Optional[MVGSNMTF] = None
        self.split_data: Optional[SplitData] = None
        self.L_h: Optional[sp.csr_matrix] = None
        self.L_s: Optional[sp.csr_matrix] = None
        self.L_hyper_h: Optional[sp.csr_matrix] = None
        self.C_hh: Optional[sp.csr_matrix] = None
        self.run_dir: Optional[str] = None
        self.history: list[Dict[str, float]] = []
        # 推断阶段 Dirichlet 先验平滑 (从 config 读取)
        self.dirichlet_alpha: float = float(
            cfg.get("inference", {}).get("dirichlet_alpha", 0.0))
        # TF-IDF 概率解耦 (从 config 读取)
        self.tfidf_decouple: bool = bool(
            cfg.get("inference", {}).get("tfidf_decouple", False))

    # -----------------------------------------------------------------
    # 准备
    # -----------------------------------------------------------------

    def setup(self, subsample: Optional[int] = None):
        """加载数据、切分、构图、初始化模型。"""
        cfg = self.cfg
        seed = cfg.get("seed", 42)
        np.random.seed(seed)

        # 1. 加载数据（带缓存）
        logger.info("=== 加载数据 ===")
        load_dosage = bool(
            cfg.get("loss_switches", {}).get("pd", False)
        )
        cache_key = json.dumps({
            "data_root": cfg.get("data_root"),
            "files": cfg.get("files"),
            "load_dosage": load_dosage,
        }, sort_keys=True, ensure_ascii=False)
        if Trainer._cached_data_key == cache_key and Trainer._cached_data is not None:
            data = Trainer._cached_data
            logger.info("命中数据缓存，跳过重复加载")
        else:
            data = load_all(
                data_root=cfg.get("data_root"),
                file_overrides=cfg.get("files"),
                pair_min_support=cfg.get("graph", {}).get("pair_min_support", 20),
                load_dosage=load_dosage,
            )
            Trainer._cached_data_key = cache_key
            Trainer._cached_data = data

        # 子采样（smoke test 用）
        if subsample and subsample < data.X_ph.shape[0]:
            rng = np.random.RandomState(seed)
            idx = rng.choice(data.X_ph.shape[0], subsample, replace=False)
            data = AllData(
                X_ph=data.X_ph[idx], X_ps=data.X_ps[idx],
                X_pd=data.X_pd[idx], M_pd=data.M_pd[idx],
                F_h=data.F_h, F_s=data.F_s,
                herb_cooc=data.herb_cooc, symptom_cooc=data.symptom_cooc,
                K_hs=data.K_hs, pair_index=data.pair_index, X_pair=data.X_pair[idx],
                herb_mutex=data.herb_mutex,
                herb_ids=data.herb_ids, herb_names=data.herb_names,
                symptom_ids=data.symptom_ids, symptom_names=data.symptom_names,
                prescription_ids=data.prescription_ids[idx],
            )
            logger.info("子采样 %d 条处方", subsample)

        # 2. 切分
        logger.info("=== 切分数据 ===")
        split_cfg = cfg.get("split", {})
        split_cfg["seed"] = seed
        self.split_data = split_data(data, split_cfg)

        # 2b. TF-IDF 前处理（在初始矩阵输入模型前）
        tfidf_cfg = cfg.get("tfidf", {})
        if tfidf_cfg.get("enabled", False):
            logger.info("=== 应用 TF-IDF 前处理 ===")
            target = tfidf_cfg.get("target", "ph")  # ph | ps | both
            use_idf = bool(tfidf_cfg.get("use_idf", True))
            smooth_idf = bool(tfidf_cfg.get("smooth_idf", True))
            sublinear_tf = bool(tfidf_cfg.get("sublinear_tf", False))
            norm = tfidf_cfg.get("norm", "l2")

            if target in ("ph", "both"):
                self.split_data.train.X_ph = _tfidf_transform_csr(
                    self.split_data.train.X_ph, use_idf, smooth_idf, sublinear_tf, norm)
                self.split_data.valid.X_ph = _tfidf_transform_csr(
                    self.split_data.valid.X_ph, use_idf, smooth_idf, sublinear_tf, norm)
                self.split_data.test.X_ph = _tfidf_transform_csr(
                    self.split_data.test.X_ph, use_idf, smooth_idf, sublinear_tf, norm)

            if target in ("ps", "both"):
                self.split_data.train.X_ps = _tfidf_transform_csr(
                    self.split_data.train.X_ps, use_idf, smooth_idf, sublinear_tf, norm)
                self.split_data.valid.X_ps = _tfidf_transform_csr(
                    self.split_data.valid.X_ps, use_idf, smooth_idf, sublinear_tf, norm)
                self.split_data.test.X_ps = _tfidf_transform_csr(
                    self.split_data.test.X_ps, use_idf, smooth_idf, sublinear_tf, norm)

        H = data.F_h.shape[0]
        S = data.F_s.shape[0]

        # 3. 构建图
        logger.info("=== 构建图 ===")
        graph_cfg = cfg.get("graph", {})
        knn_k = graph_cfg.get("knn_k", 10)
        lap_mode = graph_cfg.get("laplacian", "unnormalized")

        # 禁忌矩阵
        self.C_hh = build_contraindication_matrix(data.herb_mutex, H)

        herb_graph = build_herb_graph(
            herb_cooc=data.herb_cooc, F_h=data.F_h, H=H,
            eta=graph_cfg.get("herb_eta", [1.0, 0.5, 0.5]),
            knn_k=knn_k, laplacian_mode=lap_mode,
            C_neg=self.C_hh,
        )
        self.L_h = herb_graph.L_pos

        symptom_graph = build_symptom_graph(
            symptom_cooc=data.symptom_cooc, F_s=data.F_s, S=S,
            xi=graph_cfg.get("symptom_xi", [1.0, 0.5, 0.5]),
            knn_k=knn_k, laplacian_mode=lap_mode,
        )
        self.L_s = symptom_graph.L_pos

        # 3b. 构建超图拉普拉斯 (仅当需要 Laplacian 形式时)
        hyper_cfg = cfg.get("hypergraph", {})
        _sw = cfg.get("loss_switches", {})
        _need_laplacian = _sw.get("hyper_h", False) and (
            hyper_cfg.get("use_prescription_edges", False)
            or hyper_cfg.get("use_motif_edges", False)
            or hyper_cfg.get("use_attribute_edges", False)
        )
        if _need_laplacian:
            logger.info("=== 构建药材超图 ===")
            X_ph_train_dense = _to_dense(self.split_data.train.X_ph)
            hyper_bundle = build_herb_hypergraph(
                X_ph_train=X_ph_train_dense,
                F_h=data.F_h,
                H=H,
                cfg=hyper_cfg,
            )
            self.L_hyper_h = hyper_bundle.L_total
            self._hyper_bundle = hyper_bundle
            logger.info("超图统计: %s", hyper_bundle.stats)
        else:
            self.L_hyper_h = None
            self._hyper_bundle = None

        # 4. 初始化模型
        logger.info("=== 初始化模型 ===")
        train_cfg = cfg.get("training", {})
        P_train = self.split_data.train.X_ph.shape[0]

        self.model = MVGSNMTF(
            K=cfg["K"],
            alpha=cfg.get("alpha", 1.0),
            beta=cfg.get("beta", 0.2),
            beta_pair=cfg.get("beta_pair", 0.0),
            lambda_h=cfg.get("lambda_h", 1e-3),
            lambda_s=cfg.get("lambda_s", 1e-3),
            lambda_hyper=float(cfg.get("lambda_hyper", 0.0)),
            lambda_know=float(cfg.get("lambda_know", 0.0)),
            gamma_g=cfg.get("gamma_g", 0.0),
            gamma_h=cfg.get("gamma_h", 1e-5),
            gamma_s=cfg.get("gamma_s", 1e-5),
            gamma_d=cfg.get("gamma_d", 1e-6),
            rho=cfg.get("rho", 0.0),
            role_aware=bool(cfg.get("role_aware", False)),
            n_roles=int(cfg.get("n_roles", 4)),
            role_exclusive=float(cfg.get("role_exclusive", 0.0)),
            normalize_columns=bool(cfg.get("normalize_columns", False)),
            update_rule=train_cfg.get("update_rule", "pgd"),
            lr=train_cfg.get("lr", 1e-3),
            device=cfg.get("device", "cpu"),
            loss_switches=cfg.get("loss_switches"),
        )
        requested_device = str(cfg.get("device", "cpu")).lower()
        if requested_device == "gpu" and self.model.xp.__name__ != "cupy":
            raise RuntimeError(
                "GPU execution was requested, but the CuPy backend is unavailable. "
                "Refusing to continue on CPU."
            )
        # Hybrid 模式下 H_s 的 PGD 内迭代步数
        self.model.inner_steps_hs = int(train_cfg.get("inner_steps_hs", 1))

        init_method = train_cfg.get("init_method", "nndsvd")
        X_ph_train = _to_dense(self.split_data.train.X_ph)
        X_ps_train = _to_dense(self.split_data.train.X_ps)

        N_pair = int(self.split_data.train.X_pair.shape[1])
        self.model.init_factors(
            P=P_train, H=H, S=S,
            method=init_method,
            X_ph=X_ph_train, X_ps=X_ps_train,
            N_pair=N_pair,
            seed=seed,
        )

        # 5. 注入直接超图正则数据 (hyper_var / hyper_mean)
        sw = cfg.get("loss_switches", {})
        if (sw.get("hyper_var", False) or sw.get("hyper_mean", False)):
            all_edges = []
            all_weights = []

            # 从 hypergraph bundle 收集
            if hasattr(self, '_hyper_bundle') and self._hyper_bundle is not None:
                bundle = self._hyper_bundle
                if bundle._attr_edges and bundle._attr_weights is not None:
                    all_edges.extend(bundle._attr_edges)
                    all_weights.append(bundle._attr_weights)
                if bundle._motif_edges and bundle._motif_weights is not None:
                    all_edges.extend(bundle._motif_edges)
                    all_weights.append(bundle._motif_weights)
                if bundle._pres_edges and bundle._pres_weights is not None:
                    all_edges.extend(bundle._pres_edges)
                    all_weights.append(bundle._pres_weights)

            # 类别超边 (直接从离散分类标签构造)
            if hyper_cfg.get("use_category_edges", False):
                from gsnmf.hypergraph_builder import build_category_hyperedges
                # F_h = [C_cat(30) || A_feat(21)], 取前 30 列
                C_cat = data.F_h[:, :30]
                cat_edges, cat_weights = build_category_hyperedges(
                    C_cat,
                    min_size=hyper_cfg.get("category", {}).get("min_size", 2),
                )
                all_edges.extend(cat_edges)
                all_weights.append(cat_weights)

            if all_edges:
                self.model.hyper_edges = all_edges
                self.model.hyper_weights = np.concatenate(all_weights)
                logger.info("直接超图正则: %d 条超边注入模型", len(all_edges))
            else:
                logger.warning("hyper_var/hyper_mean 开启但无可用超边")

        # 5. Run 目录
        project_root = str(Path(__file__).resolve().parent.parent)
        artifacts_dir = os.path.join(project_root, "artifacts")
        self.run_dir = _init_run_dir(artifacts_dir, cfg)

    # -----------------------------------------------------------------
    # 训练
    # -----------------------------------------------------------------

    def train(
        self,
        eval_fn: Optional[Callable] = None,
    ) -> ModelFactors:
        """主训练循环。

        Parameters
        ----------
        eval_fn : callable, optional
            验证集评估函数 (model, split_data, C_hh) → dict

        Returns
        -------
        ModelFactors : 最佳模型因子
        """
        cfg = self.cfg
        train_cfg = cfg.get("training", {})
        max_iter = train_cfg.get("max_iter", 300)
        tol = train_cfg.get("tol", 1e-5)
        log_every = train_cfg.get("log_every", 10)
        eval_every = int(train_cfg.get("eval_every", log_every))
        patience = train_cfg.get("early_stop_patience", 20)
        es_metric = train_cfg.get("early_stop_metric", "valid_map_sym2herb")
        disable_early_stop = bool(train_cfg.get("disable_early_stop", False))

        # 正则 warmup: 前 warmup_iters 线性放大正则权重，降低早期拉偏
        warmup_iters = int(train_cfg.get("regularization_warmup_iters", 0))

        # 末段因子平均: 保存最后 N 次 checkpoint 做参数平均（近似 PTM 采样平均）
        average_last_n_checkpoints = int(train_cfg.get("average_last_n_checkpoints", 0))

        # 稠密化训练数据并传到设备
        sl = self.split_data.train
        xp = self.model.xp
        # GPU 训练统一用 float32，提升吞吐
        X_ph = to_device(_to_dense(sl.X_ph).astype(np.float32, copy=False), xp)
        X_ps = to_device(_to_dense(sl.X_ps).astype(np.float32, copy=False), xp)

        if self.model.sw["pd"]:
            X_pd_raw = _to_dense(sl.X_pd)
            M_pd_raw = _to_dense(sl.M_pd)
            # 按列最大值归一化剂量到 [0,1]，使 pd loss 与 ph/ps 同量级
            col_max = X_pd_raw.max(axis=0, keepdims=True)
            col_max = np.maximum(col_max, 1e-10)  # 避免除零
            self._dose_scale = col_max  # 保存用于评估时反归一化
            X_pd_normed = (X_pd_raw / col_max).astype(np.float32, copy=False)
            M_pd_normed = M_pd_raw.astype(np.float32, copy=False)  # mask 不变
            X_pd = to_device(X_pd_normed, xp)
            M_pd = to_device(M_pd_normed, xp)
            logger.info("剂量归一化: max=%.1f, mean_nonzero=%.3f",
                        float(col_max.max()), float(X_pd_normed[M_pd_raw > 0].mean()))
        else:
            X_pd = None
            M_pd = None

        X_pair = None
        if self.model.sw.get("pair", False):
            if xp.__name__ == "cupy":
                # Keep the pair view sparse on GPU. At the lowest support
                # threshold, a dense float32 copy alone exceeds 2 GiB.
                X_pair = to_device(
                    sl.X_pair.astype(np.float32, copy=False).tocsr(), xp
                )
            else:
                # Preserve the existing CPU path and its numerical behavior.
                X_pair = _to_dense(sl.X_pair).astype(np.float32, copy=False)

        K_hs = None
        if self.model.sw.get("know_hs", False):
            K_hs = to_device(self.split_data.meta.K_hs.astype(np.float32, copy=False), xp)

        # 稀疏拉普拉斯也传到设备
        if xp.__name__ == "cupy":
            L_h = (
                to_device(self.L_h.astype(np.float32, copy=False), xp)
                if self.L_h is not None else None
            )
            L_s = (
                to_device(self.L_s.astype(np.float32, copy=False), xp)
                if self.L_s is not None else None
            )
        else:
            L_h = to_device(self.L_h, xp) if self.L_h is not None else None
            L_s = to_device(self.L_s, xp) if self.L_s is not None else None
        C_hh = to_device(self.C_hh, xp) if self.C_hh is not None else None

        # 超图拉普拉斯: 若过于稠密则转为 dense (稀疏格式操作稠密矩阵极慢)
        L_hyper_h = None
        if self.L_hyper_h is not None:
            nnz = self.L_hyper_h.nnz
            total = self.L_hyper_h.shape[0] * self.L_hyper_h.shape[1]
            density = nnz / total if total > 0 else 0
            if density > 0.3:
                logger.info("L_hyper_h 密度 %.1f%% > 30%%, 转为 dense 加速",
                            density * 100)
                L_hyper_h = to_device(self.L_hyper_h.toarray(), xp)
            else:
                L_hyper_h = to_device(self.L_hyper_h, xp)

        # 预训练: 先用基础重构收敛, 再打开复杂正则 (保持 NMF 核心)
        pretrain_iters = int(train_cfg.get("pretrain_iters", 0))
        pretrain_switches = train_cfg.get("pretrain_switches", {
            "ph": True,
            "ps": True,
            "pd": False,
            "graph_h": False,
            "graph_s": False,
            "hyper_h": False,
            "hyper_var": False,
            "hyper_mean": False,
            "l1": False,
            "contra": False,
            "pair": False,
            "know_hs": False,
        })

        best_factors = self.model.factors()
        _lower_better = any(x in es_metric for x in
                            ["nre", "ppl", "perplexity", "loss", "mae", "rmse"])
        best_metric = np.inf if _lower_better else -np.inf
        best_iter = 0
        wait = 0
        prev_loss = np.inf

        # 末段因子平均缓存
        avg_buffer: list[ModelFactors] = []

        logger.info("=== 开始训练 (max_iter=%d, K=%d) ===", max_iter, cfg["K"])
        t0 = time.time()

        # 0) 预训练阶段 (可选)
        if pretrain_iters > 0:
            logger.info("=== 预训练阶段: %d iter (基础重构) ===", pretrain_iters)
            sw_backup = dict(self.model.sw)
            self.model._switches.update(pretrain_switches)

            for it in trange(1, pretrain_iters + 1, desc="MV-GSNMTF-Pretrain"):
                lc = self.model.fit_step(
                    X_ph, X_ps, X_pd, M_pd, X_pair,
                    L_h, L_s, C_hh, L_hyper_h, K_hs,
                )
                self.history.append({"phase": "pretrain", **lc.as_dict()})
                if it % log_every == 0 or it == 1:
                    logger.info(
                        "Pretrain %4d | total=%.4f | ph=%.4f ps=%.4f",
                        it, lc.total, lc.loss_ph, lc.loss_ps,
                    )

            # 恢复完整 loss 开关
            self.model._switches = sw_backup
            prev_loss = np.inf

        # 保存原始正则权重（用于 warmup 后恢复）
        base_regs = {
            "lambda_h": float(self.model.lambda_h),
            "lambda_s": float(self.model.lambda_s),
            "lambda_hyper": float(self.model.lambda_hyper),
            "gamma_g": float(self.model.gamma_g),
            "gamma_h": float(self.model.gamma_h),
            "gamma_s": float(self.model.gamma_s),
            "gamma_d": float(self.model.gamma_d),
            "rho": float(self.model.rho),
            "beta_pair": float(self.model.beta_pair),
            "lambda_know": float(self.model.lambda_know),
        }

        for it in trange(1, max_iter + 1, desc="MV-GSNMTF"):
            # --- regularization warmup ---
            if warmup_iters > 0 and it <= warmup_iters:
                scale = it / float(warmup_iters)
                self.model.lambda_h = base_regs["lambda_h"] * scale
                self.model.lambda_s = base_regs["lambda_s"] * scale
                self.model.lambda_hyper = base_regs["lambda_hyper"] * scale
                self.model.gamma_g = base_regs["gamma_g"] * scale
                self.model.gamma_h = base_regs["gamma_h"] * scale
                self.model.gamma_s = base_regs["gamma_s"] * scale
                self.model.gamma_d = base_regs["gamma_d"] * scale
                self.model.rho = base_regs["rho"] * scale
                self.model.beta_pair = base_regs["beta_pair"] * scale
                self.model.lambda_know = base_regs["lambda_know"] * scale
            elif warmup_iters > 0 and it == warmup_iters + 1:
                # 恢复目标权重
                self.model.lambda_h = base_regs["lambda_h"]
                self.model.lambda_s = base_regs["lambda_s"]
                self.model.lambda_hyper = base_regs["lambda_hyper"]
                self.model.gamma_g = base_regs["gamma_g"]
                self.model.gamma_h = base_regs["gamma_h"]
                self.model.gamma_s = base_regs["gamma_s"]
                self.model.gamma_d = base_regs["gamma_d"]
                self.model.rho = base_regs["rho"]
                self.model.beta_pair = base_regs["beta_pair"]
                self.model.lambda_know = base_regs["lambda_know"]

            # --- 1 step ---
            lc = self.model.fit_step(
                X_ph, X_ps, X_pd, M_pd, X_pair,
                L_h, L_s, C_hh, L_hyper_h, K_hs,
            )

            self.history.append(lc.as_dict())

            # --- 日志 ---
            if it % log_every == 0 or it == 1:
                logger.info(
                    "Iter %4d | total=%.4f | ph=%.4f ps=%.4f pd=%.4f "
                    "| gh=%.4f gs=%.4f hyp=%.4f | l1=%.4f | ctr=%.4f know=%.4f",
                    it, lc.total, lc.loss_ph, lc.loss_ps, lc.loss_pd,
                    lc.loss_graph_h, lc.loss_graph_s, lc.loss_hyper_h,
                    lc.loss_l1_g + lc.loss_l1_h + lc.loss_l1_s + lc.loss_l1_d,
                    lc.loss_contra, lc.loss_know_hs,
                )

            # --- 收敛判断 ---
            if prev_loss > 0:
                rel_change = abs(lc.total - prev_loss) / (abs(prev_loss) + 1e-10)
                if rel_change < tol:
                    logger.info("收敛于 iter %d (rel_change=%.2e < tol=%.2e)",
                                it, rel_change, tol)
                    break
            prev_loss = lc.total

            # --- checkpoint buffer for averaging ---
            if average_last_n_checkpoints > 0:
                avg_buffer.append(self.model.factors())
                if len(avg_buffer) > average_last_n_checkpoints:
                    avg_buffer.pop(0)

            # --- Early stopping (验证集) ---
            if (
                eval_fn is not None
                and eval_every > 0
                and it % eval_every == 0
            ):
                metrics = eval_fn(self.model, self.split_data, self.C_hh)
                current_metric = metrics.get(es_metric, 0.0)
                # NRE / perplexity: lower is better; MAP/NDCG: higher is better
                _lower_better = any(x in es_metric for x in
                                    ["nre", "ppl", "perplexity", "loss", "mae", "rmse"])
                if _lower_better:
                    improved = current_metric < best_metric
                else:
                    improved = current_metric > best_metric
                if improved:
                    best_metric = current_metric
                    best_factors = self.model.factors()
                    best_iter = it
                    wait = 0
                    logger.info("新 best %s=%.4f @ iter %d",
                                es_metric, best_metric, it)
                else:
                    wait += 1
                    if (not disable_early_stop) and wait >= patience:
                        logger.info("Early stopping @ iter %d (patience=%d)",
                                    it, patience)
                        break

        elapsed = time.time() - t0
        logger.info("训练完成, 耗时 %.1f s, %d 轮", elapsed, len(self.history))

        # 如果关闭 early-stop，或未提供 eval_fn，则优先使用末段平均因子
        if average_last_n_checkpoints > 0 and len(avg_buffer) > 0:
            G_p = np.mean([f.G_p for f in avg_buffer], axis=0)
            H_h = np.mean([f.H_h for f in avg_buffer], axis=0)
            H_s = np.mean([f.H_s for f in avg_buffer], axis=0)
            D_h = np.mean([f.D_h for f in avg_buffer], axis=0)
            G_p_roles = np.mean([f.G_p_roles for f in avg_buffer], axis=0) if avg_buffer[0].G_p_roles is not None else None
            H_h_roles = np.mean([f.H_h_roles for f in avg_buffer], axis=0) if avg_buffer[0].H_h_roles is not None else None
            if all(f.H_pair is not None for f in avg_buffer):
                H_pair = np.mean([f.H_pair for f in avg_buffer], axis=0)
            else:
                H_pair = None
            best_factors = ModelFactors(G_p=G_p, H_h=H_h, H_s=H_s, D_h=D_h, H_pair=H_pair, G_p_roles=G_p_roles, H_h_roles=H_h_roles)
            best_iter = len(self.history)
            logger.info("使用末段平均因子 (last_n=%d)", len(avg_buffer))
        elif eval_fn is None:
            best_factors = self.model.factors()
            best_iter = len(self.history)

        # --- H_s 后训练精调 (Post-training Refinement) ---
        refine_hs_iters = int(train_cfg.get("refine_hs_iters", 0))
        if refine_hs_iters > 0:
            logger.info("=== H_s 后训练精调 (%d 步) ===", refine_hs_iters)
            # 恢复 best 因子到模型
            self.model.load_factors(best_factors)
            # 冻结 G_p, H_h: 只保存引用, 每轮结束后恢复
            frozen_G_p = self.model.G_p.copy()
            frozen_H_h = self.model.H_h.copy()
            frozen_D_h = self.model.D_h.copy() if self.model.D_h is not None else None
            frozen_H_pair = self.model.H_pair.copy() if self.model.H_pair is not None else None
            if getattr(self.model, 'G_p_roles', None) is not None:
                frozen_G_p_roles = self.model.G_p_roles.copy()
            else:
                frozen_G_p_roles = None
            if getattr(self.model, 'H_h_roles', None) is not None:
                frozen_H_h_roles = self.model.H_h_roles.copy()
            else:
                frozen_H_h_roles = None

            refine_lr = train_cfg.get("refine_hs_lr", self.model.lr)
            refine_clip = self.model.grad_clip

            # 准备 L_s (症状图拉普拉斯)
            L_s_ref = L_s
            K_hs_ref = K_hs  # 复用训练时的知识矩阵

            for r_it in range(refine_hs_iters):
                # 只更新 H_s (PGD)
                grad_s = self.model._clip_grad(
                    self.model._grad_Hs(X_ps, L_s_ref, K_hs_ref), refine_clip)
                self.model.H_s = self.model._proj_step(
                    self.model.H_s, grad_s, refine_lr)
                if self.model.sw["l1"]:
                    self.model.H_s = self.model._prox_l1(
                        self.model.H_s, self.model.gamma_s, refine_lr)

                # 恢复冻结的 G_p, H_h (防止任何意外修改)
                self.model.G_p = frozen_G_p.copy()
                self.model.H_h = frozen_H_h.copy()
                if frozen_D_h is not None:
                    self.model.D_h = frozen_D_h.copy()
                if frozen_H_pair is not None:
                    self.model.H_pair = frozen_H_pair.copy()
                if frozen_G_p_roles is not None:
                    self.model.G_p_roles = frozen_G_p_roles.copy()
                if frozen_H_h_roles is not None:
                    self.model.H_h_roles = frozen_H_h_roles.copy()

                if (r_it + 1) % 50 == 0 or r_it == refine_hs_iters - 1:
                    # 计算当前 H_s 的症状重构残差作为参考
                    X_ps_np = X_ps.get() if hasattr(X_ps, 'get') else np.asarray(X_ps)
                    G_p_np = self.model.G_p if not hasattr(self.model.G_p, 'get') else self.model.G_p.get()
                    H_s_np = self.model.H_s if not hasattr(self.model.H_s, 'get') else self.model.H_s.get()
                    ps_res = float(np.linalg.norm(G_p_np @ H_s_np.T - X_ps_np))
                    logger.info("  refine H_s iter %d/%d: ps_residual=%.4f",
                                r_it + 1, refine_hs_iters, ps_res)

            best_factors = self.model.factors()
            logger.info("H_s 精调完成")

        # --- PPL-Aware EM 重估 H_s_ppl + H_h_ppl ---
        ppl_refine_cfg = self.cfg.get("ppl_refine", {})
        ppl_refine_iters = ppl_refine_cfg.get("em_iters", 10)
        if ppl_refine_iters > 0:
            logger.info("=== EM 重估 PPL 因子 (%d 步) ===", ppl_refine_iters)
            # 准备训练集的二值矩阵 (PPL 在二值空间计算)
            X_ph_train_np = self.split_data.train.X_ph
            if sp.issparse(X_ph_train_np):
                X_ph_train_np = X_ph_train_np.toarray()
            X_ph_train_np = (np.asarray(X_ph_train_np) > 0).astype(np.float64)

            X_ps_train_np = self.split_data.train.X_ps
            if sp.issparse(X_ps_train_np):
                X_ps_train_np = X_ps_train_np.toarray()
            X_ps_train_np = (np.asarray(X_ps_train_np) > 0).astype(np.float64)

            H_h_np = best_factors.H_h
            if hasattr(H_h_np, 'get'):
                H_h_np = H_h_np.get()
            H_s_np = best_factors.H_s
            if hasattr(H_s_np, 'get'):
                H_s_np = H_s_np.get()

            alpha_inf = self.cfg.get("inference", {}).get(
                "dirichlet_alpha", 0.01)
            beta_bar = ppl_refine_cfg.get("beta_bar", 0.1)

            # H_s_ppl: herbs → θ → 重估 P(sym|topic)
            logger.info("--- EM 重估 H_s_ppl (Sym PPL) ---")
            H_s_ppl = reestimate_Hs_em_vectorized(
                H_h=H_h_np,
                X_ph_train=X_ph_train_np,
                X_ps_train=X_ps_train_np,
                n_iters=ppl_refine_iters,
                beta_bar=beta_bar,
                dirichlet_alpha=alpha_inf,
                H_s_init=H_s_np,
            )
            self._H_s_ppl = H_s_ppl

            # H_h_ppl: symptoms → θ → 重估 P(herb|topic)
            logger.info("--- EM 重估 H_h_ppl (Herb PPL) ---")
            H_h_ppl = reestimate_Hh_em_vectorized(
                H_s=H_s_np,
                X_ps_train=X_ps_train_np,
                X_ph_train=X_ph_train_np,
                n_iters=ppl_refine_iters,
                beta=beta_bar,
                dirichlet_alpha=alpha_inf,
                H_h_init=H_h_np,
            )
            self._H_h_ppl = H_h_ppl

            logger.info("PPL 因子 EM 重估完成")
        else:
            self._H_s_ppl = None
            self._H_h_ppl = None

        # --- 保存 ---
        self._save(best_factors,
                   eval_fn=eval_fn,
                   best_iter=best_iter,
                   best_val_metric=best_metric,
                   es_metric=es_metric)

        return best_factors

    # -----------------------------------------------------------------
    # 保存
    # -----------------------------------------------------------------

    def _save(self, factors: ModelFactors,
              eval_fn=None, best_iter: int = 0,
              best_val_metric: float = 0.0,
              es_metric: str = ""):
        """保存因子、训练历史和 summary.json 到 run 目录。"""
        if self.run_dir is None:
            return

        # 因子
        save_kwargs = {
            "G_p": factors.G_p,
            "H_h": factors.H_h,
            "H_s": factors.H_s,
            "D_h": factors.D_h,
        }
        if factors.H_pair is not None:
            save_kwargs["H_pair"] = factors.H_pair
        if factors.G_p_roles is not None:
            save_kwargs["G_p_roles"] = factors.G_p_roles
        if factors.H_h_roles is not None:
            save_kwargs["H_h_roles"] = factors.H_h_roles
        # PPL 专用因子
        if getattr(self, '_H_s_ppl', None) is not None:
            save_kwargs["H_s_ppl"] = self._H_s_ppl
        if getattr(self, '_H_h_ppl', None) is not None:
            save_kwargs["H_h_ppl"] = self._H_h_ppl
        np.savez_compressed(
            os.path.join(self.run_dir, "factors.npz"),
            **save_kwargs,
        )

        # 训练历史
        with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        # summary.json (含全部测试指标)
        summary = {
            "model_name": self.cfg.get("model_name", ""),
            "train_seed": self.cfg.get("seed", 42),
            "split_seed": self.cfg.get("split", {}).get("seed",
                                       self.cfg.get("seed", 42)),
            "K": self.cfg.get("K", 30),
            "best_iter": best_iter,
            "total_iter": len(self.history),
            "early_stop_metric": es_metric,
            "best_val_metric": best_val_metric,
        }

        # 用 best 因子重新评估测试集指标
        if eval_fn is not None:
            # 恢复 best 因子
            self.model.load_factors(factors)
            test_metrics = eval_fn(self.model, self.split_data, self.C_hh)
            summary["test_metrics"] = test_metrics

        # 保存配置副本
        summary["config"] = self.cfg

        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("模型已保存至 %s", self.run_dir)
