"""
评估指标。

推荐: P@K, R@K, MAP, NDCG@K
重构: NRE
剂量: MAE, RMSE, sMAPE (仅 M_pd=1 位置)
主题: topic_coherence (先基于 H_h, 药材主题 PMI)
安全: violation_at_k, constraint_compatibility_rate
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse as sp

from gsnmf.constraints import count_violations
from gsnmf.recommender import symptoms_to_herbs

logger = logging.getLogger(__name__)


# =========================================================================
# 推荐指标
# =========================================================================

def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """P@K: 推荐 top-k 中有多少命中真实正例。"""
    top_k = np.argsort(y_scores)[-k:][::-1]
    return float(np.sum(y_true[top_k] > 0)) / k


def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """R@K: 推荐 top-k 覆盖了多少真实正例。"""
    n_pos = np.sum(y_true > 0)
    if n_pos == 0:
        return 0.0
    top_k = np.argsort(y_scores)[-k:][::-1]
    return float(np.sum(y_true[top_k] > 0)) / n_pos


def average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """AP: 单样本的 Average Precision。"""
    order = np.argsort(y_scores)[::-1]
    y_sorted = y_true[order]
    n_pos = np.sum(y_true > 0)
    if n_pos == 0:
        return 0.0
    cumsum = np.cumsum(y_sorted > 0)
    precision_at = cumsum / np.arange(1, len(y_sorted) + 1)
    return float(np.sum(precision_at * (y_sorted > 0))) / n_pos


def ndcg_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """NDCG@K。"""
    order = np.argsort(y_scores)[-k:][::-1]
    gains = y_true[order].astype(float)
    discounts = np.log2(np.arange(1, k + 1) + 1)
    dcg = np.sum(gains / discounts)

    ideal_order = np.argsort(y_true)[-k:][::-1]
    ideal_gains = y_true[ideal_order].astype(float)
    idcg = np.sum(ideal_gains / discounts)
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def _batch_recommend_metrics(
    X_query: np.ndarray,
    X_truth: np.ndarray,
    H_query: np.ndarray,
    H_target: np.ndarray,
    ks: List[int] = [5, 10],
    dirichlet_alpha: float = 0.0,
    infer_method: str = "nnls",
) -> Dict[str, float]:
    """批量计算推荐指标。"""
    P = X_query.shape[0]
    results: Dict[str, List[float]] = {
        "map": [], **{f"p@{k}": [] for k in ks},
        **{f"r@{k}": [] for k in ks},
        **{f"ndcg@{k}": [] for k in ks},
    }

    for i in range(P):
        q = X_query[i]
        truth = X_truth[i]
        if np.sum(q) == 0 or np.sum(truth) == 0:
            continue

        _, scores = symptoms_to_herbs(q, H_query, H_target,
                                      dirichlet_alpha=dirichlet_alpha,
                                      infer_method=infer_method)

        results["map"].append(average_precision(truth, scores))
        for k in ks:
            results[f"p@{k}"].append(precision_at_k(truth, scores, k))
            results[f"r@{k}"].append(recall_at_k(truth, scores, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(truth, scores, k))

    return {key: float(np.mean(vals)) if vals else 0.0
            for key, vals in results.items()}


# =========================================================================
# 重构指标
# =========================================================================

def nre(X: np.ndarray, X_hat: np.ndarray) -> float:
    """Normalized Reconstruction Error: ‖X - X̂‖_F / ‖X‖_F。"""
    norm_x = np.linalg.norm(X, 'fro')
    if norm_x == 0:
        return 0.0
    return float(np.linalg.norm(X - X_hat, 'fro') / norm_x)


def topic_perplexity(
    X: np.ndarray,
    X_hat: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """NMF 困惑度 (perplexity)。

    将重构矩阵 X_hat 行归一化为概率分布, 计算测试集
    观测药材的 log-likelihood, 再转为困惑度:

        P_hat[p, h] = X_hat[p, h] / Σ_j X_hat[p, j]
        log_lik = Σ_p Σ_{h: X[p,h]>0} log P_hat[p, h]
        perplexity = exp(-log_lik / N)

    其中 N = 分子矩阵非零元素数 (观测到的药材总数)。
    越低越好, 表示模型对新处方药材组成的预测能力越强。

    Parameters
    ----------
    X : (P, H) 真实二值/存在矩阵 (测试集)
    X_hat : (P, H) 重构矩阵 (G_val @ H_h.T)
    eps : 平滑常数, 防止 log(0)

    Returns
    -------
    float : perplexity 值
    """
    # 行归一化为概率
    X_hat_pos = np.clip(X_hat, 0, None) + eps
    row_sums = X_hat_pos.sum(axis=1, keepdims=True)
    P_hat = X_hat_pos / row_sums

    # 观测 mask
    mask = X > 0
    N = mask.sum()
    if N == 0:
        return 0.0

    # log-likelihood
    log_P = np.log(P_hat)
    log_lik = (mask * log_P).sum()

    perplexity_val = np.exp(-log_lik / N)
    return float(perplexity_val)


# ---- Held-out perplexity (三种方案) ----

def _perplexity_from_probs(
    X_heldout: np.ndarray,
    X_hat: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """内部工具: 从重构矩阵计算 heldout 位置的 perplexity。"""
    X_hat_pos = np.clip(X_hat, 0, None) + eps
    row_sums = X_hat_pos.sum(axis=1, keepdims=True)
    P_hat = X_hat_pos / row_sums
    mask = X_heldout > 0
    N = mask.sum()
    if N == 0:
        return 0.0
    log_lik = (mask * np.log(P_hat)).sum()
    return float(np.exp(-log_lik / N))


def heldout_perplexity_masked(
    X_ph: np.ndarray,
    H_h: np.ndarray,
    H_h_target: np.ndarray | None = None,
    mask_ratio: float = 0.2,
    seed: int = 42,
    eps: float = 1e-10,
    probabilize: bool = False,
    dirichlet_alpha: float = 0.1,
    infer_method: str = "nnls",
) -> float:
    """方案1: Masked Herb Perplexity.

    对每个测试处方:
      1. 将观测到的药材随机分为 observed (80%) 与 heldout (20%)
      2. 用 observed 部分通过 NNLS 推断 G_val
      3. 在 heldout 部分计算 perplexity

    Parameters
    ----------
    X_ph : (P, H) 测试集处方-药材二值矩阵
    H_h  : (H, K) 药材主题载荷 (用于推断)
    H_h_target : (H, K) 用于生成预测的载荷 (如果 TF-IDF 解耦则传入 raw H)
    mask_ratio : heldout 的比例 (默认 0.2)
    seed : 随机种子
    eps  : 平滑常数
    probabilize : 是否对因子做概率化归一
    dirichlet_alpha : Dirichlet 平滑先验 (仅 probabilize=True 时生效)

    Returns
    -------
    float : heldout perplexity
    """
    from scipy.optimize import nnls
    rng = np.random.RandomState(seed)
    P, H = X_ph.shape
    K = H_h.shape[1]
    
    if H_h_target is None:
        H_h_target = H_h

    # 因子概率化: 将 H_h_target 列归一化为 p(herb|topic)
    if probabilize:
        H_h_prob = H_h_target / (H_h_target.sum(axis=0, keepdims=True) + eps)
    
    total_log_lik = 0.0
    total_N = 0

    for i in range(P):
        obs_idx = np.where(X_ph[i] > 0)[0]
        n_obs = len(obs_idx)
        if n_obs < 2:  # 至少 2 个药材才能分
            continue

        n_heldout = max(1, int(n_obs * mask_ratio))
        perm = rng.permutation(n_obs)
        heldout_idx = obs_idx[perm[:n_heldout]]
        train_idx = obs_idx[perm[n_heldout:]]

        from gsnmf.recommender import _solve_infer
        x_train = np.zeros(H)
        x_train[train_idx] = X_ph[i, train_idx]
        g_i = _solve_infer(x_train, H_h, method=infer_method)
        if probabilize:
            g_i = np.clip(g_i, 0, None)

        if probabilize:
            # Dirichlet 平滑 + 行归一化 → θ (topic mixture)
            g_smooth = g_i + dirichlet_alpha
            theta = g_smooth / g_smooth.sum()
            p_hat = theta @ H_h_prob.T  # 概率分布, 自然归一
            p_hat = np.clip(p_hat, eps, None)  # 防 log(0)
        else:
            # 原始方式: 重构后行归一化
            x_hat = g_i @ H_h_target.T  # (H,)
            x_hat_pos = np.clip(x_hat, 0, None) + eps
            p_hat = x_hat_pos / x_hat_pos.sum()

        total_log_lik += np.log(p_hat[heldout_idx]).sum()
        total_N += n_heldout

    if total_N == 0:
        return 0.0
    return float(np.exp(-total_log_lik / total_N))


def heldout_perplexity_half(
    X_ph: np.ndarray,
    H_h: np.ndarray,
    H_h_target: np.ndarray | None = None,
    seed: int = 42,
    eps: float = 1e-10,
    probabilize: bool = False,
    dirichlet_alpha: float = 0.1,
    infer_method: str = "nnls",
) -> float:
    """方案2: Half-Split Herb Perplexity.

    将每个测试处方的药材随机分成两半:
      - 前半用于 NNLS 推断 G
      - 后半计算 perplexity

    比 masked 更严格 (50/50 分), 类似论文 5.1.3 的设计。
    """
    return heldout_perplexity_masked(
        X_ph, H_h, H_h_target=H_h_target, mask_ratio=0.5, seed=seed, eps=eps,
        probabilize=probabilize, dirichlet_alpha=dirichlet_alpha, infer_method=infer_method)


def crossmodal_perplexity(
    X_ps: np.ndarray,
    X_ph: np.ndarray,
    H_s: np.ndarray,
    H_h: np.ndarray,
    eps: float = 1e-10,
    probabilize: bool = False,
    dirichlet_alpha: float = 0.1,
    infer_method: str = "nnls",
) -> float:
    """方案3: Cross-Modal Perplexity (症状→药材).

    用症状侧信息推断主题分布, 然后预测药材:
      1. 通过 X_ps 和 H_s 用 NNLS 投影得到 G_val
      2. 用 G_val @ H_h^T 预测药材分布
      3. 在真实药材 X_ph 上算 perplexity

    最接近论文 5.1.1 的 herb predictive perplexity。

    当 probabilize=True 时:
      - H_h 列归一化为 p(herb|topic)
      - NNLS 推断的 g 加 Dirichlet 先验后行归一化为 θ
      - p(herb) = θ @ H_h_prob.T 是严格概率分布
    """
    from scipy.optimize import nnls
    P, S = X_ps.shape
    H = X_ph.shape[1]
    K = H_s.shape[1]

    # 因子概率化
    if probabilize:
        H_h_prob = H_h / (H_h.sum(axis=0, keepdims=True) + eps)

    total_log_lik = 0.0
    total_N = 0

    for i in range(P):
        if np.sum(X_ps[i]) == 0 or np.sum(X_ph[i]) == 0:
            continue

        from gsnmf.recommender import _solve_infer
        g_i = _solve_infer(X_ps[i], H_s, method=infer_method)
        if probabilize:
            g_i = np.clip(g_i, 0, None)

        if probabilize:
            g_smooth = g_i + dirichlet_alpha
            theta = g_smooth / g_smooth.sum()
            p_hat = theta @ H_h_prob.T
            p_hat = np.clip(p_hat, eps, None)
        else:
            # 药材侧预测
            x_hat = g_i @ H_h.T
            x_hat_pos = np.clip(x_hat, 0, None) + eps
            p_hat = x_hat_pos / x_hat_pos.sum()

        herb_idx = np.where(X_ph[i] > 0)[0]
        total_log_lik += np.log(p_hat[herb_idx]).sum()
        total_N += len(herb_idx)

    if total_N == 0:
        return 0.0
    return float(np.exp(-total_log_lik / total_N))


def symptom_predictive_perplexity(
    X_ph: np.ndarray,
    X_ps: np.ndarray,
    H_h: np.ndarray,
    H_s: np.ndarray,
    eps: float = 1e-10,
    probabilize: bool = False,
    dirichlet_alpha: float = 0.1,
    infer_method: str = "nnls",
) -> float:
    """PTM 兼容: Symptom Predictive Perplexity (药材→症状).

    用药材侧信息推断主题分布, 然后预测症状:
      1. 通过 X_ph 和 H_h 用 NNLS 投影得到 G_val
      2. 用 G_val @ H_s^T 预测症状分布
      3. 在真实症状 X_ps 上算 perplexity

    与 PTM 的 symptom_pred_ppl 直接可比。

    当 probabilize=True 时:
      - H_s 列归一化为 p(symptom|topic)
      - NNLS 推断的 g 加 Dirichlet 先验后行归一化为 θ
    """
    from scipy.optimize import nnls
    P = X_ph.shape[0]

    # 因子概率化
    if probabilize:
        H_s_prob = H_s / (H_s.sum(axis=0, keepdims=True) + eps)

    total_log_lik = 0.0
    total_N = 0

    for i in range(P):
        if np.sum(X_ph[i]) == 0 or np.sum(X_ps[i]) == 0:
            continue

        from gsnmf.recommender import _solve_infer
        g_i = _solve_infer(X_ph[i], H_h, method=infer_method)

        # 兼容 H_h_flat (K*R维) 的情况: 将 g_i 重新聚合成 K 维，以便与 H_s 相乘
        K_s = H_s.shape[1]
        if len(g_i) > K_s and len(g_i) % K_s == 0:
            n_roles = len(g_i) // K_s
            g_i_sum = g_i.reshape(n_roles, K_s).sum(axis=0)
        else:
            g_i_sum = g_i

        if probabilize:
            g_i_sum = np.clip(g_i_sum, 0, None)
            g_smooth = g_i_sum + dirichlet_alpha
            theta = g_smooth / g_smooth.sum()
            p_hat = theta @ H_s_prob.T
            p_hat = np.clip(p_hat, eps, None)
        else:
            # 症状侧预测
            x_hat = g_i_sum @ H_s.T
            x_hat_pos = np.clip(x_hat, 0, None) + eps
            p_hat = x_hat_pos / x_hat_pos.sum()

        sym_idx = np.where(X_ps[i] > 0)[0]
        total_log_lik += np.log(p_hat[sym_idx]).sum()
        total_N += len(sym_idx)

    if total_N == 0:
        return 0.0
    return float(np.exp(-total_log_lik / total_N))


# ---- PTM 兼容: Precision@K / Recall@K / NDCG@K ----

def _predictive_precision_recall_ndcg(
    X_source: np.ndarray,
    X_target: np.ndarray,
    H_source: np.ndarray,
    H_target: np.ndarray,
    K: int = 10,
    eps: float = 1e-10,
    length_ratio: float | None = None,
    freq: np.ndarray | None = None,
    freq_tau: float = 0.5,
    infer_method: str = "nnls",
) -> dict:
    """PTM 兼容的 precision@K, recall@K, NDCG@K.

    用 source 侧推断 G, 预测 target 侧, 与真实比较。

    可选:
    - length_ratio: 处方长度校准 (topN 随样本长度变化)
    - freq: 训练集目标模态的频率 (用于稀疏重排)
    - freq_tau: 频率惩罚强度
    - infer_method: 'nnls', 'ols' (普通最小二乘), 'dot' (简单投影)
    """
    from scipy.optimize import nnls
    P = X_source.shape[0]

    prec_sum = 0.0
    rec_sum = 0.0
    ndcg_sum = 0.0
    count = 0
    
    # 提前缓存 OLS 的伪逆矩阵以加速
    H_source_pinv = None
    if infer_method == "ols":
        H_source_pinv = np.linalg.pinv(H_source)

    for i in range(P):
        if np.sum(X_source[i]) == 0 or np.sum(X_target[i]) == 0:
            continue

        if infer_method == "nnls":
            g_i, _ = nnls(H_source, X_source[i])
        elif infer_method == "ols":
            g_i = H_source_pinv @ X_source[i]  # type: ignore
        elif infer_method == "dot":
            g_i = X_source[i] @ H_source
        else:
            raise ValueError(f"Unknown infer_method: {infer_method}")
            
        x_hat = g_i @ H_target.T
        x_hat_pos = np.clip(x_hat, 0, None) + eps
        p_hat = x_hat_pos / x_hat_pos.sum()

        if freq is not None:
            p_hat = p_hat / (np.power(freq + eps, freq_tau))

        real_idx = set(np.where(X_target[i] > 0)[0].tolist())
        if len(real_idx) == 0:
            continue

        if length_ratio is None:
            k_i = K
        else:
            n_query = int(np.sum(X_source[i] > 0))
            k_i = int(round(length_ratio * n_query))
            k_i = int(np.clip(k_i, 1, K))

        top_k = np.argsort(p_hat)[::-1][:k_i]

        # Precision@K
        hits = sum(1 for idx in top_k if idx in real_idx)
        prec_sum += hits / k_i

        # Recall@K
        rec_sum += hits / len(real_idx)

        # NDCG@K
        dcg = 0.0
        for rank, idx in enumerate(top_k):
            if idx in real_idx:
                dcg += 1.0 / np.log2(rank + 2)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(k_i, len(real_idx))))
        ndcg_sum += dcg / max(idcg, 1e-10)

        count += 1

    n = max(count, 1)
    suffix = f"@{K}" if length_ratio is None else f"@{K}_cal"
    return {
        f"precision{suffix}": prec_sum / n,
        f"recall{suffix}": rec_sum / n,
        f"ndcg{suffix}": ndcg_sum / n,
    }


# =========================================================================
# 剂量指标 (仅在 M_pd=1 位置)
# =========================================================================

def dose_mae(X_pd: np.ndarray, X_hat: np.ndarray, M_pd: np.ndarray) -> float:
    mask = M_pd > 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(X_pd[mask] - X_hat[mask])))


def dose_rmse(X_pd: np.ndarray, X_hat: np.ndarray, M_pd: np.ndarray) -> float:
    mask = M_pd > 0
    if not mask.any():
        return 0.0
    return float(np.sqrt(np.mean((X_pd[mask] - X_hat[mask]) ** 2)))


def dose_smape(X_pd: np.ndarray, X_hat: np.ndarray, M_pd: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error。"""
    mask = M_pd > 0
    if not mask.any():
        return 0.0
    actual = X_pd[mask]
    predicted = X_hat[mask]
    denom = np.abs(actual) + np.abs(predicted)
    denom = np.maximum(denom, 1e-10)
    return float(np.mean(2.0 * np.abs(actual - predicted) / denom))


# =========================================================================
# 主题指标 (先基于药材主题 PMI)
# =========================================================================

def topic_coherence(
    H_h: np.ndarray,
    X_ph: np.ndarray,
    top_n: int = 10,
) -> float:
    """基于药材主题 top-n 的 PMI coherence。

    注意: 始终将 X_ph 二值化后计算，避免 TF-IDF 权重导致 PMI 膨胀。
    """
    K = H_h.shape[1]
    # 强制二值化，确保 PMI 计算正确
    X_bin = (np.asarray(X_ph) > 0).astype(np.float64)
    P = X_bin.shape[0]
    eps = 1e-10

    # 每个药材的出现频率 (基于二值)
    freq = X_bin.sum(axis=0).flatten() / P   # (H,)

    coherences = []
    for k in range(K):
        top_idx = np.argsort(H_h[:, k])[-top_n:][::-1]
        pairs_pmi = []
        for i in range(len(top_idx)):
            for j in range(i + 1, len(top_idx)):
                hi, hj = top_idx[i], top_idx[j]
                # 共现概率
                co_occur = np.sum((X_bin[:, hi] > 0) & (X_bin[:, hj] > 0)) / P
                pmi = np.log((co_occur + eps) / (freq[hi] * freq[hj] + eps))
                pairs_pmi.append(pmi)
        if pairs_pmi:
            coherences.append(float(np.mean(pairs_pmi)))

    return float(np.mean(coherences)) if coherences else 0.0


# =========================================================================
# 安全指标
# =========================================================================

def violation_at_k(
    y_scores: np.ndarray,
    C_hh: sp.csr_matrix,
    k: int,
) -> float:
    """推荐 top-k 中禁忌共现对比率。

    Violation@K = #{(i,j) ∈ ĤK × ĤK : C_ij=1} / C(K,2)
    """
    top_k = set(np.argsort(y_scores)[-k:][::-1].tolist())
    n_violations = count_violations(top_k, C_hh)
    n_pairs = k * (k - 1) / 2
    if n_pairs == 0:
        return 0.0
    return n_violations / n_pairs


def batch_violation_at_k(
    X_query: np.ndarray,
    H_query: np.ndarray,
    H_target: np.ndarray,
    C_hh: sp.csr_matrix,
    k: int = 10,
) -> float:
    """批量计算推荐的禁忌违规率。"""
    P = X_query.shape[0]
    violations = []
    for i in range(P):
        q = X_query[i]
        if np.sum(q) == 0:
            continue
        _, scores = symptoms_to_herbs(q, H_query, H_target)
        violations.append(violation_at_k(scores, C_hh, k))
    return float(np.mean(violations)) if violations else 0.0


def constraint_compatibility_rate(
    X_truth_herb: np.ndarray,
    X_query: np.ndarray,
    H_query: np.ndarray,
    H_target: np.ndarray,
    C_hh: sp.csr_matrix,
    k: int = 10,
) -> float:
    """约束兼容命中率。

    对真实处方本身无禁忌的处方，推荐 top-k 也无禁忌的比率。
    """
    from gsnmf.constraints import is_compatible

    P = X_truth_herb.shape[0]
    compatible_cases = 0
    compatible_recs = 0

    for i in range(P):
        truth = X_truth_herb[i]
        truth_herbs = set(np.where(truth > 0)[0].tolist())
        if len(truth_herbs) < 2:
            continue
        if not is_compatible(truth_herbs, C_hh):
            continue  # 真实处方本身就有禁忌，跳过

        compatible_cases += 1
        q = X_query[i]
        if np.sum(q) == 0:
            continue
        _, scores = symptoms_to_herbs(q, H_query, H_target)
        top_k = set(np.argsort(scores)[-k:][::-1].tolist())
        if is_compatible(top_k, C_hh):
            compatible_recs += 1

    if compatible_cases == 0:
        return 1.0
    return compatible_recs / compatible_cases


# =========================================================================
# 超图特有指标
# =========================================================================

def topic_group_compactness(
    H_h: np.ndarray,
    hyperedges: List,
    top_n: int = 10,
) -> float:
    """Topic Group Compactness (TGC).

    对每个主题 top-n 药材，看它们在超图中的平均共边率。
    TGC(k) = 2/(n(n-1)) * ∑_{i<j} 1{∃e: h_i, h_j ∈ e}

    Parameters
    ----------
    H_h : (H, K) 药材主题载荷
    hyperedges : 超边列表 (tuple of node indices)
    top_n : 每个主题取 top-n 药材

    Returns
    -------
    float : 平均 TGC
    """
    K = H_h.shape[1]
    if not hyperedges:
        return 0.0

    # 预计算: 每对药材是否共享超边 (存为 set 快速查找)
    edge_sets = [set(e) for e in hyperedges]

    # 构建节点对 → 超边索引
    pair_in_edge: set = set()
    for e_set in edge_sets:
        e_sorted = sorted(e_set)
        for i in range(len(e_sorted)):
            for j in range(i + 1, len(e_sorted)):
                pair_in_edge.add((e_sorted[i], e_sorted[j]))

    compactness_list = []
    for k in range(K):
        top_idx = np.argsort(H_h[:, k])[-top_n:][::-1]
        n = len(top_idx)
        if n < 2:
            continue
        n_pairs = n * (n - 1) / 2
        shared = 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b = min(top_idx[i], top_idx[j]), max(top_idx[i], top_idx[j])
                if (a, b) in pair_in_edge:
                    shared += 1
        compactness_list.append(shared / n_pairs)

    return float(np.mean(compactness_list)) if compactness_list else 0.0


def motif_hit_rate(
    H_h: np.ndarray,
    motif_edges: List,
    top_n: int = 10,
) -> float:
    """Motif Hit Rate.

    看模型输出的 top-n 药材中，有多少频繁 motif 被完整命中。

    Parameters
    ----------
    H_h : (H, K) 药材主题载荷
    motif_edges : motif 超边列表 (tuple of node indices)
    top_n : 每个主题取 top-n 药材

    Returns
    -------
    float : 平均 motif 命中率 (per topic, 命中 motif 数 / 总 motif 数)
    """
    K = H_h.shape[1]
    if not motif_edges:
        return 0.0

    motif_sets = [set(e) for e in motif_edges]
    n_motifs = len(motif_sets)

    hit_rates = []
    for k in range(K):
        top_set = set(np.argsort(H_h[:, k])[-top_n:][::-1].tolist())
        hits = sum(1 for m in motif_sets if m.issubset(top_set))
        hit_rates.append(hits / n_motifs)

    return float(np.mean(hit_rates)) if hit_rates else 0.0


# =========================================================================
# 汇总
# =========================================================================

def _to_dense(X) -> np.ndarray:
    if sp.issparse(X):
        return X.toarray().astype(np.float64)
    return np.asarray(X, dtype=np.float64)


def _infer_G_from_source(X_src: np.ndarray, H_src: np.ndarray) -> np.ndarray:
    """为新样本推断主题表示 G (逐行 NNLS)。

    对每一行 x_i 解: min_{z>=0} ‖x_i - z H_src^T‖²

    Parameters
    ----------
    X_src : (P_new, D_src) 新样本在 source 模态的观测
    H_src : (D_src, K)     source 模态载荷

    Returns
    -------
    G_new : (P_new, K)
    """
    from scipy.optimize import nnls
    P = X_src.shape[0]
    K = H_src.shape[1]
    G_new = np.zeros((P, K))
    for i in range(P):
        G_new[i], _ = nnls(H_src, X_src[i])
    return G_new


def evaluate_all(
    model,
    split_data,
    C_hh: sp.csr_matrix,
    ks: List[int] = [5, 10],
    compute_dose: bool = False,
    compute_coherence: bool = True,
    compute_perplexity: bool = True,
    eval_seed: int = 2025,
    hypergraph_bundle=None,
    dirichlet_alpha: float = 0.0,
    tfidf_decouple: bool = False,
    H_s_ppl: Optional[np.ndarray] = None,
    H_h_ppl: Optional[np.ndarray] = None,
    infer_method: str = "nnls",
) -> Dict[str, float]:
    """汇总所有指标 (在验证/测试集上)。

    NOTE: G_p 只覆盖训练集处方。验证/测试集的重构通过
    NNLS 投影得到 G_p_val，再计算 NRE 和剂量指标。

    Parameters
    ----------
    model : MVGSNMTF 实例
    split_data : SplitData
    C_hh : 禁忌矩阵
    hypergraph_bundle : HypergraphBundle, optional
        超图束，用于计算 TGC 和 Motif Hit Rate
    dirichlet_alpha : Dirichlet 先验平滑 (推荐 0.01)
    tfidf_decouple : PPL 计算中是否解除 TF-IDF 导致的频率缩放影响
    H_s_ppl : (S, K) EM 重估的概率化 H_s，用于 Sym PPL 计算
    H_h_ppl : (H, K) EM 重估的概率化 H_h，用于 Herb PPL 计算
    infer_method : 推断方法 ("nnls", "dot", "ols")
    """
    sl = split_data.valid
    X_ps = _to_dense(sl.X_ps)
    X_ph = _to_dense(sl.X_ph)
    X_pd = _to_dense(sl.X_pd)
    M_pd = _to_dense(sl.M_pd)

    # 因子始终转回 CPU numpy（评估用 scipy，需要 numpy）
    H_h = model.H_h.get() if hasattr(model.H_h, 'get') else np.asarray(model.H_h)
    H_s = model.H_s.get() if hasattr(model.H_s, 'get') else np.asarray(model.H_s)

    H_h_flat = None
    if getattr(model, 'H_h_roles', None) is not None:
        roles = model.H_h_roles.get() if hasattr(model.H_h_roles, 'get') else np.asarray(model.H_h_roles)
        H_h_flat = np.hstack([roles[r] for r in range(roles.shape[0])])
    
    H_h_eval = H_h_flat if H_h_flat is not None else H_h

    results = {}

    # 症状→药材推荐
    sym2herb = _batch_recommend_metrics(X_ps, X_ph, H_s, H_h, ks,
                                        dirichlet_alpha=dirichlet_alpha,
                                        infer_method=infer_method)
    for k, v in sym2herb.items():
        results[f"sym2herb_{k}"] = v
    results["valid_map_sym2herb"] = sym2herb["map"]

    # 药材→症状推荐
    herb2sym = _batch_recommend_metrics(X_ph, X_ps, H_h, H_s, ks,
                                        dirichlet_alpha=dirichlet_alpha,
                                        infer_method=infer_method)
    for k, v in herb2sym.items():
        results[f"herb2sym_{k}"] = v
    results["valid_map_herb2sym"] = herb2sym["map"]

    # 综合 MAP
    results["valid_map_avg"] = (results["valid_map_sym2herb"]
                                + results["valid_map_herb2sym"]) / 2

    # 为验证集推断主题表示 (按 source 模态逐样本 NNLS)
    G_h = _infer_G_from_source(X_ph, H_h)
    G_s = _infer_G_from_source(X_ps, H_s)

    # 重构 NRE
    X_ph_hat = G_h @ H_h.T
    results["nre_ph"] = nre(X_ph, X_ph_hat)

    X_ps_hat = G_s @ H_s.T
    results["nre_ps"] = nre(X_ps, X_ps_hat)

    # 跨模态推断重构 (更贴近推荐任务)
    X_ph_hat_from_ps = G_s @ H_h.T
    X_ps_hat_from_ph = G_h @ H_s.T
    results["nre_ph_cross"] = nre(X_ph, X_ph_hat_from_ps)
    results["nre_ps_cross"] = nre(X_ps, X_ps_hat_from_ph)

    # Perplexity
    if compute_perplexity:
        H_h_raw = H_h_eval
        H_s_raw = H_s
        
        if tfidf_decouple:
            logger_eval = logging.getLogger("evaluator")
            # 解耦 TF-IDF 在概率计算上的影响: 恢复 H 到真实频域空间
            def _get_idf(X):
                P_total = X.shape[0]
                df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float64)
                return np.log((1.0 + P_total) / (1.0 + df)) + 1.0

            idf_h = _get_idf(split_data.train.X_ph)
            idf_s = _get_idf(split_data.train.X_ps)
            
            H_h_raw = H_h_eval / idf_h[:, None]
            H_s_raw = H_s / idf_s[:, None]
            
        results["perplexity_ph"] = topic_perplexity(X_ph, X_ph_hat)
        results["perplexity_ps"] = topic_perplexity(X_ps, X_ps_hat)

        # Held-out perplexity (使用 H_h_eval 发挥多角色参数优势)
        results["ppl_masked_20"] = heldout_perplexity_masked(
            X_ph, H_h_eval, H_h_target=H_h_raw, mask_ratio=0.2, seed=eval_seed, infer_method=infer_method)
        results["ppl_half"] = heldout_perplexity_half(
            X_ph, H_h_eval, H_h_target=H_h_raw, seed=eval_seed, infer_method=infer_method)

        # 因子概率化 Perplexity
        results["ppl_masked_20_prob"] = heldout_perplexity_masked(
            X_ph, H_h_eval, H_h_target=H_h_raw, mask_ratio=0.2, seed=eval_seed,
            probabilize=True, dirichlet_alpha=dirichlet_alpha, infer_method=infer_method)
        results["ppl_half_prob"] = heldout_perplexity_half(
            X_ph, H_h_eval, H_h_target=H_h_raw, seed=eval_seed,
            probabilize=True, dirichlet_alpha=dirichlet_alpha, infer_method=infer_method)

        # 跨模态 perplexity (单视图模型无法计算)
        if model.sw.get("ps", True) and np.sum(X_ps) > 0:
            # 跨模态 perplexity (symptom -> herb 使用 H_h)
            results["ppl_crossmodal"] = crossmodal_perplexity(
                X_ps, X_ph, H_s, H_h_raw, infer_method=infer_method)

            results["herb_pred_ppl"] = results["ppl_crossmodal"]
            
            # 跨模态 perplexity (herb -> symptom 使用 H_h_eval 来发挥推断时的 R*K 自由度)
            results["symptom_pred_ppl"] = symptom_predictive_perplexity(
                X_ph, X_ps, H_h_eval, H_s_raw, infer_method=infer_method)

            # 因子概率化版本
            results["ppl_crossmodal_prob"] = crossmodal_perplexity(
                X_ps, X_ph, H_s, H_h_raw, probabilize=True, dirichlet_alpha=dirichlet_alpha, infer_method=infer_method)
            results["herb_pred_ppl_prob"] = results["ppl_crossmodal_prob"]

            # EM 重估的 H_h_ppl 用于 Herb PPL
            if H_h_ppl is not None:
                results["herb_pred_ppl_em"] = crossmodal_perplexity(
                    X_ps, X_ph, H_s, H_h_ppl, infer_method=infer_method)
                results["herb_pred_ppl_em_prob"] = crossmodal_perplexity(
                    X_ps, X_ph, H_s, H_h_ppl,
                    probabilize=True, dirichlet_alpha=dirichlet_alpha, infer_method=infer_method)
                logger.info(
                    "Herb PPL: raw=%.1f, prob=%.1f, EM=%.1f, EM_prob=%.1f",
                    results["herb_pred_ppl"],
                    results["herb_pred_ppl_prob"],
                    results["herb_pred_ppl_em"],
                    results["herb_pred_ppl_em_prob"],
                )
            
            results["symptom_pred_ppl_prob"] = symptom_predictive_perplexity(
                X_ph, X_ps, H_h_eval, H_s_raw, probabilize=True, dirichlet_alpha=dirichlet_alpha)

            # EM 重估的 H_s_ppl 用于 Sym PPL
            if H_s_ppl is not None:
                results["symptom_pred_ppl_em"] = symptom_predictive_perplexity(
                    X_ph, X_ps, H_h_eval, H_s_ppl, infer_method=infer_method)
                results["symptom_pred_ppl_em_prob"] = symptom_predictive_perplexity(
                    X_ph, X_ps, H_h_eval, H_s_ppl,
                    probabilize=True, dirichlet_alpha=dirichlet_alpha, infer_method=infer_method)
                logger.info(
                    "Sym PPL: raw=%.1f, prob=%.1f, EM=%.1f, EM_prob=%.1f",
                    results["symptom_pred_ppl"],
                    results["symptom_pred_ppl_prob"],
                    results["symptom_pred_ppl_em"],
                    results["symptom_pred_ppl_em_prob"],
                )

            # PTM 兼容 Precision@10 / Recall@10 / NDCG@10
            s2h = _predictive_precision_recall_ndcg(
                X_ps, X_ph, H_s, H_h, K=10, infer_method=infer_method)
            for k, v in s2h.items():
                results[f"herb_{k}"] = v

            h2s = _predictive_precision_recall_ndcg(
                X_ph, X_ps, H_h, H_s, K=10, infer_method=infer_method)
            for k, v in h2s.items():
                results[f"symptom_{k}"] = v

            # PTM 借鉴: 离散稀疏归一化风格重排 (不改模型，仅改评估排序)
            # 1) 处方长度校准: topN 按 query 长度自适应 (并限制到 <=10)
            # 2) 高频抑制: 用训练频率做轻度 downweight, 提升长尾辨识
            train_ph = _to_dense(split_data.train.X_ph)
            train_ps = _to_dense(split_data.train.X_ps)
            freq_h = train_ph.mean(axis=0)
            freq_s = train_ps.mean(axis=0)

            len_ratio_h = float(np.mean(np.sum(train_ph > 0, axis=1))
                                / max(np.mean(np.sum(train_ps > 0, axis=1)), 1e-10))
            len_ratio_s = float(np.mean(np.sum(train_ps > 0, axis=1))
                                / max(np.mean(np.sum(train_ph > 0, axis=1)), 1e-10))

            s2h_cal = _predictive_precision_recall_ndcg(
                X_ps, X_ph, H_s, H_h, K=10,
                length_ratio=len_ratio_h, freq=freq_h, freq_tau=0.5, infer_method=infer_method)
            for k, v in s2h_cal.items():
                results[f"herb_{k}"] = v

            h2s_cal = _predictive_precision_recall_ndcg(
                X_ph, X_ps, H_h, H_s, K=10,
                length_ratio=len_ratio_s, freq=freq_s, freq_tau=0.5, infer_method=infer_method)
            for k, v in h2s_cal.items():
                results[f"symptom_{k}"] = v

    # 剂量指标
    if compute_dose and model.sw.get("pd", False):
        D_h = model.D_h.get() if hasattr(model.D_h, 'get') else np.asarray(model.D_h)
        # 剂量重构沿用药材侧推断的 G_h
        X_pd_hat = G_h @ D_h.T
        results["dose_mae"] = dose_mae(X_pd, X_pd_hat, M_pd)
        results["dose_rmse"] = dose_rmse(X_pd, X_pd_hat, M_pd)
        results["dose_smape"] = dose_smape(X_pd, X_pd_hat, M_pd)

    # 主题 coherence (基于药材, 用训练集)
    if compute_coherence:
        X_ph_train = _to_dense(split_data.train.X_ph)
        results["topic_coherence"] = topic_coherence(H_h, X_ph_train)

    # 安全指标
    results["violation@10"] = batch_violation_at_k(
        X_ps, H_s, H_h, C_hh, k=10)
    results["compat_rate@10"] = constraint_compatibility_rate(
        X_ph, X_ps, H_s, H_h, C_hh, k=10)

    # 超图特有指标
    if hypergraph_bundle is not None:
        all_edges = []
        motif_only = []
        if hypergraph_bundle.stats.get("n_pres_edges", 0) > 0:
            # 处方超边已融入 L_total，但我们需要原始边来计算 TGC
            pass
        if hasattr(hypergraph_bundle, '_pres_edges'):
            all_edges.extend(hypergraph_bundle._pres_edges)
        if hasattr(hypergraph_bundle, '_motif_edges'):
            motif_only = list(hypergraph_bundle._motif_edges)
            all_edges.extend(motif_only)
        if hasattr(hypergraph_bundle, '_attr_edges'):
            all_edges.extend(hypergraph_bundle._attr_edges)

        if all_edges:
            results["tgc@10"] = topic_group_compactness(H_h, all_edges, top_n=10)
        if motif_only:
            results["motif_hit@10"] = motif_hit_rate(H_h, motif_only, top_n=10)

    return results
