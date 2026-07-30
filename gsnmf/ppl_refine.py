"""
PPL-Aware 因子精调：通过 EM 重估改善 Sym PPL。

核心思路：
  GSNMF 的 H_s 面向 Frobenius 重构 ‖X_ps - G·H_s^T‖² 优化,
  列归一化后不是良好的概率分布；
  PTM 的 φ̄ 是 Dirichlet-Multinomial 后验, 天然适配 PPL。

  本模块通过 EM 重估让 GSNMF 也能产出类 PTM 的概率分布:
  - E-step: 用 H_h 推断每个处方的主题分布 θ (NNLS + Dirichlet)
  - M-step: 用 θ 软分配症状→主题, 累加计数 + Dirichlet 平滑 → H_s_ppl

  产出的 H_s_ppl 仅用于 PPL 评估, 不影响推荐排序 (MAP/NDCG 沿用原 H_s)。
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import nnls

logger = logging.getLogger(__name__)


def reestimate_Hs_em(
    H_h: np.ndarray,
    X_ph_train: np.ndarray,
    X_ps_train: np.ndarray,
    n_iters: int = 10,
    beta_bar: float = 0.1,
    dirichlet_alpha: float = 0.01,
    H_s_init: Optional[np.ndarray] = None,
    infer_method: str = "nnls",
) -> np.ndarray:
    """EM 重估 H_s, 产出适配 PPL 的概率分布。

    模仿 PTM 的 φ̄[k,s] = (n_s[s,k] + β_bar) / (Σ_s n_s[s,k] + S·β_bar)
    但 E-step 用 GSNMF 的 H_h 做 NNLS 推断, 而非 Gibbs 采样。

    Parameters
    ----------
    H_h : (H, K) 药材主题载荷 (已训练好, 冻结)
    X_ph_train : (P, H) 训练集处方-药材 0/1
    X_ps_train : (P, S) 训练集处方-症状 0/1
    n_iters : EM 迭代次数
    beta_bar : Dirichlet 平滑 (对应 PTM 的 β̄, 推荐 0.1)
    dirichlet_alpha : NNLS 推断后的 Dirichlet 先验 (推荐 0.01)
    H_s_init : (S, K) 初始 H_s (可选, None 则从均匀开始)

    Returns
    -------
    H_s_ppl : (S, K) 概率化症状载荷, 每列 sum=1
    """
    P, H = X_ph_train.shape
    S = X_ps_train.shape[1]
    K = H_h.shape[1]
    eps = 1e-10

    # 初始化 H_s_ppl
    if H_s_init is not None:
        # 用当前因子列归一化作为起点
        col_sums = H_s_init.sum(axis=0, keepdims=True)
        col_sums = np.maximum(col_sums, eps)
        H_s_ppl = H_s_init / col_sums
    else:
        H_s_ppl = np.ones((S, K)) / S

    # 预计算: 每个处方的主题分布 θ (只依赖 H_h, 不随迭代变化)
    logger.info("EM 重估 H_s: 预计算 %d 个处方的主题分布...", P)
    thetas = np.zeros((P, K))
    valid_mask = np.zeros(P, dtype=bool)
    for i in range(P):
        if X_ph_train[i].sum() == 0 or X_ps_train[i].sum() == 0:
            continue
        from gsnmf.recommender import _solve_infer
        g_i = _solve_infer(X_ph_train[i], H_h, method=infer_method)
        g_i = np.clip(g_i, 0, None)
        g_smooth = g_i + dirichlet_alpha
        thetas[i] = g_smooth / g_smooth.sum()
        valid_mask[i] = True

    n_valid = valid_mask.sum()
    logger.info("有效处方: %d / %d", n_valid, P)

    # EM 迭代
    for em_it in range(n_iters):
        # 累加器
        n_s = np.zeros((S, K), dtype=np.float64)

        for i in range(P):
            if not valid_mask[i]:
                continue

            theta_i = thetas[i]  # (K,)
            sym_idx = np.where(X_ps_train[i] > 0)[0]

            for s in sym_idx:
                # E-step: 症状 s 的主题责任
                # r[k] ∝ θ[k] * H_s_ppl[s, k]
                r = theta_i * H_s_ppl[s, :]
                r_sum = r.sum()
                if r_sum > eps:
                    r = r / r_sum
                else:
                    r = theta_i  # 退化为均匀分配

                # 累加到充分统计量
                n_s[s, :] += r

        # M-step: Dirichlet-Multinomial 后验
        n_s_col_sum = n_s.sum(axis=0, keepdims=True)  # (1, K)
        H_s_ppl = (n_s + beta_bar) / (n_s_col_sum + S * beta_bar)

        # 日志
        if (em_it + 1) % max(1, n_iters // 5) == 0 or em_it == 0:
            # 检查覆盖度
            coverage = (n_s > 0.5).sum(axis=0).mean()
            entropy = -np.sum(H_s_ppl * np.log(H_s_ppl + eps)) / K
            logger.info(
                "  EM iter %d/%d: avg_sym_per_topic=%.1f, entropy=%.2f",
                em_it + 1, n_iters, coverage, entropy,
            )

    # 后验验证
    col_sums = H_s_ppl.sum(axis=0)
    logger.info(
        "H_s_ppl 完成: shape=%s, col_sum=[%.4f, %.4f], "
        "nnz_ratio=%.2f%%",
        H_s_ppl.shape,
        col_sums.min(), col_sums.max(),
        (H_s_ppl > eps).sum() / H_s_ppl.size * 100,
    )

    return H_s_ppl


def reestimate_Hs_em_vectorized(
    H_h: np.ndarray,
    X_ph_train: np.ndarray,
    X_ps_train: np.ndarray,
    n_iters: int = 10,
    beta_bar: float = 0.1,
    dirichlet_alpha: float = 0.01,
    H_s_init: Optional[np.ndarray] = None,
    infer_method: str = "nnls"
) -> np.ndarray:
    """向量化版 EM 重估 (更快, 适合大数据集)。

    跟 reestimate_Hs_em 逻辑完全一致, 但用矩阵运算代替 for 循环。

    返回 H_s_ppl: (S, K)
    """
    P, H = X_ph_train.shape
    S = X_ps_train.shape[1]
    K = H_h.shape[1]
    eps = 1e-10

    # 初始化
    if H_s_init is not None:
        col_sums = H_s_init.sum(axis=0, keepdims=True)
        H_s_ppl = H_s_init / np.maximum(col_sums, eps)
    else:
        H_s_ppl = np.ones((S, K)) / S

    # 批量 NNLS 推断 θ
    logger.info("EM 重估 (向量化): 批量 NNLS 推断 %d 个处方...", P)
    thetas = np.zeros((P, K), dtype=np.float64)
    valid_mask = np.zeros(P, dtype=bool)

    for i in range(P):
        x_h = X_ph_train[i]
        x_s = X_ps_train[i]
        if x_h.sum() == 0 or x_s.sum() == 0:
            continue
        from gsnmf.recommender import _solve_infer
        g_i = _solve_infer(x_h, H_h, method=infer_method)
        g_i = np.clip(g_i, 0, None)
        g_smooth = g_i + dirichlet_alpha
        thetas[i] = g_smooth / g_smooth.sum()
        valid_mask[i] = True

    # 只保留有效样本
    thetas_valid = thetas[valid_mask]    # (P_valid, K)
    X_ps_valid = X_ps_train[valid_mask]  # (P_valid, S)
    X_ps_bin = (X_ps_valid > 0).astype(np.float64)

    n_valid = thetas_valid.shape[0]
    logger.info("有效处方: %d / %d", n_valid, P)

    for em_it in range(n_iters):
        # E-step (向量化):
        # resp[p, s, k] = θ[p,k] * H_s_ppl[s,k] / Σ_k' θ[p,k'] * H_s_ppl[s,k']
        # 但 (P, S, K) 可能太大, 改为逐 topic 累加

        # 计算 P_hat[p, s] = Σ_k θ[p,k] * H_s_ppl[s,k] = thetas @ H_s_ppl.T
        P_hat = thetas_valid @ H_s_ppl.T  # (P_valid, S)
        P_hat = np.maximum(P_hat, eps)

        # n_s[s, k] = Σ_p X_ps[p,s] * θ[p,k] * H_s_ppl[s,k] / P_hat[p,s]
        # 重组为矩阵运算:
        # weight[p, s] = X_ps_bin[p,s] / P_hat[p,s]
        weight = X_ps_bin / P_hat  # (P_valid, S), 只在观测位置有值

        # n_s[s, k] = Σ_p weight[p,s] * θ[p,k] * H_s_ppl[s,k]
        #           = H_s_ppl[s,k] * Σ_p weight[p,s] * θ[p,k]
        #           = H_s_ppl[s,k] * (weight.T @ thetas_valid)[s, k]
        weighted_theta = weight.T @ thetas_valid  # (S, K)
        n_s = H_s_ppl * weighted_theta  # (S, K)

        # M-step
        n_s_col_sum = n_s.sum(axis=0, keepdims=True)
        H_s_ppl = (n_s + beta_bar) / (n_s_col_sum + S * beta_bar)

        if (em_it + 1) % max(1, n_iters // 5) == 0 or em_it == 0:
            coverage = (n_s > 0.5).sum(axis=0).mean()
            entropy = -np.sum(H_s_ppl * np.log(H_s_ppl + eps)) / K
            logger.info(
                "  EM iter %d/%d: avg_sym_per_topic=%.1f, entropy=%.2f",
                em_it + 1, n_iters, coverage, entropy,
            )

    col_sums = H_s_ppl.sum(axis=0)
    logger.info(
        "H_s_ppl 完成 (向量化): col_sum=[%.6f, %.6f]",
        col_sums.min(), col_sums.max(),
    )

    return H_s_ppl


def reestimate_Hh_em_vectorized(
    H_s: np.ndarray,
    X_ps_train: np.ndarray,
    X_ph_train: np.ndarray,
    n_iters: int = 10,
    beta: float = 0.1,
    dirichlet_alpha: float = 0.01,
    H_h_init: Optional[np.ndarray] = None,
    infer_method: str = "nnls",
) -> np.ndarray:
    """对称版 EM 重估 H_h, 产出适配 Herb PPL 的概率分布。

    与 reestimate_Hs_em_vectorized 完全对称:
    - E-step: 用 H_s 从症状推断 θ (NNLS + Dirichlet)
    - M-step: 用 θ 软分配药材→主题, 累加计数 + Dirichlet 平滑 → H_h_ppl

    模仿 PTM 的 φ_marg[k,h] = Σ_x φ[k,x,h]
    后归一化为 P(herb|topic)。

    Parameters
    ----------
    H_s : (S, K) 症状主题载荷 (已训练好, 冻结)
    X_ps_train : (P, S) 训练集处方-症状 0/1
    X_ph_train : (P, H) 训练集处方-药材 0/1
    n_iters : EM 迭代次数
    beta : Dirichlet 平滑 (对应 PTM 的 β, 推荐 0.1)
    dirichlet_alpha : NNLS 推断后的 Dirichlet 先验 (推荐 0.01)
    H_h_init : (H, K) 初始 H_h (可选)

    Returns
    -------
    H_h_ppl : (H, K) 概率化药材载荷, 每列 sum=1
    """
    P, S = X_ps_train.shape
    H = X_ph_train.shape[1]
    K = H_s.shape[1]
    eps = 1e-10

    # 初始化
    if H_h_init is not None:
        col_sums = H_h_init.sum(axis=0, keepdims=True)
        H_h_ppl = H_h_init / np.maximum(col_sums, eps)
    else:
        H_h_ppl = np.ones((H, K)) / H

    # 批量 NNLS: 从症状推断 θ
    logger.info("EM 重估 H_h (向量化): 批量 NNLS 推断 %d 个处方...", P)
    thetas = np.zeros((P, K), dtype=np.float64)
    valid_mask = np.zeros(P, dtype=bool)

    for i in range(P):
        x_s = X_ps_train[i]
        x_h = X_ph_train[i]
        if x_s.sum() == 0 or x_h.sum() == 0:
            continue
        from gsnmf.recommender import _solve_infer
        g_i = _solve_infer(x_s, H_s, method=infer_method)
        g_i = np.clip(g_i, 0, None)
        g_smooth = g_i + dirichlet_alpha
        thetas[i] = g_smooth / g_smooth.sum()
        valid_mask[i] = True

    thetas_valid = thetas[valid_mask]    # (P_valid, K)
    X_ph_valid = X_ph_train[valid_mask]  # (P_valid, H)
    X_ph_bin = (X_ph_valid > 0).astype(np.float64)

    n_valid = thetas_valid.shape[0]
    logger.info("有效处方: %d / %d", n_valid, P)

    for em_it in range(n_iters):
        # E-step: P_hat[p, h] = Σ_k θ[p,k] * H_h_ppl[h,k]
        P_hat = thetas_valid @ H_h_ppl.T  # (P_valid, H)
        P_hat = np.maximum(P_hat, eps)

        # weight[p, h] = X_ph_bin[p,h] / P_hat[p,h]
        weight = X_ph_bin / P_hat

        # n_h[h, k] = H_h_ppl[h,k] * (weight.T @ thetas_valid)[h, k]
        weighted_theta = weight.T @ thetas_valid  # (H, K)
        n_h = H_h_ppl * weighted_theta

        # M-step
        n_h_col_sum = n_h.sum(axis=0, keepdims=True)
        H_h_ppl = (n_h + beta) / (n_h_col_sum + H * beta)

        if (em_it + 1) % max(1, n_iters // 5) == 0 or em_it == 0:
            coverage = (n_h > 0.5).sum(axis=0).mean()
            entropy = -np.sum(H_h_ppl * np.log(H_h_ppl + eps)) / K
            logger.info(
                "  EM H_h iter %d/%d: avg_herb_per_topic=%.1f, entropy=%.2f",
                em_it + 1, n_iters, coverage, entropy,
            )

    col_sums = H_h_ppl.sum(axis=0)
    logger.info(
        "H_h_ppl 完成 (向量化): col_sum=[%.6f, %.6f]",
        col_sums.min(), col_sums.max(),
    )

    return H_h_ppl


def refine_Hs_crossentropy(
    H_h: np.ndarray,
    H_s: np.ndarray,
    X_ph_train: np.ndarray,
    X_ps_train: np.ndarray,
    n_iters: int = 200,
    lr: float = 0.05,
    dirichlet_alpha: float = 0.01,
    l2_reg: float = 0.0,
) -> np.ndarray:
    """梯度下降直接优化交叉熵 (Sym PPL 目标)。

    使用 log-space 参数化 + softmax 归一化, 保证 H_s_ppl 始终为合法概率。

    Loss = -Σ_p Σ_{s: obs} log( θ_p · H_s_prob[:, s] )

    Parameters
    ----------
    H_h, H_s : 已训练因子
    X_ph_train, X_ps_train : 训练数据
    n_iters : 梯度步数
    lr : 学习率
    dirichlet_alpha : NNLS Dirichlet 平滑
    l2_reg : L2 正则 (防过拟合)

    Returns
    -------
    H_s_ppl : (S, K) 概率化症状载荷
    """
    P, S = X_ps_train.shape
    K = H_s.shape[1]
    eps = 1e-10

    # 预计算 θ
    logger.info("CE 精调: 预计算 %d 个处方的主题分布...", P)
    thetas = np.zeros((P, K), dtype=np.float64)
    valid_mask = np.zeros(P, dtype=bool)
    for i in range(P):
        if X_ph_train[i].sum() == 0 or X_ps_train[i].sum() == 0:
            continue
        g_i, _ = nnls(H_h, X_ph_train[i])
        g_smooth = g_i + dirichlet_alpha
        thetas[i] = g_smooth / g_smooth.sum()
        valid_mask[i] = True

    thetas_valid = thetas[valid_mask]
    X_ps_valid = X_ps_train[valid_mask]
    X_ps_bin = (X_ps_valid > 0).astype(np.float64)
    n_valid = thetas_valid.shape[0]
    total_obs = int(X_ps_bin.sum())

    # log-space 参数化
    log_H = np.log(H_s + eps)

    logger.info("CE 精调: %d 有效处方, %d 观测症状", n_valid, total_obs)

    for it in range(n_iters):
        # softmax 列归一化
        log_H_shifted = log_H - log_H.max(axis=0, keepdims=True)
        exp_H = np.exp(log_H_shifted)
        col_sums = exp_H.sum(axis=0, keepdims=True)
        H_prob = exp_H / col_sums  # (S, K)

        # 前向: P_hat[p, s] = θ[p,:] @ H_prob[s,:].T = thetas @ H_prob.T
        P_hat = thetas_valid @ H_prob.T  # (n_valid, S)
        P_hat = np.maximum(P_hat, eps)

        # Loss
        log_P = np.log(P_hat)
        ce_loss = -float((X_ps_bin * log_P).sum()) / total_obs

        if it % max(1, n_iters // 10) == 0 or it == n_iters - 1:
            ppl = float(np.exp(ce_loss))
            logger.info("  CE iter %d/%d: loss=%.4f, train_sym_ppl=%.1f",
                         it, n_iters, ce_loss, ppl)

        # 梯度: d_loss / d_H_prob[s,k]
        # = -Σ_p X_ps_bin[p,s] * θ[p,k] / P_hat[p,s] / total_obs
        residual = X_ps_bin / P_hat  # (n_valid, S)
        dL_dH_prob = -(residual.T @ thetas_valid) / total_obs  # (S, K)

        # 链式法则: softmax 的 Jacobian
        # d_H_prob / d_log_H 需要 softmax backprop
        # 简化: 使用 H_prob * (dL - Σ_s dL*H_prob) trick
        # grad_log_H[s,k] = H_prob[s,k] * (dL_dH_prob[s,k] - Σ_s' dL_dH_prob[s',k]*H_prob[s',k])
        sum_term = (dL_dH_prob * H_prob).sum(axis=0, keepdims=True)
        grad_log_H = H_prob * (dL_dH_prob - sum_term)

        if l2_reg > 0:
            grad_log_H += l2_reg * log_H / total_obs

        # 梯度裁剪
        grad_norm = np.linalg.norm(grad_log_H)
        if grad_norm > 10.0:
            grad_log_H = grad_log_H * (10.0 / grad_norm)

        log_H -= lr * grad_log_H

    # 最终 softmax
    log_H_shifted = log_H - log_H.max(axis=0, keepdims=True)
    exp_H = np.exp(log_H_shifted)
    H_s_ppl = exp_H / exp_H.sum(axis=0, keepdims=True)

    return H_s_ppl
