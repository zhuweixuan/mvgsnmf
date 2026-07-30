"""
推荐推断 + 约束重排。

- 症状 → 药材 (NNLS)
- 药材 → 症状 (NNLS)
- 剂量估计
- 约束重排 / 贪心组方
- 主题解释
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import nnls
from scipy import sparse as sp

from gsnmf.constraints import filter_contraindicated, is_compatible
from gsnmf.schemas import ModelFactors


# =========================================================================
# 基础推荐
# =========================================================================

def _solve_infer(q: np.ndarray, H: np.ndarray, method: str = "nnls") -> np.ndarray:
    """推断表示: min_{z>=0} ‖q - z H^T‖² 或简单投影。

    Parameters
    ----------
    q : (D,) 输入向量
    H : (D, K) 载荷矩阵
    method : 推断方法 ("nnls", "dot", "ols")
    """
    if method == "nnls":
        z, _ = nnls(H, q)
    elif method == "dot":
        z = q @ H
    elif method == "ols":
        z = np.linalg.pinv(H) @ q
    else:
        raise ValueError(f"Unknown infer method: {method}")
    return z

def symptoms_to_herbs(
    q_s: np.ndarray,
    H_s: np.ndarray,
    H_h: np.ndarray,
    dirichlet_alpha: float = 0.0,
    infer_method: str = "nnls",
) -> Tuple[np.ndarray, np.ndarray]:
    """症状 → 药材推荐。

    Parameters
    ----------
    q_s : (S,) 症状输入向量 (0/1)
    H_s : (S, K) 症状载荷
    H_h : (H, K) 药材载荷
    dirichlet_alpha : Dirichlet 先验平滑强度, >0 启用

    Returns
    -------
    z : (K,) 推断的主题表示
    scores : (H,) 每个药材的推荐分数
    """
    z = _solve_infer(q_s, H_s, method=infer_method)
    if dirichlet_alpha > 0:
        z = z + dirichlet_alpha
        z = z / z.sum()
    scores = z @ H_h.T    # (H,)
    return z, scores


def symptoms_to_doses(
    z: np.ndarray,
    D_h: np.ndarray,
) -> np.ndarray:
    """已知主题表示 z → 剂量估计。

    Parameters
    ----------
    z : (K,) 主题表示 (来自 symptoms_to_herbs 的输出)
    D_h : (H, K) 剂量载荷

    Returns
    -------
    dose_scores : (H,) 剂量偏好分数
    """
    return z @ D_h.T


def herbs_to_symptoms(
    q_h: np.ndarray,
    H_h: np.ndarray,
    H_s: np.ndarray,
    dirichlet_alpha: float = 0.0,
    infer_method: str = "nnls",
) -> Tuple[np.ndarray, np.ndarray]:
    """药材 → 症状推荐。"""
    z = _solve_infer(q_h, H_h, method=infer_method)
    if dirichlet_alpha > 0:
        z = z + dirichlet_alpha
        z = z / z.sum()
    scores = z @ H_s.T
    return z, scores


# =========================================================================
# 约束重排
# =========================================================================

def rerank_with_constraints(
    scores: np.ndarray,
    selected_herbs: Set[int],
    C_hh: sp.csr_matrix,
    mode: str = "soft",
    penalty_weight: float = 0.5,
) -> np.ndarray:
    """对推荐分数施加禁忌约束重排。

    内部调用 constraints.filter_contraindicated。
    """
    return filter_contraindicated(scores, selected_herbs, C_hh,
                                  mode=mode, penalty_weight=penalty_weight)


def select_formula_greedy(
    scores: np.ndarray,
    C_hh: sp.csr_matrix,
    max_herbs: int = 10,
    mode: str = "hard",
) -> List[int]:
    """贪心组方: 逐步选取不违规的最高分药材。

    Parameters
    ----------
    scores : (H,) 初始推荐分数
    C_hh : (H, H) 禁忌矩阵
    max_herbs : 最大药材数
    mode : "hard" | "soft"

    Returns
    -------
    selected : 选中的药材索引列表
    """
    selected: List[int] = []
    remaining = scores.copy()

    for _ in range(max_herbs):
        # 禁忌过滤
        adj_scores = filter_contraindicated(
            remaining, set(selected), C_hh, mode=mode,
        )
        best = int(np.argmax(adj_scores))
        if adj_scores[best] <= 0:
            break  # 没有正分的候选了
        selected.append(best)
        remaining[best] = -np.inf  # 已选过

    return selected


# =========================================================================
# 主题解释
# =========================================================================

def explain_topic(
    k: int,
    factors: ModelFactors,
    herb_names: List[str],
    symptom_names: List[str],
    top_n: int = 10,
) -> Dict:
    """解释第 k 个主题。

    Returns
    -------
    dict with keys: top_herbs, top_symptoms, top_dose_herbs, representative_prescriptions
    """
    # Top 药材
    hh_col = factors.H_h[:, k]
    top_herb_idx = np.argsort(hh_col)[-top_n:][::-1]
    top_herbs = [(herb_names[i], float(hh_col[i])) for i in top_herb_idx]

    # Top 症状
    hs_col = factors.H_s[:, k]
    top_sym_idx = np.argsort(hs_col)[-top_n:][::-1]
    top_symptoms = [(symptom_names[i], float(hs_col[i])) for i in top_sym_idx]

    # Top 剂量敏感药材
    dh_col = factors.D_h[:, k]
    top_dose_idx = np.argsort(dh_col)[-top_n:][::-1]
    top_dose_herbs = [(herb_names[i], float(dh_col[i])) for i in top_dose_idx]

    # 代表处方 (G_p[:, k] 最高的)
    gp_col = factors.G_p[:, k]
    top_rx_idx = np.argsort(gp_col)[-top_n:][::-1]
    representative = [(int(i), float(gp_col[i])) for i in top_rx_idx]

    return {
        "topic_id": k,
        "top_herbs": top_herbs,
        "top_symptoms": top_symptoms,
        "top_dose_herbs": top_dose_herbs,
        "representative_prescriptions": representative,
    }
