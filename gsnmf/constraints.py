"""
十八反十九畏约束系统。

三层结构:
  1. 构建 C_hh 禁忌矩阵
  2. contra_penalty 训练惩罚项: ρ · tr(H_h^T C_hh H_h)
  3. 推理过滤/重排接口
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np
from scipy import sparse as sp


# ---------------------------------------------------------------------------
# 1. 禁忌矩阵构建
# ---------------------------------------------------------------------------

def build_contraindication_matrix(
    mutex_pairs: np.ndarray,
    H: int,
) -> sp.csr_matrix:
    """从互斥索引对构建对称 0/1 禁忌矩阵 C_hh (H × H)。

    NOTE:
    - 对角线始终为 0（自身不算禁忌）。
    - C_hh 是对称的: C_hh[i,j] == C_hh[j,i]。
    - 在 contra_penalty 中由于对称性会双计数 (i,j) 和 (j,i)，
      但只要所有实验保持一致即可，不影响相对比较。
    """
    rows = np.concatenate([mutex_pairs[:, 0], mutex_pairs[:, 1]])
    cols = np.concatenate([mutex_pairs[:, 1], mutex_pairs[:, 0]])
    data = np.ones(len(rows), dtype=np.float64)
    C = sp.coo_matrix((data, (rows, cols)), shape=(H, H))
    C = C.tocsr()
    # 确保对角线为 0
    C.setdiag(0)
    C.eliminate_zeros()
    return C


# ---------------------------------------------------------------------------
# 2. 训练惩罚
# ---------------------------------------------------------------------------

def contra_penalty(H_h: np.ndarray, C_hh: sp.csr_matrix) -> float:
    """计算 tr(H_h^T C_hh H_h)。

    直觉: 如果两个禁忌药材在多个主题里同时具有高权重，此值会变大。

    NOTE:
    - C_hh 对角线为 0，不含自惩罚。
    - C_hh 对称，(i,j)+(j,i) 双计数。
      只要所有实验一致，不影响模型选择。
    """
    # tr(H^T C H) = sum_ij C_ij * (H[i] · H[j])
    # 等价于 sum( (C @ H) * H )
    CH = C_hh.dot(H_h)       # (H, K)
    return float(np.sum(CH * H_h))


def contra_gradient(H_h: np.ndarray, C_hh: sp.csr_matrix) -> np.ndarray:
    """∇_{H_h} tr(H_h^T C_hh H_h) = 2 C_hh H_h（因为 C 对称）。"""
    return 2.0 * C_hh.dot(H_h)


# ---------------------------------------------------------------------------
# 3. 推理约束
# ---------------------------------------------------------------------------

def filter_contraindicated(
    scores: np.ndarray,
    selected_herbs: Set[int],
    C_hh: sp.csr_matrix,
    mode: str = "hard",
    penalty_weight: float = 0.5,
) -> np.ndarray:
    """对推荐分数施加禁忌约束。

    Parameters
    ----------
    scores : (H,) 每个药材的推荐分数
    selected_herbs : 已选定的药材索引集合
    C_hh : (H, H) 禁忌矩阵
    mode : "hard" | "soft"
        hard: 直接将禁忌候选分数置为 -inf
        soft: scores'(h) = scores(h) - penalty_weight * penalty(h)
    penalty_weight : float
        soft 模式的惩罚系数

    Returns
    -------
    adjusted_scores : (H,) 调整后的分数
    """
    adjusted = scores.copy()

    if not selected_herbs:
        return adjusted

    # 找出与已选药材构成禁忌的候选
    for h in selected_herbs:
        # C_hh[h, :] 中非零位置就是 h 的禁忌药材
        contra_indices = C_hh[h].nonzero()[1]
        if mode == "hard":
            adjusted[contra_indices] = -np.inf
        elif mode == "soft":
            adjusted[contra_indices] -= penalty_weight
        else:
            raise ValueError(f"未知约束模式: {mode}")

    return adjusted


def is_compatible(herb_set: Set[int], C_hh: sp.csr_matrix) -> bool:
    """检查药材集合是否无禁忌冲突。"""
    herbs = sorted(herb_set)
    for i, h1 in enumerate(herbs):
        for h2 in herbs[i+1:]:
            if C_hh[h1, h2] != 0:
                return False
    return True


def count_violations(herb_set: Set[int], C_hh: sp.csr_matrix) -> int:
    """计算药材集合中的禁忌对数量。"""
    herbs = sorted(herb_set)
    count = 0
    for i, h1 in enumerate(herbs):
        for h2 in herbs[i+1:]:
            if C_hh[h1, h2] != 0:
                count += 1
    return count
