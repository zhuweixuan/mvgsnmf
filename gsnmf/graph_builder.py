"""
图邻接矩阵 & 拉普拉斯矩阵构建。

药材图:  A_h = η1·A_co + η2·A_cat + η3·A_feat  →  L_h
症状图:  A_s = ξ1·A_co + ξ2·A_loc + ξ3·A_eti   →  L_s

支持 unnormalized / symmetric (归一化) 拉普拉斯。
约束图 (C_neg) 从 constraints.py 获取，不混入 L_pos。
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from gsnmf.schemas import GraphPair

logger = logging.getLogger(__name__)


# =========================================================================
# 通用工具
# =========================================================================

def knn_sparsify(A: np.ndarray, k: int) -> sp.csr_matrix:
    """对相似度矩阵 A 做 KNN 截断，每行只保留 top-k 邻居。"""
    n = A.shape[0]
    # 对角线置 0（避免选自身）
    np.fill_diagonal(A, 0.0)
    mask = np.zeros_like(A)
    for i in range(n):
        top_k = np.argsort(A[i])[-k:]
        mask[i, top_k] = 1.0
    return sp.csr_matrix(A * mask)


def symmetrize(A: sp.spmatrix) -> sp.csr_matrix:
    """对称化: (A + A^T) / 2。"""
    return ((A + A.T) / 2.0).tocsr()


def compute_laplacian(
    A: sp.csr_matrix,
    mode: str = "unnormalized",
) -> sp.csr_matrix:
    """从邻接矩阵计算拉普拉斯。

    Parameters
    ----------
    A : (N, N) 非负对称邻接矩阵
    mode : "unnormalized" → L = D - A
           "symmetric"    → L_sym = I - D^{-1/2} A D^{-1/2}
    """
    A = A.tocsr()
    d = np.asarray(A.sum(axis=1)).flatten()

    if mode == "unnormalized":
        D = sp.diags(d)
        return (D - A).tocsr()

    elif mode == "symmetric":
        # D^{-1/2}，度为 0 的节点保持 0
        d_inv_sqrt = np.zeros_like(d)
        mask = d > 0
        d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        n = A.shape[0]
        L = sp.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
        return L.tocsr()

    else:
        raise ValueError(f"未知拉普拉斯模式: {mode}")


# =========================================================================
# 药材图组件
# =========================================================================

def build_cooc_adj(pairs: np.ndarray, n: int) -> sp.csr_matrix:
    """从共现对构建二值邻接矩阵 (n × n)。"""
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(len(rows), dtype=np.float64)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def build_category_adj(C: np.ndarray) -> sp.csr_matrix:
    """基于药材分类 one-hot 构建邻接: A = C @ C^T (同类边权=1)。

    Parameters
    ----------
    C : (H, 30) 类别 one-hot
    """
    A = C @ C.T  # (H, H)
    np.fill_diagonal(A, 0.0)
    # 二值化：只要同属至少一个类别，边权为 1
    A = (A > 0).astype(float)
    return sp.csr_matrix(A)


def build_feature_adj(F: np.ndarray, knn_k: int = 10) -> sp.csr_matrix:
    """基于属性向量余弦相似度 + KNN 截断构建邻接。"""
    sim = cosine_similarity(F)  # (n, n), [-1, 1]
    # 归一化到 [0, 1]
    sim = (sim + 1.0) / 2.0
    A_sparse = knn_sparsify(sim, knn_k)
    return symmetrize(A_sparse)


# =========================================================================
# 症状图组件
# =========================================================================

def build_loci_adj(loci: np.ndarray, knn_k: int = 10) -> sp.csr_matrix:
    """基于症状病位 one-hot 的 Jaccard 相似构建邻接。

    Parameters
    ----------
    loci : (S, 14) 病位 one-hot
    """
    # Jaccard: |A∩B| / |A∪B|
    intersection = loci @ loci.T
    row_sum = loci.sum(axis=1, keepdims=True)
    union = row_sum + row_sum.T - intersection
    # 避免除零
    union = np.maximum(union, 1e-10)
    sim = intersection / union
    A_sparse = knn_sparsify(sim, knn_k)
    return symmetrize(A_sparse)


def build_etiology_adj(
    cold_heat: np.ndarray,
    etiologies: np.ndarray,
    knn_k: int = 10,
) -> sp.csr_matrix:
    """基于病因+病性特征余弦相似 + KNN。

    Parameters
    ----------
    cold_heat : (S, 2) [寒热, 虚实]
    etiologies : (S, 15) 病因 one-hot
    """
    F = np.hstack([cold_heat, etiologies])  # (S, 17)
    return build_feature_adj(F, knn_k)


# =========================================================================
# 组合图
# =========================================================================

def build_herb_graph(
    herb_cooc: np.ndarray,
    F_h: np.ndarray,
    H: int,
    eta: List[float],
    knn_k: int = 10,
    laplacian_mode: str = "unnormalized",
    C_neg: Optional[sp.csr_matrix] = None,
) -> GraphPair:
    """构建药材图。

    A_h = η1·A_co + η2·A_cat + η3·A_feat → L_h

    Parameters
    ----------
    F_h : (H, 51) = [category_30 | nature_1 | toxicity_1 | tastes_7 | meridians_12]
    eta : [η1, η2, η3] 三种邻接的混合权重
    C_neg : 来自 constraints.py 的禁忌矩阵
    """
    C_cat = F_h[:, :30]     # 分类 one-hot
    A_feat = F_h[:, 30:]    # 性味归经 21 维

    A_co  = build_cooc_adj(herb_cooc, H)
    A_cat = build_category_adj(C_cat)
    A_f   = build_feature_adj(A_feat, knn_k)

    # 融合
    A = eta[0] * A_co + eta[1] * A_cat + eta[2] * A_f
    A = symmetrize(A)
    L = compute_laplacian(A, laplacian_mode)

    logger.info("药材图: nnz(A)=%d, laplacian=%s", A.nnz, laplacian_mode)
    return GraphPair(L_pos=L, C_neg=C_neg)


def build_symptom_graph(
    symptom_cooc: np.ndarray,
    F_s: np.ndarray,
    S: int,
    xi: List[float],
    knn_k: int = 10,
    laplacian_mode: str = "unnormalized",
) -> GraphPair:
    """构建症状图。

    A_s = ξ1·A_co + ξ2·A_loc + ξ3·A_eti → L_s

    Parameters
    ----------
    F_s : (S, 31) = [loci_14 | cold_heat_2 | etiologies_15]
    xi : [ξ1, ξ2, ξ3]
    """
    loci      = F_s[:, :14]
    cold_heat = F_s[:, 14:16]
    etiologies = F_s[:, 16:]

    A_co  = build_cooc_adj(symptom_cooc, S)
    A_loc = build_loci_adj(loci, knn_k)
    A_eti = build_etiology_adj(cold_heat, etiologies, knn_k)

    A = xi[0] * A_co + xi[1] * A_loc + xi[2] * A_eti
    A = symmetrize(A)
    L = compute_laplacian(A, laplacian_mode)

    logger.info("症状图: nnz(A)=%d, laplacian=%s", A.nnz, laplacian_mode)
    return GraphPair(L_pos=L, C_neg=None)
