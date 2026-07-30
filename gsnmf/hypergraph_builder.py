"""
三层药材超图构建 & 超图拉普拉斯。

三层超边:
  Level 1 — 处方原生超边: 每张处方的药材集合
  Level 2 — 高频配伍 motif 超边: 从处方库挖掘的高频 2/3 药组
  Level 3 — 属性一致性超边: 由药材属性相似度构造的功能组

组合方式:
  L_h^hyper = ω1·L_pres + ω2·L_motif + ω3·L_attr

归一化超图拉普拉斯:
  Θ = D_v^{-1/2} B W_e D_e^{-1} B^T D_v^{-1/2}
  L = I - Θ
"""
from __future__ import annotations

import logging
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from gsnmf.schemas import HypergraphBundle

logger = logging.getLogger(__name__)


# =========================================================================
# 基础设施: 关联矩阵 & 超图拉普拉斯
# =========================================================================

def build_incidence_matrix(
    edges: List[Tuple[int, ...]],
    edge_weights: np.ndarray,
    n_nodes: int,
) -> Tuple[sp.csr_matrix, sp.dia_matrix]:
    """构建稀疏关联矩阵 B (H × M) 和边权对角矩阵 W_e (M × M)。

    Parameters
    ----------
    edges : 超边列表, 每条超边为节点索引 tuple
    edge_weights : (M,) 每条超边的权重
    n_nodes : 节点总数 H

    Returns
    -------
    B : (H, M) 稀疏关联矩阵
    W_e : (M, M) 边权对角矩阵
    """
    M = len(edges)
    rows, cols = [], []
    for e_idx, edge in enumerate(edges):
        for node in edge:
            rows.append(node)
            cols.append(e_idx)

    data = np.ones(len(rows), dtype=np.float64)
    B = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, M)).tocsr()
    W_e = sp.diags(edge_weights.astype(np.float64))
    return B, W_e


def build_hypergraph_laplacian(
    B: sp.csr_matrix,
    W_e: sp.dia_matrix,
) -> sp.csr_matrix:
    """归一化超图拉普拉斯: L = I - D_v^{-1/2} B W_e D_e^{-1} B^T D_v^{-1/2}。

    Parameters
    ----------
    B : (H, M) 关联矩阵
    W_e : (M, M) 边权对角矩阵

    Returns
    -------
    L_hyper : (H, H) 归一化超图拉普拉斯
    """
    H, M = B.shape

    # 超边度: δ(e) = |e| = sum over v of B(v, e)
    d_e = np.asarray(B.sum(axis=0)).flatten()  # (M,)
    d_e = np.maximum(d_e, 1.0)  # 避免除零

    # 节点度: d(v) = sum_e w(e) B(v, e)
    w_diag = W_e.diagonal()
    d_v_raw = np.asarray(B.dot(sp.diags(w_diag)).sum(axis=1)).flatten()  # (H,)
    isolated = (d_v_raw == 0)  # 不属于任何超边的孤立节点
    d_v = np.maximum(d_v_raw, 1e-10)  # clamp 仅为了数值稳定

    # D_v^{-1/2}
    d_v_inv_sqrt = 1.0 / np.sqrt(d_v)
    D_v_inv_sqrt = sp.diags(d_v_inv_sqrt)

    # D_e^{-1}
    d_e_inv = 1.0 / d_e
    D_e_inv = sp.diags(d_e_inv)

    # Θ = D_v^{-1/2} B W_e D_e^{-1} B^T D_v^{-1/2}
    step1 = D_v_inv_sqrt @ B          # (H, M)
    step2 = step1 @ W_e               # (H, M)
    step3 = step2 @ D_e_inv           # (H, M)
    step4 = step3 @ B.T               # (H, H)
    Theta = step4 @ D_v_inv_sqrt      # (H, H)

    L = sp.eye(H) - Theta

    # 修复: 孤立节点的 L[v,v] = 1 是不期望的 L2 正则,
    # 将孤立节点的行/列置零, 只保留真正被超边覆盖的节点的平滑约束
    n_isolated = int(np.sum(isolated))
    if n_isolated > 0:
        L = L.tolil()
        iso_idx = np.where(isolated)[0]
        for i in iso_idx:
            L[i, :] = 0
            L[:, i] = 0
        L = L.tocsr()

    return L.tocsr()


# =========================================================================
# 直接超图正则 (不经过 Laplacian, 保留高阶结构)
# =========================================================================

def compute_hyper_direct_loss(
    H_h: np.ndarray,
    edges: List[Tuple[int, ...]],
    weights: np.ndarray,
    normalize: bool = True,
) -> float:
    """直接计算组内一致性 loss (不经过 Laplacian)。

    normalize=True  → 方差模式: Σ_e w(e)/|e| · Σ_{v∈e} ‖h_v - μ_e‖²
    normalize=False → 均值对齐: Σ_e w(e) · Σ_{v∈e} ‖h_v - μ_e‖²

    比 Laplacian 形式 (B·Bᵀ 退化为二阶) 能真正保持高阶组块约束。
    """
    loss = 0.0
    for e_idx, edge in enumerate(edges):
        nodes = list(edge)
        vecs = H_h[nodes, :]       # (|e|, K)
        mu = vecs.mean(axis=0)     # (K,)
        diff_sq = np.sum((vecs - mu) ** 2)
        w = weights[e_idx]
        if normalize:
            diff_sq /= len(edge)
        loss += w * diff_sq
    return float(loss)


def compute_hyper_direct_grad(
    H_h: np.ndarray,
    edges: List[Tuple[int, ...]],
    weights: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """直接计算组内一致性梯度。

    ∂Ω/∂H[u,k] = Σ_{e:u∈e} scale · (H[u,k] - μ[e,k])
    其中 scale = 2w/|e| (normalize) 或 2w (unnormalize)
    """
    grad = np.zeros_like(H_h)
    for e_idx, edge in enumerate(edges):
        nodes = list(edge)
        vecs = H_h[nodes, :]       # (|e|, K)
        mu = vecs.mean(axis=0)     # (K,)
        diff = vecs - mu           # (|e|, K)
        w = weights[e_idx]
        scale = 2.0 * w
        if normalize:
            scale /= len(edge)
        grad[nodes, :] += scale * diff
    return grad


# =========================================================================
# 类别超边: 从离散分类标签直接构造 (外部领域知识)
# =========================================================================

def build_category_hyperedges(
    C_cat: np.ndarray,
    min_size: int = 2,
) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
    """将药材分类 one-hot 的每一列直接转为一条超边。

    与 build_attribute_hyperedges (基于连续相似度) 不同,
    这里使用离散类别标签, 每条超边代表一个中医功效类别
    (如解表药、清热药、补气药), 是真正的外部领域知识。

    Parameters
    ----------
    C_cat : (H, n_cat) 类别 one-hot
    min_size : 每条超边至少包含的药材数

    Returns
    -------
    edges : 超边列表 (每条超边是药材索引元组)
    weights : 每条超边的权重 (均为 1.0)
    """
    H, n_cat = C_cat.shape
    edges = []
    for j in range(n_cat):
        members = tuple(np.where(C_cat[:, j] > 0)[0])
        if len(members) >= min_size:
            edges.append(members)

    weights = np.ones(len(edges), dtype=np.float64)
    logger.info("类别超边: %d 条 (来自 %d 个类别, min_size=%d)",
                len(edges), n_cat, min_size)
    return edges, weights


# =========================================================================
# Level 1: 处方原生超边
# =========================================================================

def build_prescription_hyperedges(
    X_ph: np.ndarray,
    min_size: int = 2,
    max_size: Optional[int] = None,
    length_penalty_rho: float = 0.5,
) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
    """从处方矩阵构建超边: 每张处方的药材集合为一条超边。

    Parameters
    ----------
    X_ph : (P, H) 处方-药材 0/1 矩阵 (稠密或稀疏)
    min_size : 最小超边大小
    max_size : 最大超边大小 (None=不限)
    length_penalty_rho : 权重 w = 1 / |e|^ρ, 抑制长处方主导

    Returns
    -------
    edges : 超边列表
    weights : (M,) 每条超边权重
    """
    if sp.issparse(X_ph):
        X_ph = X_ph.toarray()

    edges = []
    weights = []
    P = X_ph.shape[0]

    for p in range(P):
        herbs = tuple(np.where(X_ph[p] > 0)[0])
        if len(herbs) < min_size:
            continue
        if max_size is not None and len(herbs) > max_size:
            continue
        edges.append(herbs)
        weights.append(1.0 / (len(herbs) ** length_penalty_rho))

    logger.info("Level-1 处方超边: %d 条 (min_size=%d, rho=%.2f)",
                len(edges), min_size, length_penalty_rho)
    return edges, np.array(weights, dtype=np.float64)


# =========================================================================
# Level 2: 高频配伍 motif 超边
# =========================================================================

def _compute_pmi_matrix(X_ph: np.ndarray) -> np.ndarray:
    """计算药材对 PMI 矩阵 (H × H)。

    PMI(i,j) = log[ P(i,j) / (P(i) * P(j)) ]
    """
    P = X_ph.shape[0]
    eps = 1e-10

    # 单药频率
    freq = X_ph.sum(axis=0) / P  # (H,)

    # 共现频率 (用矩阵乘法加速)
    X_bin = (X_ph > 0).astype(np.float64)
    cooc = (X_bin.T @ X_bin) / P  # (H, H) 共现概率

    # PMI
    freq_outer = np.outer(freq, freq)
    freq_outer = np.maximum(freq_outer, eps)
    pmi = np.log((cooc + eps) / freq_outer)
    np.fill_diagonal(pmi, 0.0)

    return pmi


def mine_frequent_herb_sets(
    X_ph: np.ndarray,
    min_support: int = 20,
    max_size: int = 3,
) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], int]]:
    """挖掘高频药材组合 (枚举 + 计数, 不引入新依赖)。

    支持 2-药组和 3-药组。使用 Apriori 剪枝策略。

    Parameters
    ----------
    X_ph : (P, H) 处方-药材矩阵
    min_support : 最小支持度 (出现次数)
    max_size : 最大药组大小 (2 或 3)

    Returns
    -------
    itemsets : 频繁项集列表
    support_dict : {itemset: count}
    """
    if sp.issparse(X_ph):
        X_ph = X_ph.toarray()

    X_bin = (X_ph > 0).astype(np.int8)
    P, H = X_bin.shape

    # 1-项频率
    freq1 = X_bin.sum(axis=0)  # (H,)
    frequent_1 = set(np.where(freq1 >= min_support)[0])

    logger.info("频繁 1-项集: %d / %d (min_support=%d)",
                len(frequent_1), H, min_support)

    support_dict: Dict[Tuple[int, ...], int] = {}
    itemsets: List[Tuple[int, ...]] = []

    # 2-项组: 利用矩阵乘法一次性计算所有二元共现
    cooc_matrix = X_bin.T @ X_bin  # (H, H)

    freq_herbs = sorted(frequent_1)
    for i_idx in range(len(freq_herbs)):
        hi = freq_herbs[i_idx]
        for j_idx in range(i_idx + 1, len(freq_herbs)):
            hj = freq_herbs[j_idx]
            count = int(cooc_matrix[hi, hj])
            if count >= min_support:
                pair = (hi, hj)
                support_dict[pair] = count
                itemsets.append(pair)

    n_pairs = len(itemsets)
    logger.info("频繁 2-项集: %d", n_pairs)

    # 3-项组: 基于频繁 2-项集的 Apriori 剪枝
    if max_size >= 3 and n_pairs > 0:
        # 构建邻接: 哪些药材与其他药材组成了频繁对
        pair_set = set(itemsets)
        adj: Dict[int, set] = {}
        for hi, hj in itemsets:
            adj.setdefault(hi, set()).add(hj)
            adj.setdefault(hj, set()).add(hi)

        # 候选 3-项组: (hi, hj, hk) 当且仅当三个子对都频繁
        candidates_3 = set()
        for hi, hj in itemsets:
            common = adj.get(hi, set()) & adj.get(hj, set())
            for hk in common:
                if hk > hj:
                    # 检查三个子对
                    sub1 = (hi, hj)
                    sub2 = (min(hi, hk), max(hi, hk))
                    sub3 = (min(hj, hk), max(hj, hk))
                    if sub1 in pair_set and sub2 in pair_set and sub3 in pair_set:
                        candidates_3.add((hi, hj, hk))

        # 计算 3-项组支持度
        for triple in candidates_3:
            hi, hj, hk = triple
            count = int(np.sum(X_bin[:, hi] & X_bin[:, hj] & X_bin[:, hk]))
            if count >= min_support:
                support_dict[triple] = count
                itemsets.append(triple)

        logger.info("频繁 3-项集: %d (候选 %d)",
                    len(itemsets) - n_pairs, len(candidates_3))

    logger.info("Level-2 总频繁项集: %d", len(itemsets))
    return itemsets, support_dict


def score_and_filter_motifs(
    itemsets: List[Tuple[int, ...]],
    support_dict: Dict[Tuple[int, ...], int],
    pmi_matrix: np.ndarray,
    score_a: float = 1.0,
    score_b: float = 1.0,
    top_k: int = 500,
) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
    """对 motif 边评分并筛选 top_k。

    权重 w = log(1 + support) · max(avg_pairwise_PMI, 0)

    Parameters
    ----------
    itemsets : 频繁项集列表
    support_dict : {itemset: count}
    pmi_matrix : (H, H) PMI 矩阵
    score_a, score_b : 综合分数中 support 和 PMI 的权重
    top_k : 保留最多 top_k 条 motif 超边

    Returns
    -------
    filtered_edges : 筛选后的超边
    weights : (M,) 权重
    """
    scores = []
    for edge in itemsets:
        sup = support_dict.get(edge, 0)

        # 平均 pairwise PMI
        pairs = list(combinations(edge, 2))
        if pairs:
            avg_pmi = np.mean([pmi_matrix[hi, hj] for hi, hj in pairs])
        else:
            avg_pmi = 0.0

        # 权重公式
        w = np.log(1.0 + sup) * max(avg_pmi, 0.0)
        score = score_a * np.log(1.0 + sup) + score_b * max(avg_pmi, 0.0)
        scores.append((score, w, edge))

    # 按综合分数排序, 取 top_k
    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[:top_k]

    filtered_edges = [s[2] for s in top]
    weights = np.array([s[1] for s in top], dtype=np.float64)

    # 保底: 权重至少 > 0
    weights = np.maximum(weights, 1e-10)

    logger.info("Level-2 motif 超边: %d 条 (top_k=%d, 从 %d 中筛选)",
                len(filtered_edges), top_k, len(itemsets))
    return filtered_edges, weights


# =========================================================================
# Level 3: 属性一致性超边
# =========================================================================

def build_attribute_hyperedges(
    F_h: np.ndarray,
    knn_k: int = 10,
    group_sizes: List[int] = [2, 3],
    sim_threshold: float = 0.7,
    top_k: int = 200,
) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
    """基于药材属性相似度构造功能一致性超边。

    步骤:
    1. 计算属性空间余弦相似度
    2. 对每个药材取 KNN 邻居
    3. 从 KNN 邻域中构造 2/3 药小组
    4. 计算组内一致性分数 (平均余弦相似度)
    5. 筛选高一致性小组

    Parameters
    ----------
    F_h : (H, D) 药材属性矩阵
    knn_k : 每个药材的 KNN 邻居数
    group_sizes : 要构造的小组大小列表
    sim_threshold : 一致性分数阈值
    top_k : 最多保留 top_k 条属性超边

    Returns
    -------
    edges : 超边列表
    weights : (M,) 权重 (= 一致性分数)
    """
    H = F_h.shape[0]

    # 余弦相似度
    sim = cosine_similarity(F_h)  # (H, H)  [-1, 1]
    np.fill_diagonal(sim, 0.0)

    # KNN: 每个药材的 top-k 邻居
    neighbors: List[np.ndarray] = []
    for i in range(H):
        top_idx = np.argsort(sim[i])[-knn_k:]
        neighbors.append(top_idx)

    # 构造候选小组 + 计算一致性
    candidates: List[Tuple[float, Tuple[int, ...]]] = []

    for i in range(H):
        nb = neighbors[i]
        for gs in group_sizes:
            if gs == 2:
                for j in nb:
                    if j > i:  # 避免重复
                        score = float(sim[i, j])
                        if score >= sim_threshold:
                            candidates.append((score, (i, j)))
            elif gs == 3:
                # 从邻居中取 2 个组成 3 药组
                for j_idx in range(len(nb)):
                    for k_idx in range(j_idx + 1, len(nb)):
                        j, k = int(nb[j_idx]), int(nb[k_idx])
                        group = tuple(sorted([i, j, k]))
                        # 一致性 = 组内平均余弦
                        pairs = list(combinations(group, 2))
                        avg_sim = np.mean([sim[a, b] for a, b in pairs])
                        if avg_sim >= sim_threshold:
                            candidates.append((avg_sim, group))

    # 去重
    seen = set()
    unique = []
    for score, edge in candidates:
        if edge not in seen:
            seen.add(edge)
            unique.append((score, edge))

    # 按分数排序, 取 top_k
    unique.sort(key=lambda x: x[0], reverse=True)
    top = unique[:top_k]

    edges = [u[1] for u in top]
    weights = np.array([u[0] for u in top], dtype=np.float64)

    logger.info("Level-3 属性超边: %d 条 (knn_k=%d, threshold=%.2f, top_k=%d, 候选 %d)",
                len(edges), knn_k, sim_threshold, top_k, len(unique))
    return edges, weights


# =========================================================================
# 总入口
# =========================================================================

def build_herb_hypergraph(
    X_ph_train: np.ndarray,
    F_h: np.ndarray,
    H: int,
    cfg: Dict,
) -> HypergraphBundle:
    """构建药材超图, 返回 HypergraphBundle。

    Parameters
    ----------
    X_ph_train : (P_train, H) 训练集处方-药材矩阵
    F_h : (H, 51) 药材属性矩阵
    H : 药材总数
    cfg : 超图配置字典 (来自 YAML 的 hypergraph 节)

    Returns
    -------
    HypergraphBundle
    """
    if sp.issparse(X_ph_train):
        X_ph_dense = X_ph_train.toarray().astype(np.float64)
    else:
        X_ph_dense = np.asarray(X_ph_train, dtype=np.float64)

    omega = cfg.get("omega", [1.0, 1.0, 1.0])
    stats: Dict = {}

    L_pres = None
    L_motif = None
    L_attr = None
    pres_edges_raw: List = []
    motif_edges_raw: List = []
    attr_edges_raw: List = []
    pres_weights_raw: Optional[np.ndarray] = None
    motif_weights_raw: Optional[np.ndarray] = None
    attr_weights_raw: Optional[np.ndarray] = None

    # --- Level 1: 处方原生超边 ---
    if cfg.get("use_prescription_edges", True):
        pres_cfg = cfg.get("pres", {})
        edges, weights = build_prescription_hyperedges(
            X_ph_dense,
            min_size=pres_cfg.get("min_size", 2),
            max_size=pres_cfg.get("max_size", None),
            length_penalty_rho=pres_cfg.get("length_penalty_rho", 0.5),
        )
        if edges:
            B, W_e = build_incidence_matrix(edges, weights, H)
            L_pres = build_hypergraph_laplacian(B, W_e)
            stats["n_pres_edges"] = len(edges)
            pres_edges_raw = edges
            pres_weights_raw = weights
        else:
            stats["n_pres_edges"] = 0

    # --- Level 2: 高频配伍 motif 超边 ---
    if cfg.get("use_motif_edges", False):
        motif_cfg = cfg.get("motif", {})
        itemsets, support_dict = mine_frequent_herb_sets(
            X_ph_dense,
            min_support=motif_cfg.get("min_support", 20),
            max_size=motif_cfg.get("max_size", 3),
        )
        if itemsets:
            pmi_matrix = _compute_pmi_matrix(X_ph_dense)
            filtered, weights = score_and_filter_motifs(
                itemsets, support_dict, pmi_matrix,
                score_a=motif_cfg.get("score_a", 1.0),
                score_b=motif_cfg.get("score_b", 1.0),
                top_k=motif_cfg.get("top_k", 500),
            )
            if filtered:
                B, W_e = build_incidence_matrix(filtered, weights, H)
                L_motif = build_hypergraph_laplacian(B, W_e)
                stats["n_motif_edges"] = len(filtered)
                motif_edges_raw = filtered
                motif_weights_raw = weights
            else:
                stats["n_motif_edges"] = 0
        else:
            stats["n_motif_edges"] = 0

    # --- Level 3: 属性一致性超边 ---
    if cfg.get("use_attribute_edges", False):
        attr_cfg = cfg.get("attr", {})
        edges, weights = build_attribute_hyperedges(
            F_h,
            knn_k=attr_cfg.get("knn_k", 10),
            group_sizes=attr_cfg.get("group_sizes", [2, 3]),
            sim_threshold=attr_cfg.get("sim_threshold", 0.7),
            top_k=attr_cfg.get("top_k", 200),
        )
        if edges:
            B, W_e = build_incidence_matrix(edges, weights, H)
            L_attr = build_hypergraph_laplacian(B, W_e)
            stats["n_attr_edges"] = len(edges)
            attr_edges_raw = edges
            attr_weights_raw = weights
        else:
            stats["n_attr_edges"] = 0

    # --- 合并: L_total = ω1·L_pres + ω2·L_motif + ω3·L_attr ---
    components = []
    if L_pres is not None:
        components.append(omega[0] * L_pres)
    if L_motif is not None:
        components.append(omega[1] * L_motif)
    if L_attr is not None:
        components.append(omega[2] * L_attr)

    if components:
        L_total = components[0]
        for c in components[1:]:
            L_total = L_total + c
        L_total = L_total.tocsr()
    else:
        # 全部关闭时, 返回零矩阵
        L_total = sp.csr_matrix((H, H))
        logger.warning("所有超图层均关闭, L_total 为零矩阵")

    stats["omega"] = omega
    logger.info("药材超图构建完成: pres=%s, motif=%s, attr=%s",
                stats.get("n_pres_edges", "off"),
                stats.get("n_motif_edges", "off"),
                stats.get("n_attr_edges", "off"))

    return HypergraphBundle(
        L_pres=L_pres,
        L_motif=L_motif,
        L_attr=L_attr,
        L_total=L_total,
        stats=stats,
        _pres_edges=pres_edges_raw,
        _motif_edges=motif_edges_raw,
        _attr_edges=attr_edges_raw,
        _pres_weights=pres_weights_raw,
        _motif_weights=motif_weights_raw,
        _attr_weights=attr_weights_raw,
    )

