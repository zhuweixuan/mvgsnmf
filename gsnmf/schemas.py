"""
统一数据结构定义。

避免模块之间传 dict，所有中间数据通过这些 schema 传递。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse as sp


# ---------------------------------------------------------------------------
# 原始数据
# ---------------------------------------------------------------------------

@dataclass
class AllData:
    """从 data_loader.load_all() 返回的完整数据集。"""

    # 主视图矩阵
    X_ph: sp.csr_matrix           # (P, H) 处方-药材存在性 0/1
    X_ps: sp.csr_matrix           # (P, S) 处方-症状存在性 0/1
    X_pd: sp.csr_matrix           # (P, H) 处方-药材剂量 (克)
    M_pd: sp.csr_matrix           # (P, H) 剂量可信 mask 0/1

    # 属性矩阵
    F_h: np.ndarray               # (H, 51) = [category_30 || nature_1 || toxicity_1 || tastes_7 || meridians_12]
    F_s: np.ndarray               # (S, 31) = [loci_14 || cold_heat_2 || etiologies_15]

    # 共现对
    herb_cooc: np.ndarray         # (N_hc, 2) 药材共现对索引
    symptom_cooc: np.ndarray      # (N_sc, 2) 症状共现对索引

    # PTM 可移植增强信息
    K_hs: np.ndarray              # (H, S) herb-symptom knowledge 矩阵
    pair_index: np.ndarray        # (N_pair, 2) pair_id -> (h_i, h_j)
    X_pair: sp.csr_matrix         # (P, N_pair) 处方-药对共现矩阵

    # 禁忌对
    herb_mutex: np.ndarray        # (N_hm, 2) 十八反十九畏禁忌对索引

    # 标识
    herb_ids: np.ndarray          # (H,) 稳定数值 ID
    herb_names: List[str]         # (H,) 药材名称（展示用）
    symptom_ids: np.ndarray       # (S,) 稳定数值 ID
    symptom_names: List[str]      # (S,) 症状名称（展示用）
    prescription_ids: np.ndarray  # (P,) 处方 ID


# ---------------------------------------------------------------------------
# 切分后数据
# ---------------------------------------------------------------------------

@dataclass
class SharedMeta:
    """train/valid/test 共享的元数据（不随切分改变）。"""
    F_h: np.ndarray
    F_s: np.ndarray
    herb_cooc: np.ndarray
    symptom_cooc: np.ndarray
    K_hs: np.ndarray
    pair_index: np.ndarray
    herb_mutex: np.ndarray
    herb_ids: np.ndarray
    herb_names: List[str]
    symptom_ids: np.ndarray
    symptom_names: List[str]


@dataclass
class SplitSlice:
    """一个切片中处方级别的数据。"""
    X_ph: sp.csr_matrix           # (P_split, H)
    X_ps: sp.csr_matrix           # (P_split, S)
    X_pd: sp.csr_matrix           # (P_split, H)
    M_pd: sp.csr_matrix           # (P_split, H)
    X_pair: sp.csr_matrix         # (P_split, N_pair)
    prescription_ids: np.ndarray  # (P_split,)


@dataclass
class SplitData:
    """切分后的完整数据包（共享元数据 + 三个切片）。"""
    meta: SharedMeta
    train: SplitSlice
    valid: SplitSlice
    test: SplitSlice


# ---------------------------------------------------------------------------
# 图结构
# ---------------------------------------------------------------------------

@dataclass
class GraphPair:
    """正图拉普拉斯 + 可选的约束负图。"""
    L_pos: sp.csr_matrix          # 正图拉普拉斯 (相似/共现)
    C_neg: Optional[sp.csr_matrix] = None  # 约束负图 (禁忌)


@dataclass
class HypergraphBundle:
    """三层药材超图拉普拉斯。"""
    L_pres: Optional[sp.csr_matrix]   # Level-1 处方原生超图拉普拉斯
    L_motif: Optional[sp.csr_matrix]  # Level-2 高频配伍 motif 超图拉普拉斯
    L_attr: Optional[sp.csr_matrix]   # Level-3 属性一致性超图拉普拉斯
    L_total: sp.csr_matrix            # 加权合并后的总超图拉普拉斯
    stats: Dict                       # 统计信息 (边数, omega 等)
    # 原始超边列表 (用于评估指标 TGC / Motif Hit Rate 及直接正则)
    _pres_edges: List = field(default_factory=list)
    _motif_edges: List = field(default_factory=list)
    _attr_edges: List = field(default_factory=list)
    _pres_weights: Optional[np.ndarray] = None
    _motif_weights: Optional[np.ndarray] = None
    _attr_weights: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# 模型因子
# ---------------------------------------------------------------------------

@dataclass
class ModelFactors:
    """模型学到的隐因子。"""
    G_p: np.ndarray               # (P, K)
    H_h: np.ndarray               # (H, K)
    H_s: np.ndarray               # (S, K)
    D_h: np.ndarray               # (H, K)
    H_pair: Optional[np.ndarray] = None   # (N_pair, K)
    G_p_roles: Optional[np.ndarray] = None # (R, P, K)
    H_h_roles: Optional[np.ndarray] = None # (R, H, K)
    K: int = 0

    def __post_init__(self):
        self.K = self.G_p.shape[1]


# ---------------------------------------------------------------------------
# Loss 分量
# ---------------------------------------------------------------------------

@dataclass
class LossComponents:
    """目标函数的各项分量（用于日志记录与诊断）。"""
    loss_ph: float = 0.0
    loss_ps: float = 0.0
    loss_pd: float = 0.0
    loss_pair: float = 0.0
    loss_graph_h: float = 0.0
    loss_graph_s: float = 0.0
    loss_hyper_h: float = 0.0
    loss_hyper_var: float = 0.0
    loss_hyper_mean: float = 0.0
    loss_l1_g: float = 0.0
    loss_l1_h: float = 0.0
    loss_l1_s: float = 0.0
    loss_l1_d: float = 0.0
    loss_contra: float = 0.0
    loss_know_hs: float = 0.0
    total: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "loss_ph": self.loss_ph,
            "loss_ps": self.loss_ps,
            "loss_pd": self.loss_pd,
            "loss_pair": self.loss_pair,
            "graph_h": self.loss_graph_h,
            "graph_s": self.loss_graph_s,
            "hyper_h": self.loss_hyper_h,
            "hyper_var": self.loss_hyper_var,
            "hyper_mean": self.loss_hyper_mean,
            "l1_g": self.loss_l1_g,
            "l1_h": self.loss_l1_h,
            "l1_s": self.loss_l1_s,
            "l1_d": self.loss_l1_d,
            "contra": self.loss_contra,
            "know_hs": self.loss_know_hs,
            "total": self.total,
        }
