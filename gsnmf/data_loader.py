"""
数据加载与对齐校验。

从配置解析的路径中加载所有 CSV 数据，拼接属性矩阵，并严格校验
药材/症状列顺序在各文件间的一致性。返回统一的 AllData schema。
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from scipy import sparse as sp

from gsnmf.paths import DatasetPaths, resolve_paths
from gsnmf.schemas import AllData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> pd.DataFrame:
    logger.info("加载 %s", path)
    return pd.read_csv(path)


def _require(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# 单文件加载
# ---------------------------------------------------------------------------

def _load_presence(path: str, id_col: str = "prescription_id") -> Tuple[np.ndarray, List[str], sp.csr_matrix]:
    """加载存在性矩阵 → (ids, names, sparse 0/1)。"""
    df = _read_csv(path)
    # 如果没有 id_col，以行号为 ID
    if id_col in df.columns:
        ids = df[id_col].to_numpy()
        names = [c for c in df.columns if c != id_col]
    else:
        ids = np.arange(len(df))
        names = list(df.columns)
    X = df[names].to_numpy()
    _require(np.isin(X, [0, 1]).all(), f"非 0/1 值于存在性矩阵: {path}")
    return ids, names, sp.csr_matrix(X, dtype=np.int8)


def _load_dosage(path: str, id_col: str = "prescription_id") -> Tuple[np.ndarray, List[str], sp.csr_matrix]:
    """加载剂量矩阵 → (ids, names, sparse float)。"""
    df = _read_csv(path)
    if id_col in df.columns:
        ids = df[id_col].to_numpy()
        names = [c for c in df.columns if c != id_col]
    else:
        ids = np.arange(len(df))
        names = list(df.columns)
    X = df[names].to_numpy(dtype=float)
    _require((X >= 0).all(), f"剂量矩阵存在负数: {path}")
    return ids, names, sp.csr_matrix(X)


def _load_herb_category(path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """药材分类 one-hot (793 × 30) → (ids, names, C)。"""
    df = _read_csv(path)
    _require("herb_index" in df.columns and "herb_name" in df.columns,
             "herb_category_onehot 缺少 herb_index / herb_name 列")
    ids = df["herb_index"].to_numpy()
    names = df["herb_name"].tolist()
    cat_cols = [c for c in df.columns if c not in ("herb_index", "herb_name")]
    C = df[cat_cols].to_numpy(dtype=float)
    return ids, names, C


def _load_herb_features(path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """药材性味归经 (793 × 21) → (ids, names, feat)。"""
    df = _read_csv(path)
    _require("herb_index" in df.columns and "herb_name" in df.columns,
             "herb_feature_matrix 缺少 herb_index / herb_name")
    ids = df["herb_index"].to_numpy()
    names = df["herb_name"].tolist()
    nature = df["nature"].to_numpy(dtype=float).reshape(-1, 1)      # (H,1)
    toxicity = df["toxicity"].to_numpy(dtype=float).reshape(-1, 1)   # (H,1)
    taste_cols = ["辛", "甘", "酸", "苦", "咸", "涩", "淡"]
    meridian_cols = ["肺", "大肠", "胃", "脾", "心", "小肠",
                     "膀胱", "肾", "心包", "三焦", "胆", "肝"]
    _require(all(c in df.columns for c in taste_cols + meridian_cols),
             "herb_feature_matrix 味/归经列缺失")
    tastes = df[taste_cols].to_numpy(dtype=float)                    # (H,7)
    meridians = df[meridian_cols].to_numpy(dtype=float)              # (H,12)
    feat = np.hstack([nature, toxicity, tastes, meridians])          # (H,21)
    return ids, names, feat


def _load_symptom_features(path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """症状特征 (390 × 31) → (ids, names, feat)。"""
    df = _read_csv(path)
    _require("symptom_id" in df.columns and "symptom_name" in df.columns,
             "symptom_feature_matrix 缺少 symptom_id / symptom_name")
    ids = df["symptom_id"].to_numpy()
    names = df["symptom_name"].tolist()
    loci_cols = ['肺', '大肠', '胃', '脾', '心', '小肠',
                 '膀胱', '肾', '心包', '三焦', '胆', '肝', '肌表', '胞宫']
    cold_heat_cols = ["寒热", "虚实"]
    etio_cols = ['风', '寒', '暑', '湿', '燥', '火',
                 '毒', '痰', '瘀', '食积', '气滞', '气虚', '血虚', '阴虚', '阳虚']
    _require(all(c in df.columns for c in loci_cols + cold_heat_cols + etio_cols),
             "symptom_feature_matrix 列缺失")
    loci = df[loci_cols].to_numpy(dtype=float)              # (S,14)
    cold_heat = df[cold_heat_cols].to_numpy(dtype=float)    # (S,2)
    etiologies = df[etio_cols].to_numpy(dtype=float)        # (S,15)
    feat = np.hstack([loci, cold_heat, etiologies])         # (S,31)
    return ids, names, feat


def _load_pairs(path: str) -> np.ndarray:
    """加载 N×2 索引对矩阵。"""
    df = _read_csv(path)
    arr = df.iloc[:, :2].to_numpy(dtype=int)
    _require(arr.ndim == 2 and arr.shape[1] == 2, f"索引对格式错误: {path}")
    _require((arr >= 0).all(), f"索引对包含负数: {path}")
    return arr


def _load_symptom_herb_knowledge_indices(path: str, H: int, S: int) -> np.ndarray:
    """加载 symptom-herb 知识对 (索引版) → K_hs (H, S)。

    约定文件为两列整数索引: [symptom_idx, herb_idx] 或 [herb_idx, symptom_idx]。
    自动根据越界情况判断列顺序。
    """
    df = _read_csv(path)
    arr = df.iloc[:, :2].to_numpy(dtype=int)
    _require(arr.ndim == 2 and arr.shape[1] == 2, f"知识对格式错误: {path}")
    _require((arr >= 0).all(), f"知识对存在负数: {path}")

    col0_max = int(arr[:, 0].max()) if arr.size else -1
    col1_max = int(arr[:, 1].max()) if arr.size else -1

    # 判别列语义
    # case A: col0 是 symptom, col1 是 herb
    if col0_max < S and col1_max < H:
        symptom_idx = arr[:, 0]
        herb_idx = arr[:, 1]
    # case B: col0 是 herb, col1 是 symptom
    elif col0_max < H and col1_max < S:
        herb_idx = arr[:, 0]
        symptom_idx = arr[:, 1]
    else:
        raise ValueError(
            f"无法识别 symptom-herb 知识列顺序或索引越界: {path} "
            f"(col0_max={col0_max}, col1_max={col1_max}, H={H}, S={S})"
        )

    K_hs = np.zeros((H, S), dtype=np.float64)
    K_hs[herb_idx, symptom_idx] = 1.0
    return K_hs


def _load_herb_alias_map(path: str, herb_names: List[str]) -> Dict[str, int]:
    """加载药材别名表，返回 alias -> herb_idx 映射。"""
    alias_to_idx: Dict[str, int] = {}
    herb_to_idx = {h: i for i, h in enumerate(herb_names)}

    if not path or (not os.path.isfile(path)):
        return alias_to_idx

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = [t.strip() for t in line.split(",") if t.strip()]
            if not tokens:
                continue
            # 约定首列是规范名，其余是别名；若首列不在 herb_names，则整行跳过
            canonical = tokens[0]
            idx = herb_to_idx.get(canonical)
            if idx is None:
                continue
            for name in tokens:
                alias_to_idx[name] = idx

    return alias_to_idx


def _load_symptom_herb_knowledge_txt(
    path: str,
    herb_names: List[str],
    symptom_names: List[str],
    herb_alias_csv: Optional[str] = None,
) -> np.ndarray:
    """加载 PTM 原始 txt (症状 + 药材列表) → K_hs (H, S)。

    格式: 每行 "症状<TAB>药1 药2 ..."，药材之间以空格分隔。
    若某行没有药材，忽略。
    """
    herb_to_idx = {h: i for i, h in enumerate(herb_names)}
    sym_to_idx = {s: i for i, s in enumerate(symptom_names)}
    alias_to_idx = _load_herb_alias_map(herb_alias_csv, herb_names)

    K_hs = np.zeros((len(herb_names), len(symptom_names)), dtype=np.float64)
    unknown_sym = 0
    unknown_herb: Set[str] = set()
    alias_hit = 0
    direct_hit = 0
    total_herb_tokens = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 先按 TAB 分割 (PTM 里多为 tab)
            parts = line.split("\t")
            symptom = parts[0].strip()
            herbs_str = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
            if not herbs_str:
                continue
            s_idx = sym_to_idx.get(symptom)
            if s_idx is None:
                unknown_sym += 1
                continue
            for herb in herbs_str.split():
                total_herb_tokens += 1
                h_idx = herb_to_idx.get(herb)
                if h_idx is not None:
                    direct_hit += 1
                else:
                    h_idx = alias_to_idx.get(herb)
                    if h_idx is not None:
                        alias_hit += 1
                if h_idx is None:
                    unknown_herb.add(herb)
                    continue
                K_hs[h_idx, s_idx] = 1.0

    matched_tokens = direct_hit + alias_hit
    matched_ratio = (matched_tokens / total_herb_tokens) if total_herb_tokens > 0 else 0.0

    if unknown_sym or unknown_herb:
        logger.warning("知识文件匹配不到: symptom=%d, herb=%d", unknown_sym, len(unknown_herb))

    logger.info(
        "知识映射统计: total=%d, direct=%d, alias=%d, matched_ratio=%.2f%%",
        total_herb_tokens, direct_hit, alias_hit, matched_ratio * 100.0,
    )

    # K_hs 覆盖度（用于判断知识是否有效进入模型）
    nnz = int(K_hs.sum())
    per_sym = (K_hs.sum(axis=0) > 0).sum()
    per_herb = (K_hs.sum(axis=1) > 0).sum()
    logger.info(
        "K_hs覆盖: nnz=%d, covered_symptoms=%d/%d, covered_herbs=%d/%d",
        nnz, int(per_sym), len(symptom_names), int(per_herb), len(herb_names),
    )

    if unknown_herb:
        samples = sorted(list(unknown_herb))[:20]
        logger.info("未匹配药名样例(前20): %s", "、".join(samples))

    return K_hs


def _load_symptom_herb_knowledge(path: str, H: int, S: int,
                                 herb_names: List[str], symptom_names: List[str],
                                 herb_alias_csv: Optional[str] = None) -> np.ndarray:
    """统一入口: 支持 PTM txt 或索引 CSV。"""
    if path.lower().endswith(".txt"):
        return _load_symptom_herb_knowledge_txt(
            path,
            herb_names,
            symptom_names,
            herb_alias_csv=herb_alias_csv,
        )
    return _load_symptom_herb_knowledge_indices(path, H=H, S=S)


def _build_pair_view(X_ph: sp.csr_matrix, min_support: int = 20) -> Tuple[np.ndarray, sp.csr_matrix]:
    """从处方-药材矩阵构建处方-药对视图（稀疏矩阵加速版）。

    1) 用 X^T X 统计全局共现频次并筛选高频 pair；
    2) 仅在筛选后的 pair 上逐行填充 X_pair。
    """
    X_bin = (X_ph > 0).astype(np.int8).tocsr()
    P, H = X_bin.shape

    # 全局 pair 频次: (H, H)
    cooc = (X_bin.T @ X_bin).tocoo()

    # 只保留上三角 + 达到支持阈值
    mask = (cooc.row < cooc.col) & (cooc.data >= min_support)
    pair_i = cooc.row[mask]
    pair_j = cooc.col[mask]

    if pair_i.size == 0:
        return np.zeros((0, 2), dtype=int), sp.csr_matrix((P, 0), dtype=np.float64)

    pair_index = np.vstack([pair_i, pair_j]).T.astype(int)
    # 保证稳定顺序
    order = np.lexsort((pair_index[:, 1], pair_index[:, 0]))
    pair_index = pair_index[order]

    pair_to_col = {(int(a), int(b)): idx for idx, (a, b) in enumerate(pair_index)}

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    indptr = X_bin.indptr
    indices = X_bin.indices
    for p in range(P):
        hs = indices[indptr[p]:indptr[p + 1]]
        n_h = len(hs)
        if n_h < 2:
            continue
        for i in range(n_h):
            a = int(hs[i])
            for j in range(i + 1, n_h):
                b = int(hs[j])
                if a < b:
                    key = (a, b)
                else:
                    key = (b, a)
                col = pair_to_col.get(key)
                if col is not None:
                    rows.append(p)
                    cols.append(col)
                    vals.append(1.0)

    X_pair = sp.coo_matrix((vals, (rows, cols)), shape=(P, len(pair_index)), dtype=np.float64).tocsr()
    X_pair.data[:] = 1.0
    X_pair.eliminate_zeros()
    return pair_index, X_pair


# ---------------------------------------------------------------------------
# 对齐校验
# ---------------------------------------------------------------------------

def _assert_name_alignment(name_a: List[str], name_b: List[str],
                            src_a: str, src_b: str):
    """断言两个名字列表完全一致（顺序 + 内容）。"""
    if name_a != name_b:
        # 找出第一个不同位置
        for i, (a, b) in enumerate(zip(name_a, name_b)):
            if a != b:
                raise ValueError(
                    f"列名不对齐: {src_a}[{i}]={a!r} vs {src_b}[{i}]={b!r}")
        if len(name_a) != len(name_b):
            raise ValueError(
                f"列数不同: {src_a}={len(name_a)} vs {src_b}={len(name_b)}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def load_all(
    data_root: Optional[str] = None,
    file_overrides: Optional[Dict[str, str]] = None,
    pair_min_support: int = 20,
    load_dosage: bool = True,
) -> AllData:
    """加载所有数据并做对齐校验。

    Parameters
    ----------
    data_root : str, optional
        数据根目录。None 则从环境变量/默认值解析。
    file_overrides : dict, optional
        覆盖默认文件名。

    Returns
    -------
    AllData
    """
    dpaths = resolve_paths(data_root, file_overrides)
    dpaths.validate(require_dosage=load_dosage)

    # 1. 主视图
    pid_ph, herb_names_ph, X_ph = _load_presence(dpaths.herb_presence_csv)
    pid_ps, sym_names_ps, X_ps = _load_presence(dpaths.symptom_presence_csv)
    if load_dosage:
        pid_pd, herb_names_pd, X_pd = _load_dosage(dpaths.herb_dosage_csv)
    else:
        logger.info("Skipping dosage matrix (load_dosage=False)")
        pid_pd, herb_names_pd = pid_ph, herb_names_ph
        X_pd = sp.csr_matrix(X_ph.shape, dtype=np.float64)

    # 处方 ID 对齐
    _require(np.array_equal(pid_ph, pid_ps),
             "herb_presence 与 symptom_presence 处方 ID 不一致")
    _require(np.array_equal(pid_ph, pid_pd),
             "herb_presence 与 herb_dosage 处方 ID 不一致")

    # 药材列名对齐: X_ph vs X_pd
    _assert_name_alignment(herb_names_ph, herb_names_pd,
                           "herb_presence", "herb_dosage")

    # 2. 药材属性
    hid_cat, hnames_cat, C_cat = _load_herb_category(dpaths.herb_category_csv)
    hid_feat, hnames_feat, A_feat = _load_herb_features(dpaths.herb_features_csv)

    _require(np.array_equal(hid_cat, hid_feat),
             "herb_category 与 herb_features herb_index 不一致")
    _assert_name_alignment(hnames_cat, hnames_feat,
                           "herb_category", "herb_features")
    # 药材列名 vs 主矩阵
    _assert_name_alignment(herb_names_ph, hnames_cat,
                           "herb_presence 列", "herb_category")

    # 拼接 F_h = [C_cat(30) || A_feat(21)] → (793, 51)
    F_h = np.hstack([C_cat, A_feat])
    _require(F_h.shape == (len(hnames_cat), 51),
             f"F_h 维度错误: {F_h.shape}")

    # 3. 症状属性
    sid_feat, snames_feat, F_s = _load_symptom_features(dpaths.symptom_features_csv)
    _assert_name_alignment(sym_names_ps, snames_feat,
                           "symptom_presence 列", "symptom_features")
    _require(F_s.shape == (len(snames_feat), 31),
             f"F_s 维度错误: {F_s.shape}")

    # 4. 共现对 & 禁忌对
    herb_cooc = _load_pairs(dpaths.herb_cooc_pairs_csv)
    symptom_cooc = _load_pairs(dpaths.symptom_cooc_pairs_csv)
    herb_mutex = _load_pairs(dpaths.herb_mutex_pairs_csv)

    # 索引范围校验
    H = len(hnames_cat)
    S = len(snames_feat)
    _require(herb_cooc.max() < H, f"药材共现索引越界: max={herb_cooc.max()}, H={H}")
    _require(symptom_cooc.max() < S, f"症状共现索引越界: max={symptom_cooc.max()}, S={S}")
    _require(herb_mutex.max() < H, f"禁忌对索引越界: max={herb_mutex.max()}, H={H}")

    # 5. PTM 可移植增强信息
    if dpaths.symptom_herb_knowledge_csv and os.path.isfile(dpaths.symptom_herb_knowledge_csv):
        K_hs = _load_symptom_herb_knowledge(
            dpaths.symptom_herb_knowledge_csv,
            H=H,
            S=S,
            herb_names=hnames_cat,
            symptom_names=snames_feat,
            herb_alias_csv=dpaths.herb_alias_csv,
        )
    else:
        logger.warning("未找到 symptom-herb 知识文件，使用全零 K_hs")
        K_hs = np.zeros((H, S), dtype=np.float64)
    pair_index, X_pair = _build_pair_view(X_ph, min_support=pair_min_support)

    # 6. 剂量 mask: M_pd = 1(X_pd > 0)
    #    接口预留: 未来可接入更精确的缺失标记
    M_pd = (X_pd > 0).astype(np.int8)  # type: sp.csr_matrix

    P = X_ph.shape[0]
    logger.info("数据加载完成: P=%d, H=%d, S=%d, N_pair=%d, |K_hs|=%d, F_h=%s, F_s=%s",
                P, H, S, X_pair.shape[1], int(K_hs.sum()), F_h.shape, F_s.shape)

    return AllData(
        X_ph=X_ph, X_ps=X_ps, X_pd=X_pd, M_pd=M_pd,
        F_h=F_h, F_s=F_s,
        herb_cooc=herb_cooc, symptom_cooc=symptom_cooc,
        K_hs=K_hs, pair_index=pair_index, X_pair=X_pair,
        herb_mutex=herb_mutex,
        herb_ids=hid_cat, herb_names=hnames_cat,
        symptom_ids=sid_feat, symptom_names=snames_feat,
        prescription_ids=pid_ph,
    )
