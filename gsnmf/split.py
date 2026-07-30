"""
数据切分策略。

支持三种切分:
  1. prescription     — 按处方行随机划分 train/valid/test
  2. leave_k_herbs    — 每张测试处方遮掉 k 个药材
  3. leave_k_symptoms — 每张测试处方遮掉 k 个症状
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from scipy import sparse as sp

from gsnmf.schemas import AllData, SharedMeta, SplitSlice, SplitData

logger = logging.getLogger(__name__)


def _make_shared_meta(data: AllData) -> SharedMeta:
    return SharedMeta(
        F_h=data.F_h, F_s=data.F_s,
        herb_cooc=data.herb_cooc, symptom_cooc=data.symptom_cooc,
        K_hs=data.K_hs, pair_index=data.pair_index,
        herb_mutex=data.herb_mutex,
        herb_ids=data.herb_ids, herb_names=data.herb_names,
        symptom_ids=data.symptom_ids, symptom_names=data.symptom_names,
    )


def _make_slice(data: AllData, idx: np.ndarray) -> SplitSlice:
    return SplitSlice(
        X_ph=data.X_ph[idx],
        X_ps=data.X_ps[idx],
        X_pd=data.X_pd[idx],
        M_pd=data.M_pd[idx],
        X_pair=data.X_pair[idx],
        prescription_ids=data.prescription_ids[idx],
    )


# ---------------------------------------------------------------------------
# 按处方划分
# ---------------------------------------------------------------------------

def _split_prescription(
    data: AllData,
    train_ratio: float,
    valid_ratio: float,
    rng: np.random.RandomState,
) -> SplitData:
    P = data.X_ph.shape[0]
    perm = rng.permutation(P)
    n_train = int(P * train_ratio)
    n_valid = int(P * valid_ratio)

    idx_train = perm[:n_train]
    idx_valid = perm[n_train:n_train + n_valid]
    idx_test  = perm[n_train + n_valid:]

    logger.info("Prescription split: train=%d, valid=%d, test=%d",
                len(idx_train), len(idx_valid), len(idx_test))

    return SplitData(
        meta=_make_shared_meta(data),
        train=_make_slice(data, idx_train),
        valid=_make_slice(data, idx_valid),
        test=_make_slice(data, idx_test),
    )


# ---------------------------------------------------------------------------
# Leave-k-out 系列 (在 test 切片上做 masking)
# ---------------------------------------------------------------------------

def _mask_k_per_row(X: sp.csr_matrix, k: int, rng: np.random.RandomState):
    """对每一行随机遮掉 k 个非零位置，返回 (X_masked, X_held_out)。"""
    X_dense = X.toarray().astype(float)
    held = np.zeros_like(X_dense)
    for i in range(X_dense.shape[0]):
        nz = np.where(X_dense[i] > 0)[0]
        if len(nz) <= k:
            # 至少保留 1 个可观测
            n_mask = max(len(nz) - 1, 0)
        else:
            n_mask = k
        if n_mask > 0:
            sel = rng.choice(nz, n_mask, replace=False)
            held[i, sel] = X_dense[i, sel]
            X_dense[i, sel] = 0
    return sp.csr_matrix(X_dense), sp.csr_matrix(held)


def _split_leave_k(
    data: AllData,
    train_ratio: float,
    valid_ratio: float,
    leave_k: int,
    target: str,   # "herbs" | "symptoms"
    rng: np.random.RandomState,
) -> SplitData:
    """先做 prescription split，再对 valid/test 做 leave-k masking。"""
    base = _split_prescription(data, train_ratio, valid_ratio, rng)

    for split_name, sl in [("valid", base.valid), ("test", base.test)]:
        if target == "herbs":
            sl.X_ph, _ = _mask_k_per_row(sl.X_ph, leave_k, rng)
            # 同步更新剂量 mask
            sl.M_pd = sl.M_pd.multiply(sl.X_ph)
        else:
            sl.X_ps, _ = _mask_k_per_row(sl.X_ps, leave_k, rng)
        logger.info("Leave-%d-%s masking applied to %s set",
                     leave_k, target, split_name)

    return base


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def split_data(data: AllData, cfg: Dict) -> SplitData:
    """根据配置切分数据。

    Parameters
    ----------
    data : AllData
    cfg : dict
        split 配置块，需包含 method, train_ratio, valid_ratio, test_ratio, leave_k

    Returns
    -------
    SplitData
    """
    method = cfg.get("method", "prescription")
    train_r = cfg.get("train_ratio", 0.8)
    valid_r = cfg.get("valid_ratio", 0.1)
    leave_k = cfg.get("leave_k", 3)
    seed = cfg.get("seed", 42)
    rng = np.random.RandomState(seed)

    if method == "prescription":
        return _split_prescription(data, train_r, valid_r, rng)
    elif method == "leave_k_herbs":
        return _split_leave_k(data, train_r, valid_r, leave_k, "herbs", rng)
    elif method == "leave_k_symptoms":
        return _split_leave_k(data, train_r, valid_r, leave_k, "symptoms", rng)
    else:
        raise ValueError(f"未知切分方法: {method}")
