"""
MV-GSNMTF 核心模型。

min_{G,H_h,H_s,D_h >= 0}
    ‖X_ph - G H_h^T‖²    + α‖X_ps - G H_s^T‖²
  + β‖M ⊙ (X_pd - G D_h^T)‖²
  + λ_h tr(H_h^T L_h H_h) + λ_s tr(H_s^T L_s H_s)
  + γ_g‖G‖₁ + γ_h‖H_h‖₁ + γ_s‖H_s‖₁ + γ_d‖D_h‖₁
  + ρ tr(H_h^T C_hh H_h)

交替投影梯度法: G_p → H_h → H_s → D_h
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from scipy import sparse as sp

from gsnmf.backend import get_backend, to_numpy
from gsnmf.hypergraph_builder import compute_hyper_direct_loss, compute_hyper_direct_grad
from gsnmf.constraints import contra_gradient, contra_penalty
from gsnmf.schemas import LossComponents, ModelFactors

logger = logging.getLogger(__name__)


# =========================================================================
# 初始化
# =========================================================================

def _nndsvd_init(X: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """NNDSVD 初始化 (Boutsidis & Gallopoulos 2008)。

    对 X ≈ W H^T，返回 (W, H) 均非负。
    """
    U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
    W = np.zeros((X.shape[0], K))
    H = np.zeros((X.shape[1], K))

    # 第一个分量
    avg = np.sqrt(X.mean())
    W[:, 0] = avg
    H[:, 0] = avg

    for j in range(1, min(K, len(Sigma))):
        u = U[:, j]
        v = Vt[j, :]
        s = Sigma[j]

        u_pos = np.maximum(u, 0)
        u_neg = np.maximum(-u, 0)
        v_pos = np.maximum(v, 0)
        v_neg = np.maximum(-v, 0)

        n_u_pos = np.linalg.norm(u_pos)
        n_u_neg = np.linalg.norm(u_neg)
        n_v_pos = np.linalg.norm(v_pos)
        n_v_neg = np.linalg.norm(v_neg)

        mp = n_u_pos * n_v_pos
        mn = n_u_neg * n_v_neg

        if mp >= mn:
            scale = np.sqrt(s * mp)
            if n_u_pos > 0:
                W[:, j] = scale * u_pos / n_u_pos
            if n_v_pos > 0:
                H[:, j] = scale * v_pos / n_v_pos
        else:
            scale = np.sqrt(s * mn)
            if n_u_neg > 0:
                W[:, j] = scale * u_neg / n_u_neg
            if n_v_neg > 0:
                H[:, j] = scale * v_neg / n_v_neg

    W = np.maximum(W, 1e-10)
    H = np.maximum(H, 1e-10)
    return W, H


def _random_init(shape: tuple, rng: np.random.RandomState) -> np.ndarray:
    return np.abs(rng.randn(*shape)) * 0.1 + 1e-4


# =========================================================================
# 模型类
# =========================================================================

class MVGSNMTF:
    """Multi-View Graph-Regularized Sparse NMT-Factorization."""

    def __init__(
        self,
        K: int,
        alpha: float = 1.0,
        beta: float = 0.2,
        lambda_h: float = 1e-3,
        lambda_s: float = 1e-3,
        lambda_hyper: float = 0.0,
        beta_pair: float = 0.0,
        gamma_g: float = 0.0,
        gamma_h: float = 1e-5,
        gamma_s: float = 1e-5,
        gamma_d: float = 1e-6,
        rho: float = 0.0,
        lambda_know: float = 0.0,
        role_aware: bool = False,
        n_roles: int = 4,
        role_exclusive: float = 0.0,
        lr: float = 1e-3,
        grad_clip: float = 1e4,
        normalize_columns: bool = False,
        update_rule: str = "pgd",
        device: str = "cpu",
        loss_switches: Optional[Dict[str, bool]] = None,
    ):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.lambda_h = lambda_h
        self.lambda_s = lambda_s
        self.lambda_hyper = lambda_hyper
        self.beta_pair = beta_pair
        self.gamma_g = gamma_g
        self.gamma_h = gamma_h
        self.gamma_s = gamma_s
        self.gamma_d = gamma_d
        self.rho = rho
        self.lambda_know = lambda_know
        self.role_aware = role_aware
        self.n_roles = n_roles
        self.role_exclusive = role_exclusive
        self.lr = lr
        self.grad_clip = grad_clip
        self.normalize_columns = normalize_columns
        self.update_rule = update_rule
        self.device = device
        self.xp, self.xsp = get_backend(device)

        # 默认 loss 开关
        self._switches = {
            "ph": True, "ps": True, "pd": False,
            "graph_h": False, "graph_s": False,
            "hyper_h": False,
            "hyper_var": False,
            "hyper_mean": False,
            "l1": False, "contra": False,
            "pair": False,
            "know_hs": False,
        }
        if loss_switches:
            self._switches.update(loss_switches)

        # 因子矩阵（init_factors 后填充）
        self.G_p: Optional[np.ndarray] = None
        self.G_p_roles: Optional[np.ndarray] = None
        self.H_h: Optional[np.ndarray] = None
        self.H_h_roles: Optional[np.ndarray] = None
        self.H_s: Optional[np.ndarray] = None
        self.D_h: Optional[np.ndarray] = None
        self.H_pair: Optional[np.ndarray] = None

        # 直接超图正则化数据 (由 trainer 注入)
        self.hyper_edges: List = []
        self.hyper_weights: Optional[np.ndarray] = None

    # ----- 属性 -----

    @property
    def sw(self) -> Dict[str, bool]:
        return self._switches

    def _effective_Hh(self):
        """返回参与重构/耦合的药材主题矩阵。"""
        if self.role_aware and self.H_h_roles is not None:
            return self.xp.sum(self.H_h_roles, axis=0)
        return self.H_h

    def factors(self) -> ModelFactors:
        """返回因子 (始终为 CPU numpy 数组)。"""
        H_h_eff = self._effective_Hh()
        return ModelFactors(
            G_p=to_numpy(self.G_p).copy(),
            H_h=to_numpy(H_h_eff).copy(),
            H_s=to_numpy(self.H_s).copy(),
            D_h=to_numpy(self.D_h).copy(),
            H_pair=None if self.H_pair is None else to_numpy(self.H_pair).copy(),
            G_p_roles=None if not self.role_aware or self.G_p_roles is None else to_numpy(self.G_p_roles).copy(),
            H_h_roles=None if not self.role_aware or self.H_h_roles is None else to_numpy(self.H_h_roles).copy(),
        )

    def load_factors(self, f: ModelFactors):
        """从 ModelFactors 恢复因子到模型。"""
        xp = self.xp
        self.G_p = xp.asarray(f.G_p) if hasattr(xp, 'asarray') else f.G_p.copy()
        self.H_h = xp.asarray(f.H_h) if hasattr(xp, 'asarray') else f.H_h.copy()
        if self.role_aware:
            if getattr(f, 'H_h_roles', None) is not None and getattr(f, 'G_p_roles', None) is not None:
                self.H_h_roles = xp.asarray(f.H_h_roles) if hasattr(xp, 'asarray') else f.H_h_roles.copy()
                self.G_p_roles = xp.asarray(f.G_p_roles) if hasattr(xp, 'asarray') else f.G_p_roles.copy()
            else:
                self.H_h_roles = xp.stack([self.H_h / float(self.n_roles)
                                           for _ in range(self.n_roles)], axis=0)
                self.G_p_roles = xp.stack([self.G_p / float(self.n_roles)
                                           for _ in range(self.n_roles)], axis=0)
        self.H_s = xp.asarray(f.H_s) if hasattr(xp, 'asarray') else f.H_s.copy()
        self.D_h = xp.asarray(f.D_h) if hasattr(xp, 'asarray') else f.D_h.copy()
        if f.H_pair is not None:
            self.H_pair = xp.asarray(f.H_pair) if hasattr(xp, 'asarray') else f.H_pair.copy()

    # -----------------------------------------------------------------
    # 初始化
    # -----------------------------------------------------------------

    def init_factors(
        self,
        P: int, H: int, S: int,
        method: str = "nndsvd",
        X_ph: Optional[np.ndarray] = None,
        X_ps: Optional[np.ndarray] = None,
        N_pair: Optional[int] = None,
        seed: int = 42,
    ):
        """初始化四组隐因子。

        Parameters
        ----------
        method : "nndsvd" | "random"
        X_ph : 用于 NNDSVD 的药材存在矩阵 (稠密)
        X_ps : 用于 NNDSVD 的症状存在矩阵 (稠密)
        """
        rng = np.random.RandomState(seed)

        # 初始化在 CPU 上做，然后传到设备
        if method == "nndsvd" and X_ph is not None:
            # NNDSVD 需要 CPU numpy 输入
            X_ph_cpu = to_numpy(X_ph) if not isinstance(X_ph, np.ndarray) else X_ph
            X_ps_cpu = to_numpy(X_ps) if (X_ps is not None and not isinstance(X_ps, np.ndarray)) else X_ps
            G, Hh = _nndsvd_init(X_ph_cpu, self.K)
            if X_ps_cpu is not None:
                _, Hs = _nndsvd_init(X_ps_cpu, self.K)
            else:
                Hs = _random_init((S, self.K), rng)
            Dh = _random_init((H, self.K), rng)
            # G_p 行归一化
            row_norm = G.sum(axis=1, keepdims=True)
            row_norm = np.maximum(row_norm, 1e-10)
            G = G / row_norm
        else:
            G = _random_init((P, self.K), rng)
            Hh = _random_init((H, self.K), rng)
            Hs = _random_init((S, self.K), rng)
            Dh = _random_init((H, self.K), rng)

        # 传到目标设备
        xp = self.xp
        # 统一 float32，提升 GPU 吞吐
        self.G_p = xp.asarray(G, dtype=np.float32)
        self.H_h = xp.asarray(Hh, dtype=np.float32)
        if self.role_aware:
            # 轻微扰动后按 role 均分初始化
            base_H = self.H_h / float(self.n_roles)
            base_G = self.G_p / float(self.n_roles)
            roles_H = []
            roles_G = []
            for r in range(self.n_roles):
                noise_H = xp.asarray(rng.rand(*base_H.shape), dtype=np.float32) * 0.01
                roles_H.append(xp.maximum(base_H * (1.0 + noise_H), 1e-10))
                noise_G = xp.asarray(rng.rand(*base_G.shape), dtype=np.float32) * 0.01
                roles_G.append(xp.maximum(base_G * (1.0 + noise_G), 1e-10))
            self.H_h_roles = xp.stack(roles_H, axis=0)
            self.G_p_roles = xp.stack(roles_G, axis=0)
            # 重新同步聚合
            self.H_h = xp.sum(self.H_h_roles, axis=0)
            self.G_p = xp.sum(self.G_p_roles, axis=0)
        else:
            self.H_h_roles = None
            self.G_p_roles = None
        self.H_s = xp.asarray(Hs, dtype=np.float32)
        self.D_h = xp.asarray(Dh, dtype=np.float32)
        if N_pair is not None and N_pair > 0:
            Hp = _random_init((N_pair, self.K), rng).astype(np.float32, copy=False)
            self.H_pair = xp.asarray(Hp, dtype=np.float32)
        else:
            self.H_pair = None

        logger.info("因子初始化完成 (method=%s, device=%s): G(%s) H_h(%s) H_s(%s) D_h(%s) H_pair(%s)",
                     method, self.device, self.G_p.shape, self.H_h.shape,
                     self.H_s.shape, self.D_h.shape,
                     None if self.H_pair is None else self.H_pair.shape)

    # -----------------------------------------------------------------
    # Loss 计算
    # -----------------------------------------------------------------

    def compute_loss(
        self,
        X_ph: np.ndarray,
        X_ps: np.ndarray,
        X_pd: Optional[np.ndarray] = None,
        M_pd: Optional[np.ndarray] = None,
        X_pair: Optional[np.ndarray] = None,
        L_h: Optional[sp.csr_matrix] = None,
        L_s: Optional[sp.csr_matrix] = None,
        C_hh: Optional[sp.csr_matrix] = None,
        L_hyper_h: Optional[sp.csr_matrix] = None,
        K_hs: Optional[np.ndarray] = None,
    ) -> LossComponents:
        """计算目标函数各分量。所有输入应为稠密数组 (xp.ndarray)。"""
        xp = self.xp
        lc = LossComponents()

        H_h_eff = self._effective_Hh()

        if self.sw["ph"]:
            if self.role_aware and self.H_h_roles is not None and self.G_p_roles is not None:
                ph_hat = xp.zeros_like(X_ph, dtype=np.float32)
                for r in range(self.n_roles):
                    ph_hat += self.G_p_roles[r] @ self.H_h_roles[r].T
                res_ph = X_ph - ph_hat
            else:
                res_ph = X_ph - self.G_p @ H_h_eff.T
            lc.loss_ph = float(xp.sum(res_ph ** 2))

        if self.sw["ps"]:
            res_ps = X_ps - self.G_p @ self.H_s.T
            lc.loss_ps = self.alpha * float(xp.sum(res_ps ** 2))

        if self.sw["pd"] and X_pd is not None and M_pd is not None:
            res_pd = M_pd * (X_pd - self.G_p @ self.D_h.T)
            lc.loss_pd = self.beta * float(xp.sum(res_pd ** 2))

        if self.sw.get("pair", False) and X_pair is not None and self.H_pair is not None:
            # Avoid materializing the dense P x N_pair reconstruction/residual.
            # This is mathematically identical to ||X - GH^T||_F^2:
            #   ||X||_F^2 - 2 tr(G^T X H) + tr((G^T G)(H^T H)).
            # Keeping X_pair sparse is essential for low support thresholds on
            # memory-constrained GPUs.
            if hasattr(X_pair, "multiply"):
                x_sq = X_pair.multiply(X_pair).sum(dtype=xp.float64)
                xh = X_pair @ self.H_pair
                cross = xp.sum(self.G_p * xh, dtype=xp.float64)
                gram_g = self.G_p.T @ self.G_p
                gram_h = self.H_pair.T @ self.H_pair
                recon_sq = xp.sum(gram_g * gram_h.T, dtype=xp.float64)
                pair_sq = (
                    xp.asarray(x_sq, dtype=xp.float64)
                    - 2.0 * cross
                    + recon_sq
                )
                lc.loss_pair = self.beta_pair * float(
                    xp.maximum(pair_sq, 0.0)
                )
            else:
                res_pair = X_pair - self.G_p @ self.H_pair.T
                lc.loss_pair = self.beta_pair * float(xp.sum(res_pair ** 2))

        if self.sw["graph_h"] and L_h is not None:
            LH = L_h.dot(self.H_h)
            lc.loss_graph_h = self.lambda_h * float(xp.sum(self.H_h * LH))

        if self.sw["graph_s"] and L_s is not None:
            LH = L_s.dot(self.H_s)
            lc.loss_graph_s = self.lambda_s * float(xp.sum(self.H_s * LH))

        if self.sw["hyper_h"] and L_hyper_h is not None:
            LH = L_hyper_h @ self.H_h
            lc.loss_hyper_h = self.lambda_hyper * float(xp.sum(self.H_h * LH))

        if self.sw["hyper_var"] and self.hyper_edges:
            H_h_np = to_numpy(self.H_h)
            lc.loss_hyper_var = self.lambda_hyper * compute_hyper_direct_loss(
                H_h_np, self.hyper_edges, self.hyper_weights, normalize=True)

        if self.sw["hyper_mean"] and self.hyper_edges:
            H_h_np = to_numpy(self.H_h)
            lc.loss_hyper_mean = self.lambda_hyper * compute_hyper_direct_loss(
                H_h_np, self.hyper_edges, self.hyper_weights, normalize=False)

        if self.sw["l1"]:
            lc.loss_l1_g = self.gamma_g * float(xp.sum(xp.abs(self.G_p)))
            lc.loss_l1_h = self.gamma_h * float(xp.sum(xp.abs(self.H_h)))
            lc.loss_l1_s = self.gamma_s * float(xp.sum(xp.abs(self.H_s)))
            lc.loss_l1_d = self.gamma_d * float(xp.sum(xp.abs(self.D_h)))

        if self.sw["contra"] and C_hh is not None:
            lc.loss_contra = self.rho * contra_penalty(self.H_h, C_hh)

        if self.sw["know_hs"] and K_hs is not None:
            # K_hs: (H, S), 预测为 H_h_eff @ H_s^T
            diff_know = K_hs - (H_h_eff @ self.H_s.T)
            lc.loss_know_hs = self.lambda_know * float(xp.sum(diff_know ** 2))

        if self.role_aware and self.H_h_roles is not None and self.role_exclusive > 0:
            # role exclusivity: 抑制同一 herb-topic 在多个 role 同时激活
            # penalty = sum_{r<r'} <H_r, H_r'>
            excl = 0.0
            for r in range(self.n_roles):
                Hr = self.H_h_roles[r]
                for r2 in range(r + 1, self.n_roles):
                    excl += float(xp.sum(Hr * self.H_h_roles[r2]))
            lc.loss_l1_h += self.role_exclusive * excl

        lc.total = (lc.loss_ph + lc.loss_ps + lc.loss_pd + lc.loss_pair
                    + lc.loss_graph_h + lc.loss_graph_s
                    + lc.loss_hyper_h + lc.loss_hyper_var + lc.loss_hyper_mean
                    + lc.loss_l1_g + lc.loss_l1_h + lc.loss_l1_s + lc.loss_l1_d
                    + lc.loss_contra + lc.loss_know_hs)
        return lc

    # -----------------------------------------------------------------
    # 梯度
    # -----------------------------------------------------------------

    def _grad_Gp_shared(
        self, X_ps: np.ndarray,
        X_pd: Optional[np.ndarray], M_pd: Optional[np.ndarray],
        X_pair: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        xp = self.xp
        grad = xp.zeros_like(self.G_p)

        if self.sw["ps"]:
            grad += 2.0 * self.alpha * (self.G_p @ self.H_s.T - X_ps) @ self.H_s

        if self.sw["pd"] and X_pd is not None and M_pd is not None:
            res = M_pd * (self.G_p @ self.D_h.T - X_pd)
            grad += 2.0 * self.beta * res @ self.D_h

        if self.sw.get("pair", False) and X_pair is not None and self.H_pair is not None:
            if hasattr(X_pair, "multiply"):
                pair_grad = (
                    self.G_p @ (self.H_pair.T @ self.H_pair)
                    - X_pair @ self.H_pair
                )
            else:
                pair_grad = (
                    self.G_p @ self.H_pair.T - X_pair
                ) @ self.H_pair
            grad += 2.0 * self.beta_pair * pair_grad

        return grad

    def _grad_Hh_shared(
        self,
        L_h: Optional[sp.csr_matrix],
        C_hh: Optional[sp.csr_matrix],
        L_hyper_h: Optional[sp.csr_matrix] = None,
        K_hs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        xp = self.xp
        grad = xp.zeros_like(self.H_h)
        H_h_eff = self._effective_Hh()

        if self.sw["graph_h"] and L_h is not None:
            grad += 2.0 * self.lambda_h * L_h.dot(self.H_h)

        if self.sw["hyper_h"] and L_hyper_h is not None:
            grad += 2.0 * self.lambda_hyper * (L_hyper_h @ self.H_h)

        if self.sw["hyper_var"] and self.hyper_edges:
            H_h_np = to_numpy(self.H_h)
            grad += self.lambda_hyper * compute_hyper_direct_grad(
                H_h_np, self.hyper_edges, self.hyper_weights, normalize=True)

        if self.sw["hyper_mean"] and self.hyper_edges:
            H_h_np = to_numpy(self.H_h)
            grad += self.lambda_hyper * compute_hyper_direct_grad(
                H_h_np, self.hyper_edges, self.hyper_weights, normalize=False)

        if self.sw["contra"] and C_hh is not None:
            grad += self.rho * contra_gradient(self.H_h, C_hh)

        if self.sw["know_hs"] and K_hs is not None:
            grad += 2.0 * self.lambda_know * (H_h_eff @ self.H_s.T - K_hs) @ self.H_s

        return grad

    def _grad_Hs(
        self,
        X_ps: np.ndarray,
        L_s: Optional[sp.csr_matrix],
        K_hs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        xp = self.xp
        grad = xp.zeros_like(self.H_s)

        if self.sw["ps"]:
            GtG = self.G_p.T @ self.G_p
            grad += 2.0 * self.alpha * (self.H_s @ GtG - X_ps.T @ self.G_p)

        if self.sw["graph_s"] and L_s is not None:
            grad += 2.0 * self.lambda_s * L_s.dot(self.H_s)

        H_h_eff = self._effective_Hh()

        if self.sw["know_hs"] and K_hs is not None:
            # d/dH_s ||K_hs - H_h H_s^T||^2 = 2 (H_h H_s^T - K_hs)^T H_h
            grad += 2.0 * self.lambda_know * (H_h_eff @ self.H_s.T - K_hs).T @ H_h_eff

        return grad

    def _grad_Dh(
        self,
        X_pd: np.ndarray,
        M_pd: np.ndarray,
    ) -> np.ndarray:
        xp = self.xp
        grad = xp.zeros_like(self.D_h)

        if self.sw["pd"]:
            res = M_pd * (self.G_p @ self.D_h.T - X_pd)
            grad += 2.0 * self.beta * res.T @ self.G_p

        return grad

    def _grad_Hpair(
        self,
        X_pair: np.ndarray,
    ) -> np.ndarray:
        xp = self.xp
        grad = xp.zeros_like(self.H_pair)
        if self.sw.get("pair", False) and self.H_pair is not None:
            grad += 2.0 * self.beta_pair * (self.H_pair @ (self.G_p.T @ self.G_p) - X_pair.T @ self.G_p)
        return grad

    # -----------------------------------------------------------------
    # 更新步骤
    # -----------------------------------------------------------------

    def _clip_grad(self, grad, max_norm: float):
        """按 L2 范数裁剪梯度，防止数值爆炸。"""
        xp = self.xp
        grad_norm = float(xp.linalg.norm(grad))
        if grad_norm > max_norm:
            grad = grad * (max_norm / grad_norm)
        return grad

    def _proj_step(self, var, grad, lr: float):
        """梯度下降 + 非负投影。"""
        return self.xp.maximum(var - lr * grad, 0.0)

    def _prox_l1(self, var, gamma: float, lr: float):
        """L1 近端算子 (soft-threshold) + 非负投影。"""
        return self.xp.maximum(var - lr * gamma, 0.0)

    def _normalize_H_columns(self, H, G):
        """列归一化: 把 H 的每列归一化到 L2=1，尺度吸收进 G。

        X ≈ G @ H.T = (G * s) @ (H / s).T，保持重构不变。
        消除 NMF 主题尺度歧义，让 NNLS 推断更稳定。
        """
        xp = self.xp
        col_norms = xp.linalg.norm(H, axis=0, keepdims=True)  # (1, K)
        col_norms = xp.maximum(col_norms, 1e-10)
        H_new = H / col_norms
        G_new = G * col_norms  # 把尺度吸收进 G
        return H_new, G_new

    # -----------------------------------------------------------------
    # 一轮完整更新
    # -----------------------------------------------------------------

    def fit_step(
        self,
        X_ph: np.ndarray,
        X_ps: np.ndarray,
        X_pd: Optional[np.ndarray] = None,
        M_pd: Optional[np.ndarray] = None,
        X_pair: Optional[np.ndarray] = None,
        L_h: Optional[sp.csr_matrix] = None,
        L_s: Optional[sp.csr_matrix] = None,
        C_hh: Optional[sp.csr_matrix] = None,
        L_hyper_h: Optional[sp.csr_matrix] = None,
        K_hs: Optional[np.ndarray] = None,
    ) -> LossComponents:
        """一轮交替更新 (自动分发到 PGD / MUR / Hybrid)。"""
        rule = getattr(self, 'update_rule', 'pgd')
        if rule == 'mur':
            return self.fit_step_mur(
                X_ph, X_ps, X_pd, M_pd, X_pair,
                L_h, L_s, C_hh, L_hyper_h, K_hs)
        elif rule == 'hybrid':
            return self.fit_step_hybrid(
                X_ph, X_ps, X_pd, M_pd, X_pair,
                L_h, L_s, C_hh, L_hyper_h, K_hs)
        else:
            return self.fit_step_pgd(
                X_ph, X_ps, X_pd, M_pd, X_pair,
                L_h, L_s, C_hh, L_hyper_h, K_hs)

    def fit_step_pgd(
        self,
        X_ph: np.ndarray,
        X_ps: np.ndarray,
        X_pd: Optional[np.ndarray] = None,
        M_pd: Optional[np.ndarray] = None,
        X_pair: Optional[np.ndarray] = None,
        L_h: Optional[sp.csr_matrix] = None,
        L_s: Optional[sp.csr_matrix] = None,
        C_hh: Optional[sp.csr_matrix] = None,
        L_hyper_h: Optional[sp.csr_matrix] = None,
        K_hs: Optional[np.ndarray] = None,
    ) -> LossComponents:
        """一轮交替更新: G_p → H_h → H_s → D_h (投影梯度法)。

        所有矩阵输入应为稠密 ndarray (已在 trainer 中转换)。

        Returns
        -------
        LossComponents : 更新后的 loss 各分量
        """
        lr = self.lr
        clip = self.grad_clip

        # --- Compute Predict & Updates ---
        # 提取共享梯度
        grad_g_shared = self._grad_Gp_shared(X_ps, X_pd, M_pd, X_pair)
        grad_h_shared = self._grad_Hh_shared(L_h, C_hh, L_hyper_h, K_hs)

        if self.role_aware and self.G_p_roles is not None and self.H_h_roles is not None:
            # TRUE ROLE-AWARE 拆分计算
            ph_hat = self.xp.zeros_like(X_ph, dtype=np.float32)
            for r in range(self.n_roles):
                ph_hat += self.G_p_roles[r] @ self.H_h_roles[r].T
            res_ph = ph_hat - X_ph
            
            # Update G_p_roles
            for r in range(self.n_roles):
                grad_ph_g_r = 2.0 * res_ph @ self.H_h_roles[r] if self.sw["ph"] else 0.0
                grad_g_r = self._clip_grad(grad_ph_g_r + grad_g_shared, clip)
                self.G_p_roles[r] = self._proj_step(self.G_p_roles[r], grad_g_r, lr)
                if self.sw["l1"]:
                    self.G_p_roles[r] = self._prox_l1(self.G_p_roles[r], self.gamma_g, lr)
            self.G_p = self.xp.sum(self.G_p_roles, axis=0)
            
            # 重算 res_ph, 因为上面 G_p_roles 变了 (或者继续用刚才的近似，为了速度这里复用)
            # 不过为了准确推荐重算
            ph_hat = self.xp.zeros_like(X_ph, dtype=np.float32)
            for r in range(self.n_roles):
                ph_hat += self.G_p_roles[r] @ self.H_h_roles[r].T
            res_ph = ph_hat - X_ph

            # Update H_h_roles
            for r in range(self.n_roles):
                grad_ph_h_r = 2.0 * res_ph.T @ self.G_p_roles[r] if self.sw["ph"] else 0.0
                grad_h_r = grad_ph_h_r + grad_h_shared
                
                if self.role_exclusive > 0:
                    cross = self.xp.zeros_like(self.H_h_roles[r])
                    for r2 in range(self.n_roles):
                        if r2 != r:
                            cross += self.H_h_roles[r2]
                    grad_h_r += self.role_exclusive * cross
                    
                grad_h_r = self._clip_grad(grad_h_r, clip)
                self.H_h_roles[r] = self._proj_step(self.H_h_roles[r], grad_h_r, lr)
                if self.sw["l1"]:
                    self.H_h_roles[r] = self._prox_l1(self.H_h_roles[r], self.gamma_h, lr)
            self.H_h = self.xp.sum(self.H_h_roles, axis=0)

        else:
            # DEFAULT GSNMF:
            # Update G_p
            grad_ph_g = 2.0 * (self.G_p @ self.H_h.T - X_ph) @ self.H_h if self.sw["ph"] else 0.0
            grad_g = self._clip_grad(grad_ph_g + grad_g_shared, clip)
            self.G_p = self._proj_step(self.G_p, grad_g, lr)
            if self.sw["l1"]:
                self.G_p = self._prox_l1(self.G_p, self.gamma_g, lr)
                
            # Update H_h
            H_h_eff = self._effective_Hh()
            GtG = self.G_p.T @ self.G_p
            grad_ph_h = 2.0 * (H_h_eff @ GtG - X_ph.T @ self.G_p) if self.sw["ph"] else 0.0
            grad_h = self._clip_grad(grad_ph_h + grad_h_shared, clip)
            self.H_h = self._proj_step(self.H_h, grad_h, lr)
            if self.sw["l1"]:
                self.H_h = self._prox_l1(self.H_h, self.gamma_h, lr)

            # 列归一化: H_h 尺度 → G_p
            if self.normalize_columns:
                self.H_h, self.G_p = self._normalize_H_columns(self.H_h, self.G_p)

        # --- Update H_s ---
        grad_s = self._clip_grad(self._grad_Hs(X_ps, L_s, K_hs), clip)
        self.H_s = self._proj_step(self.H_s, grad_s, lr)
        if self.sw["l1"]:
            self.H_s = self._prox_l1(self.H_s, self.gamma_s, lr)

        # 列归一化: H_s 尺度 → G_p
        if self.normalize_columns:
            self.H_s, self.G_p = self._normalize_H_columns(self.H_s, self.G_p)

        # --- Update D_h ---
        if self.sw["pd"] and X_pd is not None and M_pd is not None:
            grad_d = self._clip_grad(self._grad_Dh(X_pd, M_pd), clip)
            self.D_h = self._proj_step(self.D_h, grad_d, lr)
            if self.sw["l1"]:
                self.D_h = self._prox_l1(self.D_h, self.gamma_d, lr)

        # --- Update H_pair ---
        if self.sw.get("pair", False) and X_pair is not None and self.H_pair is not None:
            grad_pair = self._clip_grad(self._grad_Hpair(X_pair), clip)
            self.H_pair = self._proj_step(self.H_pair, grad_pair, lr)

        # --- Compute loss after update ---
        return self.compute_loss(X_ph, X_ps, X_pd, M_pd, X_pair, L_h, L_s, C_hh, L_hyper_h, K_hs)

    # -----------------------------------------------------------------
    # 乘法更新规则 (Multiplicative Update Rules)
    # -----------------------------------------------------------------

    def _split_sparse_pos_neg(self, L):
        """将稀疏矩阵(如Laplacian)分解为正部和负部: L = L_pos - L_neg。

        对于 Laplacian L = D - W:
          L_pos 包含对角(度)元素 → 正项 (进入 MUR 分母)
          L_neg 包含邻接权重    → 正项 (进入 MUR 分子)
        """
        if L is None:
            return None, None
        if sp.issparse(L) or (
            hasattr(L, "format") and hasattr(L, "maximum")
        ):
            L_pos = L.maximum(0)   # 非负部分
            L_neg = (-L).maximum(0) # 绝对值的负部分
        else:
            L_pos = self.xp.maximum(L, 0)
            L_neg = self.xp.maximum(-L, 0)
        return L_pos, L_neg

    def fit_step_mur(
        self,
        X_ph: np.ndarray,
        X_ps: np.ndarray,
        X_pd: Optional[np.ndarray] = None,
        M_pd: Optional[np.ndarray] = None,
        X_pair: Optional[np.ndarray] = None,
        L_h: Optional[sp.csr_matrix] = None,
        L_s: Optional[sp.csr_matrix] = None,
        C_hh: Optional[sp.csr_matrix] = None,
        L_hyper_h: Optional[sp.csr_matrix] = None,
        K_hs: Optional[np.ndarray] = None,
    ) -> LossComponents:
        """一轮交替乘法更新: G_p → H_h → H_s → D_h。

        Lee-Seung 乘法更新规则 (MUR) 天然保非负、无需学习率调参。
        对于 min||X - GH^T||^2 + 正则:
          H ← H * (numerator / denominator)
        其中 numerator 收集 -grad 的正项, denominator 收集 -grad 的负项。

        参考: Lee & Seung 2001, Cai et al. 2011 (GNMF)
        """
        xp = self.xp
        eps = 1e-10

        # 预分解 Laplacian 正/负部分 (每步都做以支持 warmup 期间权重变化)
        L_h_pos, L_h_neg = self._split_sparse_pos_neg(L_h)
        L_s_pos, L_s_neg = self._split_sparse_pos_neg(L_s)

        H_h_eff = self._effective_Hh()

        # =================================================================
        # Update G_p (非角色感知模式)
        # =================================================================
        if not (self.role_aware and self.G_p_roles is not None):
            num_g = xp.zeros_like(self.G_p)
            den_g = xp.zeros_like(self.G_p)

            if self.sw["ph"]:
                num_g += X_ph @ self.H_h                        # X @ H
                den_g += self.G_p @ (self.H_h.T @ self.H_h)    # G @ H^T H

            if self.sw["ps"]:
                num_g += self.alpha * (X_ps @ self.H_s)
                den_g += self.alpha * (self.G_p @ (self.H_s.T @ self.H_s))

            if self.sw.get("pair", False) and X_pair is not None and self.H_pair is not None:
                num_g += self.beta_pair * (X_pair @ self.H_pair)
                den_g += self.beta_pair * (self.G_p @ (self.H_pair.T @ self.H_pair))

            if self.sw["pd"] and X_pd is not None and M_pd is not None:
                M2 = M_pd * M_pd  # element-wise mask squared
                num_g += self.beta * ((M2 * X_pd) @ self.D_h)
                den_g += self.beta * ((M2 * (self.G_p @ self.D_h.T)) @ self.D_h)

            if self.sw["l1"]:
                den_g += self.gamma_g  # L1 对非负变量的梯度恒为 1

            self.G_p = self.G_p * (num_g / (den_g + eps))

        else:
            # Role-aware G_p 更新
            ph_hat = xp.zeros_like(X_ph, dtype=np.float32)
            for r in range(self.n_roles):
                ph_hat += self.G_p_roles[r] @ self.H_h_roles[r].T

            for r in range(self.n_roles):
                num_g = xp.zeros_like(self.G_p_roles[r])
                den_g = xp.zeros_like(self.G_p_roles[r])

                if self.sw["ph"]:
                    num_g += X_ph @ self.H_h_roles[r]
                    den_g += ph_hat @ self.H_h_roles[r]

                if self.sw["ps"]:
                    num_g += self.alpha * (X_ps @ self.H_s)
                    den_g += self.alpha * (self.G_p_roles[r] @ (self.H_s.T @ self.H_s))

                if self.sw.get("pair", False) and X_pair is not None and self.H_pair is not None:
                    num_g += self.beta_pair * (X_pair @ self.H_pair)
                    den_g += self.beta_pair * (self.G_p_roles[r] @ (self.H_pair.T @ self.H_pair))

                if self.sw["l1"]:
                    den_g += self.gamma_g

                self.G_p_roles[r] = self.G_p_roles[r] * (num_g / (den_g + eps))
            self.G_p = xp.sum(self.G_p_roles, axis=0)

        # =================================================================
        # Update H_h
        # =================================================================
        if not (self.role_aware and self.H_h_roles is not None):
            num_h = xp.zeros_like(self.H_h)
            den_h = xp.zeros_like(self.H_h)

            if self.sw["ph"]:
                num_h += X_ph.T @ self.G_p                      # X^T G
                den_h += self.H_h @ (self.G_p.T @ self.G_p)    # H G^T G

            # 图正则: L = L_pos - L_neg
            if self.sw["graph_h"] and L_h is not None:
                den_h += self.lambda_h * (L_h_pos @ self.H_h)
                num_h += self.lambda_h * (L_h_neg @ self.H_h)

            # 超图 Laplacian 正则 (处理方式同图正则)
            if self.sw["hyper_h"] and L_hyper_h is not None:
                Lh_pos, Lh_neg = self._split_sparse_pos_neg(L_hyper_h)
                den_h += self.lambda_hyper * (Lh_pos @ self.H_h)
                num_h += self.lambda_hyper * (Lh_neg @ self.H_h)

            # 禁忌约束: C_hh 非负 → 全进分母
            if self.sw["contra"] and C_hh is not None:
                den_h += self.rho * (C_hh @ self.H_h)

            # 知识耦合: ||K_hs - H_h H_s^T||^2
            if self.sw["know_hs"] and K_hs is not None:
                num_h += self.lambda_know * (K_hs @ self.H_s)
                den_h += self.lambda_know * (H_h_eff @ (self.H_s.T @ self.H_s))

            if self.sw["l1"]:
                den_h += self.gamma_h

            self.H_h = self.H_h * (num_h / (den_h + eps))

        else:
            # Role-aware H_h 更新
            ph_hat = xp.zeros_like(X_ph, dtype=np.float32)
            for r in range(self.n_roles):
                ph_hat += self.G_p_roles[r] @ self.H_h_roles[r].T

            for r in range(self.n_roles):
                num_h = xp.zeros_like(self.H_h_roles[r])
                den_h = xp.zeros_like(self.H_h_roles[r])

                if self.sw["ph"]:
                    num_h += X_ph.T @ self.G_p_roles[r]
                    den_h += ph_hat.T @ self.G_p_roles[r]

                if self.sw["graph_h"] and L_h is not None:
                    den_h += self.lambda_h * (L_h_pos @ self.H_h_roles[r])
                    num_h += self.lambda_h * (L_h_neg @ self.H_h_roles[r])

                if self.sw["contra"] and C_hh is not None:
                    den_h += self.rho * (C_hh @ self.H_h_roles[r])

                if self.sw["know_hs"] and K_hs is not None:
                    num_h += self.lambda_know * (K_hs @ self.H_s)
                    den_h += self.lambda_know * (self.H_h_roles[r] @ (self.H_s.T @ self.H_s))

                if self.role_exclusive > 0:
                    for r2 in range(self.n_roles):
                        if r2 != r:
                            den_h += self.role_exclusive * self.H_h_roles[r2]

                if self.sw["l1"]:
                    den_h += self.gamma_h

                self.H_h_roles[r] = self.H_h_roles[r] * (num_h / (den_h + eps))
            self.H_h = xp.sum(self.H_h_roles, axis=0)

        # =================================================================
        # Update H_s
        # =================================================================
        num_s = xp.zeros_like(self.H_s)
        den_s = xp.zeros_like(self.H_s)

        if self.sw["ps"]:
            num_s += self.alpha * (X_ps.T @ self.G_p)
            den_s += self.alpha * (self.H_s @ (self.G_p.T @ self.G_p))

        if self.sw["graph_s"] and L_s is not None:
            den_s += self.lambda_s * (L_s_pos @ self.H_s)
            num_s += self.lambda_s * (L_s_neg @ self.H_s)

        H_h_eff = self._effective_Hh()
        if self.sw["know_hs"] and K_hs is not None:
            num_s += self.lambda_know * (K_hs.T @ H_h_eff)
            den_s += self.lambda_know * (self.H_s @ (H_h_eff.T @ H_h_eff))

        if self.sw["l1"]:
            den_s += self.gamma_s

        self.H_s = self.H_s * (num_s / (den_s + eps))

        # =================================================================
        # Update D_h
        # =================================================================
        if self.sw["pd"] and X_pd is not None and M_pd is not None:
            M2 = M_pd * M_pd
            num_d = self.beta * ((M2 * X_pd).T @ self.G_p)
            den_d = self.beta * ((M2 * (self.G_p @ self.D_h.T)).T @ self.G_p)
            if self.sw["l1"]:
                den_d += self.gamma_d
            self.D_h = self.D_h * (num_d / (den_d + eps))

        # =================================================================
        # Update H_pair
        # =================================================================
        if self.sw.get("pair", False) and X_pair is not None and self.H_pair is not None:
            GtG = self.G_p.T @ self.G_p
            num_p = self.beta_pair * (X_pair.T @ self.G_p)
            den_p = self.beta_pair * (self.H_pair @ GtG)
            self.H_pair = self.H_pair * (num_p / (den_p + eps))

        # =================================================================
        # 列归一化 (可选)
        # =================================================================
        if self.normalize_columns:
            self.H_h, self.G_p = self._normalize_H_columns(self.H_h, self.G_p)
            self.H_s, self.G_p = self._normalize_H_columns(self.H_s, self.G_p)

        # --- Compute loss after update ---
        return self.compute_loss(X_ph, X_ps, X_pd, M_pd, X_pair, L_h, L_s, C_hh, L_hyper_h, K_hs)

    # -----------------------------------------------------------------
    # 混合更新规则 (G_p/H_h: MUR, H_s: PGD)
    # -----------------------------------------------------------------

    def fit_step_hybrid(
        self,
        X_ph: np.ndarray,
        X_ps: np.ndarray,
        X_pd: Optional[np.ndarray] = None,
        M_pd: Optional[np.ndarray] = None,
        X_pair: Optional[np.ndarray] = None,
        L_h: Optional[sp.csr_matrix] = None,
        L_s: Optional[sp.csr_matrix] = None,
        C_hh: Optional[sp.csr_matrix] = None,
        L_hyper_h: Optional[sp.csr_matrix] = None,
        K_hs: Optional[np.ndarray] = None,
    ) -> LossComponents:
        """混合更新: G_p/H_h 用 MUR (高 MAP), H_s 用 PGD (稳 PPL)。

        MUR 天然保非负、收敛快, 但会让 H_s 过度稀疏导致 sym_ppl 崩溃。
        Hybrid 只对 G_p/H_h 用 MUR 获取 MAP 增益, H_s 保留 PGD 的
        小步梯度更新以维持概率结构。
        """
        xp = self.xp
        eps = 1e-10
        lr = self.lr
        clip = self.grad_clip

        L_h_pos, L_h_neg = self._split_sparse_pos_neg(L_h)
        H_h_eff = self._effective_Hh()

        # =================================================================
        # Update G_p — MUR
        # =================================================================
        if not (self.role_aware and self.G_p_roles is not None):
            num_g = xp.zeros_like(self.G_p)
            den_g = xp.zeros_like(self.G_p)

            if self.sw["ph"]:
                num_g += X_ph @ self.H_h
                den_g += self.G_p @ (self.H_h.T @ self.H_h)

            if self.sw["ps"]:
                num_g += self.alpha * (X_ps @ self.H_s)
                den_g += self.alpha * (self.G_p @ (self.H_s.T @ self.H_s))

            if self.sw.get("pair", False) and X_pair is not None and self.H_pair is not None:
                num_g += self.beta_pair * (X_pair @ self.H_pair)
                den_g += self.beta_pair * (self.G_p @ (self.H_pair.T @ self.H_pair))

            if self.sw["pd"] and X_pd is not None and M_pd is not None:
                M2 = M_pd * M_pd
                num_g += self.beta * ((M2 * X_pd) @ self.D_h)
                den_g += self.beta * ((M2 * (self.G_p @ self.D_h.T)) @ self.D_h)

            if self.sw["l1"]:
                den_g += self.gamma_g

            self.G_p = self.G_p * (num_g / (den_g + eps))
        else:
            # Role-aware G_p — MUR
            ph_hat = xp.zeros_like(X_ph, dtype=np.float32)
            for r in range(self.n_roles):
                ph_hat += self.G_p_roles[r] @ self.H_h_roles[r].T
            for r in range(self.n_roles):
                num_g = xp.zeros_like(self.G_p_roles[r])
                den_g = xp.zeros_like(self.G_p_roles[r])
                if self.sw["ph"]:
                    num_g += X_ph @ self.H_h_roles[r]
                    den_g += ph_hat @ self.H_h_roles[r]
                if self.sw["ps"]:
                    num_g += self.alpha * (X_ps @ self.H_s)
                    den_g += self.alpha * (self.G_p_roles[r] @ (self.H_s.T @ self.H_s))
                if self.sw.get("pair", False) and X_pair is not None and self.H_pair is not None:
                    num_g += self.beta_pair * (X_pair @ self.H_pair)
                    den_g += self.beta_pair * (self.G_p_roles[r] @ (self.H_pair.T @ self.H_pair))
                if self.sw["l1"]:
                    den_g += self.gamma_g
                self.G_p_roles[r] = self.G_p_roles[r] * (num_g / (den_g + eps))
            self.G_p = xp.sum(self.G_p_roles, axis=0)

        # =================================================================
        # Update H_h — MUR
        # =================================================================
        if not (self.role_aware and self.H_h_roles is not None):
            num_h = xp.zeros_like(self.H_h)
            den_h = xp.zeros_like(self.H_h)
            if self.sw["ph"]:
                num_h += X_ph.T @ self.G_p
                den_h += self.H_h @ (self.G_p.T @ self.G_p)
            if self.sw["graph_h"] and L_h is not None:
                den_h += self.lambda_h * (L_h_pos @ self.H_h)
                num_h += self.lambda_h * (L_h_neg @ self.H_h)
            if self.sw["hyper_h"] and L_hyper_h is not None:
                Lh_pos, Lh_neg = self._split_sparse_pos_neg(L_hyper_h)
                den_h += self.lambda_hyper * (Lh_pos @ self.H_h)
                num_h += self.lambda_hyper * (Lh_neg @ self.H_h)
            if self.sw["contra"] and C_hh is not None:
                den_h += self.rho * (C_hh @ self.H_h)
            H_h_eff = self._effective_Hh()
            if self.sw["know_hs"] and K_hs is not None:
                num_h += self.lambda_know * (K_hs @ self.H_s)
                den_h += self.lambda_know * (H_h_eff @ (self.H_s.T @ self.H_s))
            if self.sw["l1"]:
                den_h += self.gamma_h
            self.H_h = self.H_h * (num_h / (den_h + eps))
        else:
            # Role-aware H_h — MUR
            ph_hat = xp.zeros_like(X_ph, dtype=np.float32)
            for r in range(self.n_roles):
                ph_hat += self.G_p_roles[r] @ self.H_h_roles[r].T
            for r in range(self.n_roles):
                num_h = xp.zeros_like(self.H_h_roles[r])
                den_h = xp.zeros_like(self.H_h_roles[r])
                if self.sw["ph"]:
                    num_h += X_ph.T @ self.G_p_roles[r]
                    den_h += ph_hat.T @ self.G_p_roles[r]
                if self.sw["graph_h"] and L_h is not None:
                    den_h += self.lambda_h * (L_h_pos @ self.H_h_roles[r])
                    num_h += self.lambda_h * (L_h_neg @ self.H_h_roles[r])
                if self.sw["contra"] and C_hh is not None:
                    den_h += self.rho * (C_hh @ self.H_h_roles[r])
                if self.sw["know_hs"] and K_hs is not None:
                    num_h += self.lambda_know * (K_hs @ self.H_s)
                    den_h += self.lambda_know * (self.H_h_roles[r] @ (self.H_s.T @ self.H_s))
                if self.role_exclusive > 0:
                    for r2 in range(self.n_roles):
                        if r2 != r:
                            den_h += self.role_exclusive * self.H_h_roles[r2]
                if self.sw["l1"]:
                    den_h += self.gamma_h
                self.H_h_roles[r] = self.H_h_roles[r] * (num_h / (den_h + eps))
            self.H_h = xp.sum(self.H_h_roles, axis=0)

        # =================================================================
        # Update H_s — PGD (保持概率结构, 稳定 sym_ppl)
        # 支持多轮内迭代让 H_s 追上 MUR 更新的 G_p
        # =================================================================
        n_inner = getattr(self, 'inner_steps_hs', 1)
        for _ in range(n_inner):
            grad_s = self._clip_grad(self._grad_Hs(X_ps, L_s, K_hs), clip)
            self.H_s = self._proj_step(self.H_s, grad_s, lr)
            if self.sw["l1"]:
                self.H_s = self._prox_l1(self.H_s, self.gamma_s, lr)

        # =================================================================
        # Update D_h — MUR
        # =================================================================
        if self.sw["pd"] and X_pd is not None and M_pd is not None:
            M2 = M_pd * M_pd
            num_d = self.beta * ((M2 * X_pd).T @ self.G_p)
            den_d = self.beta * ((M2 * (self.G_p @ self.D_h.T)).T @ self.G_p)
            if self.sw["l1"]:
                den_d += self.gamma_d
            self.D_h = self.D_h * (num_d / (den_d + eps))

        # =================================================================
        # Update H_pair — MUR
        # =================================================================
        if self.sw.get("pair", False) and X_pair is not None and self.H_pair is not None:
            GtG = self.G_p.T @ self.G_p
            num_p = self.beta_pair * (X_pair.T @ self.G_p)
            den_p = self.beta_pair * (self.H_pair @ GtG)
            self.H_pair = self.H_pair * (num_p / (den_p + eps))

        # 列归一化
        if self.normalize_columns:
            self.H_h, self.G_p = self._normalize_H_columns(self.H_h, self.G_p)
            self.H_s, self.G_p = self._normalize_H_columns(self.H_s, self.G_p)

        return self.compute_loss(X_ph, X_ps, X_pd, M_pd, X_pair, L_h, L_s, C_hh, L_hyper_h, K_hs)
