"""
计算后端: 自动选择 NumPy (CPU) 或 CuPy (GPU)。

用法:
    from gsnmf.backend import get_backend
    xp, xsp = get_backend("gpu")  # xp = cupy, xsp = cupyx.scipy.sparse
    xp, xsp = get_backend("cpu")  # xp = numpy, xsp = scipy.sparse

矩阵在 CPU ↔ GPU 之间转换:
    gpu_arr = to_device(cpu_arr, xp)  # numpy → cupy
    cpu_arr = to_numpy(gpu_arr)       # cupy → numpy
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import sparse as sp

logger = logging.getLogger(__name__)

_HAS_CUPY = None


def _configure_cupy_temp_dirs() -> None:
    """Use a dedicated NVRTC temp/cache directory on Windows.

    NVRTC may fail to open generated source files when its temporary path
    contains non-ASCII characters. A dedicated subdirectory also avoids slow
    scans in a crowded system temp root. Users can override the location with
    MVGSNMF_CUDA_TMP; on Windows that path should contain ASCII characters only.
    """
    if os.name != "nt":
        return

    base_override = os.environ.get("MVGSNMF_CUDA_TMP")
    base = (
        Path(base_override)
        if base_override
        else Path(tempfile.gettempdir()) / "mvgsnmf_cuda"
    )
    runtime_tmp = base / "tmp"
    cache_dir = base / "cache"
    try:
        runtime_tmp.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create CuPy temp directories under %s: %s", base, exc)
        return

    if not str(base).isascii():
        logger.warning(
            "CuPy temp path contains non-ASCII characters (%s). "
            "Set MVGSNMF_CUDA_TMP to an ASCII-only writable directory.",
            base,
        )

    os.environ["TEMP"] = str(runtime_tmp)
    os.environ["TMP"] = str(runtime_tmp)
    os.environ.setdefault("CUPY_CACHE_DIR", str(cache_dir))
    tempfile.tempdir = str(runtime_tmp)


def _check_cupy() -> bool:
    global _HAS_CUPY
    if _HAS_CUPY is None:
        try:
            import cupy  # noqa
            _HAS_CUPY = True
        except ImportError:
            _HAS_CUPY = False
    return _HAS_CUPY


def get_backend(device: str = "cpu") -> Tuple:
    """返回 (xp, xsp) 数组库和稀疏库。

    Parameters
    ----------
    device : "cpu" | "gpu"

    Returns
    -------
    xp  : numpy 或 cupy 模块
    xsp : scipy.sparse 或 cupyx.scipy.sparse 模块
    """
    if device == "gpu":
        _configure_cupy_temp_dirs()
        if not _check_cupy():
            logger.warning("CuPy 未安装，回退到 CPU。pip install cupy-cuda12x")
            return np, sp
        try:
            import cupy
            import cupyx.scipy.sparse as cusp
            # 运行时探测：触发一次最小 kernel 编译/执行，避免训练中途崩溃
            x = cupy.asarray([1.0], dtype=cupy.float32)
            y = x + 1.0
            _ = float(y.get()[0])
            logger.info("使用 GPU 后端 (CuPy, device=%s)", cupy.cuda.Device())
            return cupy, cusp
        except Exception as e:
            logger.warning("GPU 后端初始化失败，自动回退到 CPU。原因: %s", e)
            return np, sp
    else:
        return np, sp


def to_device(arr, xp):
    """将 numpy 数组/scipy 稀疏矩阵转到目标设备。"""
    if xp.__name__ == "cupy":
        import cupy
        import cupyx.scipy.sparse as cusp
        if sp.issparse(arr):
            return cusp.csr_matrix(arr)
        return cupy.asarray(arr)
    else:
        # 已在 CPU
        if sp.issparse(arr):
            return arr
        return np.asarray(arr)


def to_numpy(arr) -> np.ndarray:
    """将设备数组转回 numpy (CPU)。"""
    if hasattr(arr, 'get'):
        # CuPy array
        return arr.get()
    if sp.issparse(arr):
        return arr.toarray()
    if hasattr(arr, 'toarray'):
        # cupyx sparse
        return arr.get().toarray() if hasattr(arr, 'get') else arr.toarray()
    return np.asarray(arr)


def to_dense(arr, xp):
    """稀疏 → 稠密 (在当前设备上)。"""
    if hasattr(arr, 'toarray'):
        return xp.asarray(arr.toarray()) if xp.__name__ == "numpy" else arr.toarray()
    return xp.asarray(arr)
