"""
MV-GSNMTF: Multi-View Graph-Regularized Sparse Nonnegative Matrix Tri-Factorization

学习四组核心隐变量:
  G_p  (P × K) : 处方主题矩阵 (共享)
  H_h  (H × K) : 药材主题载荷
  H_s  (S × K) : 症状主题载荷
  D_h  (H × K) : 剂量风格载荷
"""

__version__ = "0.1.0"
