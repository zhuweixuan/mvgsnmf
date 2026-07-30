"""
配置驱动的路径解析。

优先级: YAML data_root > 环境变量 GSNMF_DATA_ROOT > 内置默认值
只定义相对文件名，运行时拼接 data_root。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


# 默认的相对文件名（相对于 data_root）
_DEFAULT_FILES: Dict[str, str] = {
    "herb_presence":     "matrix_from_mappings_herb_presence.csv",
    "symptom_presence":  "matrix_from_mappings_symptom.csv",
    "herb_dosage":       "matrix_from_mappings_herb_g.csv",
    "herb_category":     "herb_category_onehot.csv",
    "herb_features":     "herb_feature_matrix_793.csv",
    "symptom_features":  "symptom_feature_matrix.csv",
    "herb_cooc_pairs":   "herb_cooccurrence_pairs.csv",
    "symptom_cooc_pairs": "symptom_cooccurrence_pairs.csv",
    "herb_mutex_pairs":  "herb_mutual_exclusion_pairs_index_pairs.csv",
    "symptom_herb_knowledge": "symptom_herb_tcm_mesh.csv",
    "herb_alias": "herb_alias_user.csv",
}

_DEFAULT_DATA_ROOT = "data"


@dataclass(frozen=True)
class DatasetPaths:
    """所有数据文件的绝对路径，从配置解析得到。"""
    herb_presence_csv: str
    symptom_presence_csv: str
    herb_dosage_csv: str
    herb_category_csv: str
    herb_features_csv: str
    symptom_features_csv: str
    herb_cooc_pairs_csv: str
    symptom_cooc_pairs_csv: str
    herb_mutex_pairs_csv: str
    symptom_herb_knowledge_csv: Optional[str] = None
    herb_alias_csv: Optional[str] = None

    def validate(self, require_dosage: bool = True) -> None:
        """检查所有文件是否存在。"""
        required = [
            self.herb_presence_csv, self.symptom_presence_csv,
            self.herb_category_csv,
            self.herb_features_csv, self.symptom_features_csv,
            self.herb_cooc_pairs_csv, self.symptom_cooc_pairs_csv,
            self.herb_mutex_pairs_csv,
        ]
        if require_dosage:
            required.append(self.herb_dosage_csv)
        for fname in required:
            if not os.path.isfile(fname):
                raise FileNotFoundError(f"数据文件不存在: {fname}")


def resolve_paths(
    data_root: Optional[str] = None,
    file_overrides: Optional[Dict[str, str]] = None,
) -> DatasetPaths:
    """根据 data_root(或环境变量) + 文件名映射 → DatasetPaths。"""
    # 1. 确定 data_root
    if data_root is None:
        data_root = os.environ.get("GSNMF_DATA_ROOT", _DEFAULT_DATA_ROOT)
    root = Path(data_root)

    # 2. 合并文件名
    files = dict(_DEFAULT_FILES)
    if file_overrides:
        files.update(file_overrides)

    # 3. 拼接并返回
    def _abs(key: str) -> str:
        return str(root / files[key])

    return DatasetPaths(
        herb_presence_csv=_abs("herb_presence"),
        symptom_presence_csv=_abs("symptom_presence"),
        herb_dosage_csv=_abs("herb_dosage"),
        herb_category_csv=_abs("herb_category"),
        herb_features_csv=_abs("herb_features"),
        symptom_features_csv=_abs("symptom_features"),
        herb_cooc_pairs_csv=_abs("herb_cooc_pairs"),
        symptom_cooc_pairs_csv=_abs("symptom_cooc_pairs"),
        herb_mutex_pairs_csv=_abs("herb_mutex_pairs"),
        symptom_herb_knowledge_csv=_abs("symptom_herb_knowledge"),
        herb_alias_csv=_abs("herb_alias"),
    )


@dataclass(frozen=True)
class ProjectPaths:
    """项目级目录。"""
    project_root: str = str(Path(__file__).resolve().parent.parent)
    artifacts_dir: str = ""

    def __post_init__(self):
        if not self.artifacts_dir:
            object.__setattr__(
                self, "artifacts_dir",
                str(Path(self.project_root) / "artifacts"),
            )
