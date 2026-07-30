#!/usr/bin/env python3
"""
Paired Wilcoxon Signed-Rank Test for MV-GSNMTF Main Experiments.

Statistical significance analysis across matched (K, seed) settings.
Each paired observation = same K + same seed for two models.
8 topic counts × 5 seeds = 40 paired observations per comparison.

Two hypothesis families with separate Holm–Bonferroni correction:
  Family A: Domain-prior ablation (6 ablation configs)
  Family B: Early vs late coupling (baseline comparisons)

Outputs:
  - artifacts/revision_significance/wilcoxon_results.json
  - artifacts/revision_significance/wilcoxon_family_a.csv
  - artifacts/revision_significance/wilcoxon_family_b.csv
  - artifacts/revision_significance/wilcoxon_main_table.tex
  - artifacts/revision_significance/wilcoxon_appendix_table.tex

Usage:
  python scripts/revision/run_wilcoxon_significance.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "paper_results" / "significance"

# ── Practical significance thresholds ─────────────────────────────────
PRACTICAL_THRESHOLDS = {
    "valid_map_avg": 0.005,
    "sym2herb_map": 0.005,
    "herb2sym_map": 0.005,
    "symptom_pred_ppl_em_prob": 2.0,
    "herb_pred_ppl_em_prob": 5.0,
    "topic_coherence": 0.02,
}

# ── Model name mapping ───────────────────────────────────────────────
# Ablation stage → short name for paper
ABLATION_STAGE_TO_NAME = {
    "00_active_graph_h__graph_s__l1__pair__know_hs": "C_full",
    "01_active_graph_h__graph_s__l1__pair": "C_-K",
    "02_active_graph_h__graph_s__l1": "C_-KP",
    "03_active_graph_h__graph_s": "C_graph",
    "04_active_graph_h": "C_herb",
    "05_recon_only_ph_ps_nonneg": "C_min",
}

# Baseline variant → short name for paper
BASELINE_VARIANT_TO_NAME = {
    "recon_only_mvgsnmf": "MV-NMTF",
    "vanilla_nmf_fro": "Vanilla NMF",
    "indep_nmf_procrustes_bridge": "Indep+Procrustes",
    "sparse_nmf_l1": "Sparse NMF",
    "gnmf_graph": "GNMF",
}

# ── Metric metadata ──────────────────────────────────────────────────
METRIC_META = {
    "valid_map_avg": {"higher_is_better": True, "label": "MAP$_{\\text{avg}}$", "fmt": ".4f"},
    "sym2herb_map": {"higher_is_better": True, "label": "S$\\to$H MAP", "fmt": ".4f"},
    "herb2sym_map": {"higher_is_better": True, "label": "H$\\to$S MAP", "fmt": ".4f"},
    "symptom_pred_ppl_em_prob": {"higher_is_better": False, "label": "Sym PPL", "fmt": ".1f"},
    "herb_pred_ppl_em_prob": {"higher_is_better": False, "label": "Herb PPL", "fmt": ".1f"},
    "topic_coherence": {"higher_is_better": True, "label": "Coherence", "fmt": ".4f"},
}


# =====================================================================
# Core statistical functions
# =====================================================================

def rank_biserial_effect(d: np.ndarray) -> float:
    """Rank-biserial correlation as effect size for Wilcoxon test.

    r = (R+ - R-) / (R+ + R-)
    Ranges from -1 to 1. |r| > 0.3 is medium, > 0.5 is large.
    """
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    r_pos = ranks[d > 0].sum()
    r_neg = ranks[d < 0].sum()
    total = r_pos + r_neg
    if total == 0:
        return 0.0
    return float((r_pos - r_neg) / total)


def hodges_lehmann_ci(d: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """Bootstrap 95% CI for the median of paired differences.

    Uses percentile bootstrap (10000 resamples) as a robust fallback
    since exact Hodges-Lehmann CI tables for n=40 are cumbersome.
    """
    d = np.asarray(d, dtype=float)
    rng = np.random.RandomState(42)
    n_boot = 10000
    medians = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(d, size=len(d), replace=True)
        medians[b] = np.median(sample)
    lo = np.percentile(medians, 100 * alpha / 2)
    hi = np.percentile(medians, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def paired_wilcoxon_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    higher_is_better: bool = True,
) -> Dict:
    """Run paired Wilcoxon signed-rank test.

    Convention: positive d means model A is better.
    - higher_is_better=True:  d = A - B
    - higher_is_better=False: d = B - A  (so positive = A has lower PPL = better)

    Returns dict with all required statistics.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)

    if higher_is_better:
        d = a - b
    else:
        d = b - a

    n_total = len(d)
    n_nonzero = int(np.sum(d != 0))
    n_positive = int(np.sum(d > 0))
    n_negative = int(np.sum(d < 0))
    n_zero = int(np.sum(d == 0))

    # Wilcoxon test
    if n_nonzero < 2:
        # Cannot run test with fewer than 2 non-zero differences
        return {
            "n": n_total,
            "n_nonzero": n_nonzero,
            "win_rate": f"{n_positive}/{n_total}",
            "mean_delta": float(np.mean(d)),
            "median_delta": float(np.median(d)),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "W": float("nan"),
            "p": 1.0,
            "rank_biserial": 0.0,
            "interpretation": "insufficient_data",
        }

    try:
        stat, p = wilcoxon(
            d,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="auto",
        )
    except ValueError:
        stat, p = float("nan"), 1.0

    ci_lo, ci_hi = hodges_lehmann_ci(d)
    rb = rank_biserial_effect(d)

    return {
        "n": n_total,
        "n_nonzero": n_nonzero,
        "win_rate": f"{n_positive}/{n_total}",
        "mean_delta": float(np.mean(d)),
        "median_delta": float(np.median(d)),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "W": float(stat),
        "p": float(p),
        "rank_biserial": rb,
        "interpretation": "",  # filled after Holm correction
    }


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Holm–Bonferroni step-down correction for multiple comparisons."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(p * (n - rank), 1.0)
        cummax = max(cummax, adj_p)
        adjusted[orig_idx] = cummax
    return adjusted


def interpret(
    holm_p: float,
    median_delta: float,
    metric: str,
    alpha: float = 0.05,
) -> str:
    """Interpret a test result with both statistical and practical significance."""
    threshold = PRACTICAL_THRESHOLDS.get(metric, 0.005)
    stat_sig = holm_p < alpha
    pract_sig = abs(median_delta) >= threshold

    if stat_sig and pract_sig:
        direction = "A better" if median_delta > 0 else "B better"
        return f"significant ({direction})"
    elif stat_sig and not pract_sig:
        return "stat. significant but negligible"
    elif not stat_sig and not pract_sig:
        return "non-significant & negligible"
    else:
        return "non-significant"


# =====================================================================
# Data loading
# =====================================================================

def load_ablation_data() -> pd.DataFrame:
    """Load ablation per-(K, seed) metrics."""
    csv_path = PROJECT_ROOT / "paper_results" / "ablation" / "all_metrics_by_k_full_list.csv"
    if not csv_path.exists():
        logger.error("Ablation CSV not found: %s", csv_path)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    # Map stage names to short names
    df["model"] = df["stage"].map(ABLATION_STAGE_TO_NAME)
    logger.info("Loaded ablation data: %d rows, models=%s", len(df), df["model"].unique().tolist())
    return df


def load_baseline_data() -> pd.DataFrame:
    """Load baseline per-(K, seed) metrics."""
    csv_path = PROJECT_ROOT / "paper_results" / "ppl_baselines" / "ppl_compare_metrics_by_k_requested_fields.csv"
    if not csv_path.exists():
        logger.error("Baseline CSV not found: %s", csv_path)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    df["model"] = df["variant"].map(BASELINE_VARIANT_TO_NAME)
    logger.info("Loaded baseline data: %d rows, models=%s", len(df), df["model"].unique().tolist())
    return df


def get_paired_values(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str,
    model_col: str = "model",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract matched (K, seed) paired values for two models."""
    a = df[df[model_col] == model_a][["K", "seed", metric]].copy()
    b = df[df[model_col] == model_b][["K", "seed", metric]].copy()

    merged = a.merge(b, on=["K", "seed"], suffixes=("_a", "_b")).dropna()

    if len(merged) == 0:
        logger.warning("No matched pairs for %s vs %s on %s", model_a, model_b, metric)
        return np.array([]), np.array([])

    return merged[f"{metric}_a"].values, merged[f"{metric}_b"].values


# =====================================================================
# Hypothesis families
# =====================================================================

def define_family_a() -> List[Dict]:
    """Family A: Domain-prior ablation comparisons."""
    return [
        {"model_a": "C_full", "model_b": "C_-K", "metric": "valid_map_avg",
         "claim": "Knowledge coupling useful?"},
        {"model_a": "C_-K", "model_b": "C_-KP", "metric": "valid_map_avg",
         "claim": "Herb-pair view useful?"},
        {"model_a": "C_graph", "model_b": "C_-KP", "metric": "valid_map_avg",
         "claim": "L1 harmful (MAP)?"},
        {"model_a": "C_graph", "model_b": "C_-KP", "metric": "herb2sym_map",
         "claim": "L1 harmful (H→S)?"},
        {"model_a": "C_graph", "model_b": "C_min", "metric": "valid_map_avg",
         "claim": "Graph improves ranking?"},
        {"model_a": "C_graph", "model_b": "C_min", "metric": "symptom_pred_ppl_em_prob",
         "claim": "Graph improves Sym PPL?"},
        {"model_a": "C_graph", "model_b": "C_min", "metric": "herb_pred_ppl_em_prob",
         "claim": "Graph improves Herb PPL?"},
        {"model_a": "C_herb", "model_b": "C_graph", "metric": "valid_map_avg",
         "claim": "Unilateral graph destabilizing?"},
        {"model_a": "C_herb", "model_b": "C_min", "metric": "valid_map_avg",
         "claim": "Herb-only graph vs minimal?"},
    ]


def define_family_b() -> List[Dict]:
    """Family B: Early vs late coupling (baseline) comparisons."""
    return [
        {"model_a": "MV-NMTF", "model_b": "Indep+Procrustes", "metric": "valid_map_avg",
         "claim": "Early binding significant?"},
        {"model_a": "MV-NMTF", "model_b": "Indep+Procrustes", "metric": "sym2herb_map",
         "claim": "Early binding (S→H)?"},
        {"model_a": "Vanilla NMF", "model_b": "Indep+Procrustes", "metric": "valid_map_avg",
         "claim": "Vanilla vs late coupling?"},
        {"model_a": "GNMF", "model_b": "Indep+Procrustes", "metric": "valid_map_avg",
         "claim": "GNMF vs late coupling?"},
        {"model_a": "Vanilla NMF", "model_b": "MV-NMTF", "metric": "valid_map_avg",
         "claim": "Vanilla vs MV-NMTF?"},
        {"model_a": "GNMF", "model_b": "MV-NMTF", "metric": "valid_map_avg",
         "claim": "GNMF vs MV-NMTF?"},
        {"model_a": "Vanilla NMF", "model_b": "GNMF", "metric": "valid_map_avg",
         "claim": "Vanilla vs GNMF?"},
    ]


def run_family(
    family_name: str,
    comparisons: List[Dict],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run all tests in a hypothesis family, apply Holm correction."""
    results = []

    for comp in comparisons:
        model_a = comp["model_a"]
        model_b = comp["model_b"]
        metric = comp["metric"]
        meta = METRIC_META[metric]

        vals_a, vals_b = get_paired_values(df, model_a, model_b, metric)

        if len(vals_a) == 0:
            logger.warning("Skipping %s vs %s (%s): no data", model_a, model_b, metric)
            continue

        result = paired_wilcoxon_test(vals_a, vals_b, meta["higher_is_better"])
        result["family"] = family_name
        result["comparison"] = f"{model_a} vs {model_b}"
        result["model_a"] = model_a
        result["model_b"] = model_b
        result["metric"] = metric
        result["metric_label"] = meta["label"]
        result["claim"] = comp["claim"]
        results.append(result)

    if not results:
        return pd.DataFrame()

    # Holm–Bonferroni within family
    raw_ps = [r["p"] for r in results]
    holm_ps = holm_bonferroni(raw_ps)

    for r, hp in zip(results, holm_ps):
        r["holm_p"] = hp
        r["interpretation"] = interpret(hp, r["median_delta"], r["metric"])

    return pd.DataFrame(results)


# =====================================================================
# LaTeX output
# =====================================================================

def to_pval_str(p: float) -> str:
    """Format p-value for LaTeX."""
    if np.isnan(p):
        return "---"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def to_sig_marker(holm_p: float) -> str:
    if np.isnan(holm_p):
        return ""
    if holm_p < 0.001:
        return "$^{***}$"
    if holm_p < 0.01:
        return "$^{**}$"
    if holm_p < 0.05:
        return "$^{*}$"
    return ""


def generate_main_latex(df_a: pd.DataFrame, df_b: pd.DataFrame) -> str:
    """Generate compact main-text LaTeX table."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Paired Wilcoxon Signed-Rank Tests for Main Experimental Claims. "
        r"Each test uses $n{=}40$ matched (K, seed) observations. "
        r"$\Delta{>}0$ means model A is better. "
        r"$^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$ (Holm-corrected).}",
        r"\label{tab:wilcoxon}",
        r"\small",
        r"\begin{tabular}{llcrrrrrrl}",
        r"\toprule",
        r"Comparison & Metric & $n$ & Med.\,$\Delta$ & 95\% CI & $W$ & $p$ & Holm $p$ & $r_{\text{rb}}$ & Win Rate \\",
        r"\midrule",
        r"\multicolumn{10}{l}{\textit{Family A: Domain-Prior Ablation}} \\",
        r"\midrule",
    ]

    for _, row in df_a.iterrows():
        sig = to_sig_marker(row["holm_p"])
        meta = METRIC_META[row["metric"]]
        fmt = meta["fmt"]
        ci_str = f"[{row['ci_lo']:{fmt}}, {row['ci_hi']:{fmt}}]"
        lines.append(
            f"{row['comparison']}{sig} & {row['metric_label']} & "
            f"{row['n']} & {row['median_delta']:{fmt}} & {ci_str} & "
            f"{row['W']:.0f} & {to_pval_str(row['p'])} & {to_pval_str(row['holm_p'])} & "
            f"{row['rank_biserial']:.2f} & {row['win_rate']} \\\\"
        )

    lines += [
        r"\midrule",
        r"\multicolumn{10}{l}{\textit{Family B: Early vs Late Coupling}} \\",
        r"\midrule",
    ]

    for _, row in df_b.iterrows():
        sig = to_sig_marker(row["holm_p"])
        meta = METRIC_META[row["metric"]]
        fmt = meta["fmt"]
        ci_str = f"[{row['ci_lo']:{fmt}}, {row['ci_hi']:{fmt}}]"
        lines.append(
            f"{row['comparison']}{sig} & {row['metric_label']} & "
            f"{row['n']} & {row['median_delta']:{fmt}} & {ci_str} & "
            f"{row['W']:.0f} & {to_pval_str(row['p'])} & {to_pval_str(row['holm_p'])} & "
            f"{row['rank_biserial']:.2f} & {row['win_rate']} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def generate_appendix_latex(df_a: pd.DataFrame, df_b: pd.DataFrame) -> str:
    """Generate extended appendix table with interpretation column."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Extended Wilcoxon Significance Results (Appendix). "
        r"Includes interpretation based on both statistical ($\alpha{=}0.05$, Holm-corrected) "
        r"and practical significance thresholds.}",
        r"\label{tab:wilcoxon_appendix}",
        r"\footnotesize",
        r"\begin{tabular}{llcrrrrrl}",
        r"\toprule",
        r"Claim & Metric & $n$ & Mean\,$\Delta$ & Med.\,$\Delta$ & Holm $p$ & $r_{\text{rb}}$ & Win & Interpretation \\",
        r"\midrule",
    ]

    all_df = pd.concat([df_a, df_b], ignore_index=True)
    for _, row in all_df.iterrows():
        meta = METRIC_META[row["metric"]]
        fmt = meta["fmt"]
        interp = row["interpretation"].replace("&", r"\&")
        lines.append(
            f"{row['claim']} & {row['metric_label']} & "
            f"{row['n']} & {row['mean_delta']:{fmt}} & {row['median_delta']:{fmt}} & "
            f"{to_pval_str(row['holm_p'])} & {row['rank_biserial']:.2f} & "
            f"{row['win_rate']} & {interp} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# =====================================================================
# Main
# =====================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df_ablation = load_ablation_data()
    df_baseline = load_baseline_data()

    # Family A: ablation comparisons (all from df_ablation)
    logger.info("=== Family A: Domain-Prior Ablation ===")
    family_a = define_family_a()
    results_a = run_family("A_ablation", family_a, df_ablation)

    # Family B: baseline comparisons (all from df_baseline)
    logger.info("=== Family B: Early vs Late Coupling ===")
    family_b = define_family_b()
    results_b = run_family("B_coupling", family_b, df_baseline)

    # Print summary
    print("\n" + "=" * 80)
    print("FAMILY A: Domain-Prior Ablation")
    print("=" * 80)
    if len(results_a) > 0:
        for _, row in results_a.iterrows():
            sig = "***" if row["holm_p"] < 0.001 else "**" if row["holm_p"] < 0.01 else "*" if row["holm_p"] < 0.05 else "ns"
            print(f"  [{sig:>3s}] {row['claim']:<40s} | "
                  f"Δ={row['median_delta']:+.4f} | "
                  f"p={row['p']:.4f} → Holm={row['holm_p']:.4f} | "
                  f"r={row['rank_biserial']:+.2f} | "
                  f"win={row['win_rate']}")

    print("\n" + "=" * 80)
    print("FAMILY B: Early vs Late Coupling")
    print("=" * 80)
    if len(results_b) > 0:
        for _, row in results_b.iterrows():
            sig = "***" if row["holm_p"] < 0.001 else "**" if row["holm_p"] < 0.01 else "*" if row["holm_p"] < 0.05 else "ns"
            print(f"  [{sig:>3s}] {row['claim']:<40s} | "
                  f"Δ={row['median_delta']:+.4f} | "
                  f"p={row['p']:.4f} → Holm={row['holm_p']:.4f} | "
                  f"r={row['rank_biserial']:+.2f} | "
                  f"win={row['win_rate']}")

    # Save CSV
    if len(results_a) > 0:
        csv_a = OUTPUT_DIR / "wilcoxon_family_a.csv"
        results_a.to_csv(csv_a, index=False)
        logger.info("Saved Family A: %s", csv_a)

    if len(results_b) > 0:
        csv_b = OUTPUT_DIR / "wilcoxon_family_b.csv"
        results_b.to_csv(csv_b, index=False)
        logger.info("Saved Family B: %s", csv_b)

    # Save JSON (all results)
    all_results = pd.concat([results_a, results_b], ignore_index=True)
    json_path = OUTPUT_DIR / "wilcoxon_results.json"
    all_results.to_json(json_path, orient="records", indent=2, force_ascii=False)
    logger.info("Saved JSON: %s", json_path)

    # Generate LaTeX
    if len(results_a) > 0 and len(results_b) > 0:
        tex_main = generate_main_latex(results_a, results_b)
        tex_main_path = OUTPUT_DIR / "wilcoxon_main_table.tex"
        tex_main_path.write_text(tex_main, encoding="utf-8")
        logger.info("Saved main LaTeX: %s", tex_main_path)

        tex_app = generate_appendix_latex(results_a, results_b)
        tex_app_path = OUTPUT_DIR / "wilcoxon_appendix_table.tex"
        tex_app_path.write_text(tex_app, encoding="utf-8")
        logger.info("Saved appendix LaTeX: %s", tex_app_path)

    # Summary statistics
    print(f"\n{'─' * 60}")
    print(f"Total comparisons: {len(all_results)}")
    print(f"  Family A: {len(results_a)}")
    print(f"  Family B: {len(results_b)}")
    n_sig = (all_results["holm_p"] < 0.05).sum() if len(all_results) > 0 else 0
    print(f"  Significant (Holm p < 0.05): {n_sig}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
