#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a GPS-weighted causal analysis from continuous agent activity to adjusted reputation.

This copy treats non-ERC8004 other transactions as an observed confounder in the
GPS model.

Outputs:
    ERC8004/causal_2/agent_scores.csv
    ERC8004/causal_2/causal_results.csv
    ERC8004/causal_2/covariate_balance.csv
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as exc:
    raise SystemExit(
        "scikit-learn is not installed. Run:\n"
        "    pip install scikit-learn"
    ) from exc


# =========================
# 0. Paths and parameters
# =========================

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULT_DIR = REPO_ROOT / "causal_2"

AGENT_STATISTIC_CSV = DATA_DIR / "agent_statistic.csv"

DEFAULT_K_R = 5.0
MIN_SIGMA = 1e-8
GPS_WEIGHT_LOWER_Q = 0.01
GPS_WEIGHT_UPPER_Q = 0.99
END_BLOCK = 25277687

ACTIVITY_NAME = "identity_ecosystem_tx_other_tx_confounder"
ACTIVITY_COL = "activity_score"
OUTCOME_COL = "adjusted_reputation"
ACTIVITY_DESCRIPTION = "activity_score = log1p(identity_operation_count + ecosystem_operation_count)"
OUTCOME_DESCRIPTION = "adjusted_reputation = shrinkage-adjusted reputation * diversity_penalty"

GPS_COVARIATES = [
    "log_survival_blocks",
    "owner_agent_count",
    "log_other_operation_count",
]

BALANCE_COVARIATES = [
    "log_survival_blocks",
    "owner_agent_count",
    "log_other_operation_count",
]


# =========================
# 1. Statistical helpers
# =========================

def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    mu = weighted_mean(x, w)
    return float(np.sum(w * (x - mu) ** 2) / np.sum(w))


def weighted_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Compute weighted Pearson correlation for balance checks."""
    if weights is None:
        weights = np.ones(len(x), dtype=float)

    x_var = weighted_var(x, weights)
    y_var = weighted_var(y, weights)
    if x_var <= 0 or y_var <= 0:
        return 0.0

    x_mu = weighted_mean(x, weights)
    y_mu = weighted_mean(y, weights)
    cov = float(np.sum(weights * (x - x_mu) * (y - y_mu)) / np.sum(weights))
    return float(cov / math.sqrt(x_var * y_var))


def normal_pdf(x: np.ndarray, mean: np.ndarray | float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), MIN_SIGMA)
    z = (x - mean) / sigma
    return np.exp(-0.5 * z ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def weighted_ols(y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> Dict[str, np.ndarray | float]:
    """Fit weighted OLS with sandwich robust covariance."""
    x_design = np.column_stack([np.ones(len(y)), x])
    w = weights.reshape(-1, 1)
    xtwx = x_design.T @ (w * x_design)
    xtwy = x_design.T @ (weights * y)
    xtwx_inv = np.linalg.pinv(xtwx)
    coef = xtwx_inv @ xtwy

    residual = y - x_design @ coef
    meat = x_design.T @ ((weights ** 2 * residual ** 2).reshape(-1, 1) * x_design)
    cov = xtwx_inv @ meat @ xtwx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    y_bar = weighted_mean(y, weights)
    ss_res = float(np.sum(weights * residual ** 2))
    ss_tot = float(np.sum(weights * (y - y_bar) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "coef": coef,
        "se": se,
        "cov": cov,
        "r_squared": float(r_squared),
    }


def p_value_from_t(t_value: float) -> float:
    return math.erfc(abs(t_value) / math.sqrt(2.0)) if math.isfinite(t_value) else float("nan")


def format_p_value(p_value: float) -> str:
    if not math.isfinite(p_value):
        return "nan"
    return f"{p_value:.3e}" if p_value < 0.001 else f"{p_value:.4f}"


def effective_sample_size(weights: np.ndarray) -> float:
    return float((weights.sum() ** 2) / np.sum(weights ** 2))


# =========================
# 2. Data loading and feature construction
# =========================

def load_data(k_r: float = DEFAULT_K_R) -> pd.DataFrame:
    """Load the source CSV and build analysis variables."""
    df = pd.read_csv(AGENT_STATISTIC_CSV)
    required_cols = {
        "owner_wallet",
        "agent_wallet",
        "owner_agent_count",
        "identity_operation_count",
        "ecosystem_operation_count",
        "other_operation_count",
        "reputation",
        "feedback_count",
        "client_count",
        "block_stamp",
    }
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise SystemExit(
            "agent_statistic.csv is missing required fields: "
            + ", ".join(missing_cols)
            + ". Run ERC8004/scripts/main.py first."
        )

    for col in ["owner_wallet", "agent_wallet"]:
        df[col] = df[col].fillna("").astype(str).str.lower().str.strip()
    df["owner_agent_count"] = df["owner_agent_count"].fillna(1).astype(float)
    df["agent_wallet_missing"] = (df["agent_wallet"] == "").astype(float)
    df["owner_equals_agent_wallet"] = (
        (df["agent_wallet"] != "")
        & (df["owner_wallet"] == df["agent_wallet"])
    ).astype(float)

    identity_count = df["identity_operation_count"].astype(float).clip(lower=0.0)
    ecosystem_count = df["ecosystem_operation_count"].astype(float).clip(lower=0.0)
    other_count = df["other_operation_count"].astype(float).clip(lower=0.0)
    df["activity_score"] = np.log1p(identity_count + ecosystem_count)
    df["log_other_operation_count"] = np.log1p(other_count)

    global_mean = float(df["reputation"].mean())
    feedback_count = df["feedback_count"].astype(float)
    shrink_weight = feedback_count / (feedback_count + float(k_r))
    df["core_score"] = shrink_weight * df["reputation"].astype(float) + (
        1.0 - shrink_weight
    ) * global_mean

    numerator = np.log1p(df["client_count"].astype(float))
    denominator = np.log1p(feedback_count)
    df["diversity_penalty"] = np.sqrt(numerator / denominator)
    df["diversity_penalty"] = (
        df["diversity_penalty"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    df["adjusted_reputation"] = df["core_score"] * df["diversity_penalty"]
    df["adjusted_reputation_index"] = (df["adjusted_reputation"] / 100.0).clip(0.0, 1.0)

    block_stamp = df["block_stamp"].astype(float)
    df["survival_blocks"] = (float(END_BLOCK) - block_stamp).clip(lower=0.0)
    df["log_survival_blocks"] = np.log1p(df["survival_blocks"])
    df["owner_agent_count_sq"] = df["owner_agent_count"].astype(float) ** 2

    return df


# =========================
# 3. GPS weights
# =========================

def add_gps_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate stabilized generalized propensity score weights."""
    out = df.copy()
    x = out[GPS_COVARIATES].astype(float)
    activity = out[ACTIVITY_COL].astype(float).to_numpy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    treatment_model = LinearRegression()
    treatment_model.fit(x_scaled, activity)
    expected_activity = treatment_model.predict(x_scaled)

    residual = activity - expected_activity
    dof = max(len(activity) - len(GPS_COVARIATES) - 1, 1)
    conditional_sigma = math.sqrt(float(np.sum(residual ** 2) / dof))
    marginal_sigma = float(np.std(activity, ddof=1))
    marginal_mean = float(np.mean(activity))

    conditional_density = normal_pdf(activity, expected_activity, conditional_sigma)
    marginal_density = normal_pdf(activity, marginal_mean, marginal_sigma)
    gps_weight = marginal_density / np.clip(conditional_density, MIN_SIGMA, None)
    gps_weight = np.clip(
        gps_weight,
        float(np.quantile(gps_weight, GPS_WEIGHT_LOWER_Q)),
        float(np.quantile(gps_weight, GPS_WEIGHT_UPPER_Q)),
    )

    out["gps_expected_activity"] = expected_activity
    out["gps_conditional_density"] = conditional_density
    out["gps_marginal_density"] = marginal_density
    out["gps_weight"] = gps_weight
    out["gps_conditional_sigma"] = conditional_sigma
    out["gps_marginal_sigma"] = marginal_sigma
    out["gps_marginal_mean"] = marginal_mean
    out["gps_effective_sample_size"] = effective_sample_size(gps_weight)
    for cov_name, coef in zip(GPS_COVARIATES, treatment_model.coef_):
        out[f"gps_model_{cov_name}_coef"] = float(coef)
    out["gps_model_intercept"] = float(treatment_model.intercept_)
    return out


# =========================
# 4. Analyses
# =========================

def make_balance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Check GPS-weighted covariate balance."""
    activity = df[ACTIVITY_COL].astype(float).to_numpy()
    weights = df["gps_weight"].astype(float).to_numpy()

    rows = []
    for cov in BALANCE_COVARIATES:
        values = df[cov].astype(float).to_numpy()
        corr_before = weighted_corr(activity, values)
        corr_after = weighted_corr(activity, values, weights)
        rows.append(
            {
                "analysis": ACTIVITY_NAME,
                "treatment": ACTIVITY_COL,
                "covariate": cov,
                "corr_before": corr_before,
                "corr_after_gps": corr_after,
                "abs_corr_before": abs(corr_before),
                "abs_corr_after_gps": abs(corr_after),
            }
        )
    return pd.DataFrame(rows)


def run_gps_linear_analysis(df: pd.DataFrame) -> Dict[str, float | str]:
    """Run a GPS-weighted linear dose-response model."""
    activity = df[ACTIVITY_COL].astype(float).to_numpy()
    outcome = df[OUTCOME_COL].astype(float).to_numpy()
    weights = df["gps_weight"].astype(float).to_numpy()

    fit = weighted_ols(outcome, activity.reshape(-1, 1), weights)
    coef = fit["coef"]
    se = fit["se"]
    beta = float(coef[1])
    beta_se = float(se[1])
    t_value = beta / beta_se if beta_se > 0 else float("nan")

    return {
        "analysis": "gps_linear",
        "activity_definition": ACTIVITY_DESCRIPTION,
        "outcome_definition": OUTCOME_DESCRIPTION,
        "model": "GPS stabilized weighted linear dose-response",
        "treatment": ACTIVITY_COL,
        "outcome": OUTCOME_COL,
        "n": len(df),
        "activity_mean": float(activity.mean()),
        "activity_sd": float(activity.std(ddof=1)),
        "activity_min": float(activity.min()),
        "activity_max": float(activity.max()),
        "outcome_mean": float(outcome.mean()),
        "gps_weight_mean": float(weights.mean()),
        "gps_weight_min": float(weights.min()),
        "gps_weight_max": float(weights.max()),
        "gps_effective_sample_size": effective_sample_size(weights),
        "gps_conditional_sigma": float(df["gps_conditional_sigma"].iloc[0]),
        "gps_marginal_sigma": float(df["gps_marginal_sigma"].iloc[0]),
        "beta": beta,
        "se": beta_se,
        "t_value": float(t_value),
        "p_value": p_value_from_t(t_value),
        "r_squared": float(fit["r_squared"]),
        "beta_activity": beta,
        "beta_activity_sq": "",
    }


def run_gps_quadratic_sensitivity(df: pd.DataFrame) -> Dict[str, float | str]:
    """Run a quadratic dose-response sensitivity model."""
    activity = df[ACTIVITY_COL].astype(float).to_numpy()
    outcome = df[OUTCOME_COL].astype(float).to_numpy()
    weights = df["gps_weight"].astype(float).to_numpy()
    x = np.column_stack([activity, activity ** 2])

    fit = weighted_ols(outcome, x, weights)
    coef = fit["coef"]
    cov = fit["cov"]
    mean_activity = float(activity.mean())
    contrast = np.array([0.0, 1.0, 2.0 * mean_activity])
    beta = float(contrast @ coef)
    beta_se = math.sqrt(float(contrast @ cov @ contrast))
    t_value = beta / beta_se if beta_se > 0 else float("nan")

    return {
        "analysis": "gps_quadratic_sensitivity",
        "activity_definition": ACTIVITY_DESCRIPTION,
        "outcome_definition": OUTCOME_DESCRIPTION,
        "model": "GPS stabilized weighted quadratic dose-response; beta is marginal effect at mean activity",
        "treatment": ACTIVITY_COL,
        "outcome": OUTCOME_COL,
        "n": len(df),
        "activity_mean": mean_activity,
        "activity_sd": float(activity.std(ddof=1)),
        "activity_min": float(activity.min()),
        "activity_max": float(activity.max()),
        "outcome_mean": float(outcome.mean()),
        "gps_weight_mean": float(weights.mean()),
        "gps_weight_min": float(weights.min()),
        "gps_weight_max": float(weights.max()),
        "gps_effective_sample_size": effective_sample_size(weights),
        "gps_conditional_sigma": float(df["gps_conditional_sigma"].iloc[0]),
        "gps_marginal_sigma": float(df["gps_marginal_sigma"].iloc[0]),
        "beta": beta,
        "se": beta_se,
        "t_value": float(t_value),
        "p_value": p_value_from_t(t_value),
        "r_squared": float(fit["r_squared"]),
        "beta_activity": float(coef[1]),
        "beta_activity_sq": float(coef[2]),
    }


# =========================
# 5. Outputs
# =========================

def clean_old_outputs() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    keep = {"agent_scores.csv", "causal_results.csv", "covariate_balance.csv"}
    for path in RESULT_DIR.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()


def write_agent_scores(df: pd.DataFrame) -> None:
    """Write agent-level scores and GPS weights."""
    wallet_cols = [
        "owner_wallet",
        "agent_wallet",
        "owner_agent_count",
        "owner_agent_count_sq",
        "owner_equals_agent_wallet",
    ]
    score_cols = [
        "block_stamp",
        "survival_blocks",
        "log_survival_blocks",
        "tx_count",
        "identity_operation_count",
        "ecosystem_operation_count",
        "other_operation_count",
        "log_other_operation_count",
        "activity_score",
        "core_score",
        "diversity_penalty",
        "adjusted_reputation",
        "adjusted_reputation_index",
    ]
    gps_cols = [
        "gps_expected_activity",
        "gps_conditional_density",
        "gps_marginal_density",
        "gps_weight",
        "gps_effective_sample_size",
        "gps_model_log_survival_blocks_coef",
        "gps_model_owner_agent_count_coef",
        "gps_model_log_other_operation_count_coef",
        "gps_model_intercept",
    ]

    original_cols = [col for col in pd.read_csv(AGENT_STATISTIC_CSV, nrows=0).columns if col in df.columns]
    output_cols = list(dict.fromkeys([*original_cols, *wallet_cols, *score_cols, *gps_cols]))
    df.loc[:, output_cols].to_csv(RESULT_DIR / "agent_scores.csv", index=False)


# =========================
# 6. Main flow
# =========================

def main() -> None:
    clean_old_outputs()

    df = load_data()
    df = add_gps_weights(df)

    results: List[Dict[str, float | str]] = [
        run_gps_linear_analysis(df),
        run_gps_quadratic_sensitivity(df),
    ]
    result_df = pd.DataFrame(results)
    balance_df = make_balance_table(df)

    write_agent_scores(df)
    result_df.to_csv(RESULT_DIR / "causal_results.csv", index=False)
    balance_df.to_csv(RESULT_DIR / "covariate_balance.csv", index=False)

    print("[done] sklearn GPS causal analysis finished")
    print(f"[done] outputs saved to {RESULT_DIR}")
    for row in results:
        print(
            f"[result] {row['analysis']}: beta={float(row['beta']):.4f}, "
            f"p={format_p_value(float(row['p_value']))}"
        )


if __name__ == "__main__":
    main()
