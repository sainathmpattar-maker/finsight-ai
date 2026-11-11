"""FinSight AI - Spending Anomaly and Outlier Detection.

This module detects suspicious, irregular, or unexpected transactions such as
unusually high charge amounts, sudden category spending spikes, duplicate billings,
or erratic spending patterns.

Techniques:
1. Unsupervised Isolation Forest (scikit-learn) with multi-feature representations:
   - amount
   - log(amount)
   - day_of_week
   - hour (if available in timestamps)
   - category frequency
   - ratio of transaction amount to category median spend
2. Rule-Based Deterministic Guards:
   - Duplicate charge detector (same merchant & amount within 24 hours)
   - 3-Sigma Outlier thresholding (amount > category mean + 3 * category std)
3. Explainable Anomaly Reasons:
   - Human-readable explanations for flagged transactions.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
try:
    from sklearn.ensemble import IsolationForest
    _ISO_FOREST_AVAILABLE = True
except Exception:
    _ISO_FOREST_AVAILABLE = False


def extract_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Extract engineered features for Isolation Forest anomaly scoring.

    Parameters
    ----------
    df : pd.DataFrame
        Expense transaction DataFrame.

    Returns
    -------
    tuple of (pd.DataFrame, bool)
        (feature_matrix, has_hour_feature)
    """
    df_feat = pd.DataFrame(index=df.index)

    # 1. Base amounts
    amounts = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df_feat["amount"] = amounts
    df_feat["log_amount"] = np.log1p(np.maximum(0.0, amounts))

    # 2. Date features
    dates = pd.to_datetime(df["date"], errors="coerce")
    df_feat["day_of_week"] = dates.dt.dayofweek.fillna(0).astype(int)

    # 3. Hour (if available with non-zero variance)
    hours = dates.dt.hour.fillna(0).astype(int)
    has_hour = bool(hours.nunique() > 1)
    if has_hour:
        df_feat["hour"] = hours

    # 4. Category frequency (rarer categories get lower frequency score)
    if "category" in df.columns:
        cat_counts = df["category"].value_counts(normalize=True).to_dict()
        df_feat["category_freq"] = df["category"].map(cat_counts).fillna(0.0)
    else:
        df_feat["category_freq"] = 1.0

    # 5. Ratio to category median spend
    if "category" in df.columns:
        cat_medians = df.groupby("category")["amount"].transform("median")
        df_feat["category_median_ratio"] = amounts / (cat_medians + 1e-6)
    else:
        median_all = amounts.median()
        df_feat["category_median_ratio"] = amounts / (median_all + 1e-6)

    return df_feat, has_hour


def check_duplicates(df: pd.DataFrame, window_hours: int = 24) -> pd.Series:
    """Identify duplicate transactions with identical description and amount within window.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction DataFrame.
    window_hours : int, optional
        Time window threshold in hours (default 24).

    Returns
    -------
    pd.Series of bool
        Boolean series indicating whether each row is part of a duplicate pair.
    """
    if len(df) == 0:
        return pd.Series(dtype=bool)

    dates = pd.to_datetime(df["date"], errors="coerce")
    amounts = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    descriptions = df["description"].fillna("").astype(str).str.strip().str.lower()

    temp = pd.DataFrame({
        "orig_idx": df.index,
        "date": dates,
        "amount": amounts,
        "description": descriptions,
    }).sort_values(by="date")

    is_duplicate = pd.Series(False, index=df.index)

    # Group by exact description and amount
    grouped = temp.groupby(["description", "amount"])
    for _, group in grouped:
        if len(group) > 1:
            # Check pairwise time delta
            group_dates = group["date"].values
            group_indices = group["orig_idx"].values
            for i in range(len(group_dates)):
                for j in range(i + 1, len(group_dates)):
                    dt_diff = abs(pd.Timedelta(group_dates[j] - group_dates[i]).total_seconds())
                    if dt_diff <= window_hours * 3600:
                        is_duplicate.loc[group_indices[i]] = True
                        is_duplicate.loc[group_indices[j]] = True

    return is_duplicate


def check_statistical_outliers(
    df: pd.DataFrame,
    threshold_sigma: float = 3.0,
) -> Tuple[pd.Series, pd.Series]:
    """Flag transactions exceeding N standard deviations from category mean.

    Parameters
    ----------
    df : pd.DataFrame
        Expense transaction DataFrame.
    threshold_sigma : float, optional
        Standard deviation multiplier threshold (default 3.0).

    Returns
    -------
    tuple of (pd.Series of bool, pd.Series of float)
        (is_outlier, z_scores)
    """
    if len(df) == 0 or "category" not in df.columns:
        return pd.Series(False, index=df.index), pd.Series(0.0, index=df.index)

    amounts = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    cat_means = df.groupby("category")["amount"].transform("mean")
    cat_stds = df.groupby("category")["amount"].transform("std").fillna(0.0)

    z_scores = pd.Series(0.0, index=df.index)
    valid_std = cat_stds > 1e-4
    z_scores[valid_std] = (amounts[valid_std] - cat_means[valid_std]) / cat_stds[valid_std]

    is_outlier = (z_scores >= threshold_sigma) & (amounts > cat_means)
    return is_outlier, z_scores


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.035,
    random_state: int = 42,
) -> pd.DataFrame:
    """Detect unusual spending patterns using Isolation Forest and deterministic guards.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction DataFrame (must contain 'date', 'description', 'amount').
    contamination : float, optional
        Expected outlier proportion for Isolation Forest (default 0.035).
    random_state : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    pd.DataFrame
        Original DataFrame augmented with:
        - is_anomaly (bool)
        - anomaly_score (float, 0.0 to 1.0)
        - reason (str or None)
    """
    if df.empty:
        out = df.copy()
        out["is_anomaly"] = []
        out["anomaly_score"] = []
        out["reason"] = []
        return out

    out = df.copy()
    out["is_anomaly"] = False
    out["anomaly_score"] = 0.0
    out["reason"] = None

    # Focus anomaly scoring on expense transactions
    if "type" in out.columns:
        expense_mask = out["type"].astype(str).str.lower() == "expense"
    else:
        expense_mask = pd.Series(True, index=out.index)

    if not expense_mask.any():
        return out

    exp_df = out[expense_mask].copy()

    # 1. Feature Engineering & Isolation Forest (requires >= 20 expenses)
    if len(exp_df) >= 20 and _ISO_FOREST_AVAILABLE:
        X_features, has_hour = extract_features(exp_df)

        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=150,
        )
        iso_forest.fit(X_features)

        # Decision function: negative values indicate outliers.
        raw_decisions = iso_forest.decision_function(X_features)
        min_d, max_d = raw_decisions.min(), raw_decisions.max()
        if max_d > min_d:
            norm_scores = (max_d - raw_decisions) / (max_d - min_d)
        else:
            norm_scores = np.zeros_like(raw_decisions)

        iso_flagged = iso_forest.predict(X_features) == -1
    else:
        has_hour = False
        norm_scores = np.zeros(len(exp_df))
        iso_flagged = pd.Series(False, index=exp_df.index)

    # 2. Rule-based checks: Duplicates within 24h (works from the start)
    dup_flagged = check_duplicates(exp_df, window_hours=24)

    # 3. Rule-based checks: 3-Sigma Category Outliers (if >= 20 samples)
    if len(exp_df) >= 20:
        stat_flagged, z_scores = check_statistical_outliers(exp_df, threshold_sigma=3.0)
    else:
        stat_flagged = pd.Series(False, index=exp_df.index)
        z_scores = pd.Series(0.0, index=exp_df.index)

    # Combine anomaly flags: any trigger marks as anomaly
    combined_anomaly = iso_flagged | dup_flagged | stat_flagged

    # Boost anomaly scores for rule-triggered items
    boosted_scores = norm_scores.copy()
    boosted_scores[dup_flagged] = np.maximum(boosted_scores[dup_flagged], 0.90)
    boosted_scores[stat_flagged] = np.maximum(boosted_scores[stat_flagged], 0.92)

    # 4. Generate Explainable Human-Readable Reasons
    reasons: List[Optional[str]] = []
    cat_medians = (
        exp_df.groupby("category")["amount"].transform("median")
        if "category" in exp_df.columns
        else exp_df["amount"].median()
    )

    for idx in exp_df.index:
        is_anom = combined_anomaly.loc[idx]
        if not is_anom:
            reasons.append(None)
            continue

        reason_parts: List[str] = []
        row_amt = float(exp_df.loc[idx, "amount"])
        row_cat = str(exp_df.loc[idx, "category"]) if "category" in exp_df.columns else "usual"
        median_val = float(cat_medians.loc[idx])
        ratio = row_amt / (median_val + 1e-6)
        z = float(z_scores.loc[idx])

        # Duplicate check priority
        if dup_flagged.loc[idx]:
            reason_parts.append("Possible duplicate charge (identical amount & merchant detected within 24h)")

        # High spending ratio
        if ratio >= 2.5:
            reason_parts.append(f"{ratio:.1f}x higher than your usual {row_cat} spend (typical median Rs {median_val:,.0f})")
        elif stat_flagged.loc[idx]:
            reason_parts.append(f"Spending spike: {z:.1f} standard deviations above {row_cat} average")

        # Late-night / unusual hour check (if hour was available)
        if has_hour and "date" in exp_df.columns:
            dt = pd.to_datetime(exp_df.loc[idx, "date"], errors="coerce")
            if dt is not pd.NaT and 1 <= dt.hour <= 5:
                reason_parts.append(f"Unusual late-night transaction time ({dt.strftime('%I:%M %p')})")

        # Fallback explanation if only isolation forest caught multi-feature anomaly
        if not reason_parts:
            reason_parts.append(f"Unusual spending pattern detected by Isolation Forest in {row_cat}")

        reasons.append(" | ".join(reason_parts))

    # Assign back to main output DataFrame
    out.loc[expense_mask, "is_anomaly"] = combined_anomaly.values
    out.loc[expense_mask, "anomaly_score"] = np.round(boosted_scores, 4)
    out.loc[expense_mask, "reason"] = reasons

    return out


if __name__ == "__main__":
    print("=" * 65)
    print("  FinSight AI - Spending Anomaly Detection & Benchmark")
    print("=" * 65)

    data_file = "data/transactions.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"'{data_file}' not found. Run src/data_generator.py first.")

    raw_df = pd.read_csv(data_file)
    print(f"\n[+] Loaded {len(raw_df)} transactions from '{data_file}'")

    # Run anomaly detection
    evaluated_df = detect_anomalies(raw_df, contamination=0.035)

    # -------------------------------------------------------------
    # Unit-Test Style Asserts
    # -------------------------------------------------------------
    # 1. Column presence and types
    assert "is_anomaly" in evaluated_df.columns, "Assertion failed: 'is_anomaly' column missing"
    assert "anomaly_score" in evaluated_df.columns, "Assertion failed: 'anomaly_score' column missing"
    assert "reason" in evaluated_df.columns, "Assertion failed: 'reason' column missing"
    assert evaluated_df["is_anomaly"].dtype == bool, "Assertion failed: 'is_anomaly' must be boolean"
    assert (evaluated_df["anomaly_score"] >= 0.0).all() and (
        evaluated_df["anomaly_score"] <= 1.0
    ).all(), "Assertion failed: anomaly_score must be between 0.0 and 1.0"

    # 2. Check injected anomalies recall
    # Specific known merchant tokens injected during data generation
    injected_keywords = [
        "APPLE IPAD AIR",
        "VIP LOUNGE",
        "SWIGGY ORDER 8842",
        "GOA ROUNDTRIP FLIGHT",
        "UBER TRIP HELP-9133",
        "WHOLESALE PANTRY",
        "JETBRAINS",
        "TANISHQ JEWELLERY",
        "MANIPAL HOSPITAL",
        "BILLPAY-DUP",
    ]

    injected_mask = evaluated_df["description"].apply(
        lambda d: any(kw in str(d) for kw in injected_keywords)
    )
    total_injected = int(injected_mask.sum())
    caught_injected = int((injected_mask & evaluated_df["is_anomaly"]).sum())
    recall = caught_injected / total_injected if total_injected > 0 else 0.0

    print(f"\n[+] Total Injected Ground-Truth Anomalies: {total_injected}")
    print(f"[+] Injected Anomalies Caught:            {caught_injected}")
    print(f"[+] Anomaly Detection Recall:             {recall * 100:.2f}%")
    print(f"[+] Total Anomalies Flagged in Dataset:   {evaluated_df['is_anomaly'].sum()} ({(evaluated_df['is_anomaly'].sum() / len(evaluated_df)) * 100:.1f}%)")

    # Assert high recall (> 90%)
    assert recall >= 0.90, f"Assertion failed: Recall {recall:.2%} is below 90%"
    print("[OK] All Unit-Test Asserts Passed Successfully!")

    # -------------------------------------------------------------
    # Preview Flagged Anomalies
    # -------------------------------------------------------------
    print("\n--- Preview of Detected Anomalies & Explainable Reasons ---")
    flagged = evaluated_df[evaluated_df["is_anomaly"]].sort_values(
        by="anomaly_score", ascending=False
    )
    cols_to_show = ["date", "description", "amount", "category", "anomaly_score", "reason"]
    print(flagged[cols_to_show].head(12).to_string(index=False))
