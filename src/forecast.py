"""FinSight AI - Spending Forecasting Engine.

This module forecasts overall and category-level expenditure for the upcoming
month using an explainable ensemble model: Linear Regression on historical month
indices blended with a 3-month moving average.

Key Capabilities:
- Monthly aggregation of category-wise and total expenses.
- Blended forecasting: (LinearRegression + 3-Month Moving Average) / 2.
- Handles limited historical windows gracefully (< 3 months).
- Projected savings calculation: (Average Monthly Income - Predicted Total Expense).
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _forecast_series(values: np.ndarray) -> Tuple[float, float]:
    """Forecast next period value using Linear Regression blended with moving average.

    Parameters
    ----------
    values : np.ndarray
        Chronological 1D array of historical monthly spending.

    Returns
    -------
    tuple of (float, float)
        (blended_prediction, last_month_value)
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0

    last_val = float(values[-1])

    # Case 1: Only 1 month of history
    if n == 1:
        return round(last_val, 2), round(last_val, 2)

    # Case 2: Exactly 2 months of history
    if n == 2:
        x = np.arange(2).reshape(-1, 1)
        lr = LinearRegression().fit(x, values)
        lr_pred = max(0.0, float(lr.predict([[2]])[0]))
        ma_pred = float(np.mean(values))
        blended = (lr_pred + ma_pred) / 2.0
        return round(blended, 2), round(last_val, 2)

    # Case 3: 3 or more months of history (default)
    x = np.arange(n).reshape(-1, 1)
    lr = LinearRegression().fit(x, values)
    lr_pred = max(0.0, float(lr.predict([[n]])[0]))

    # 3-month moving average
    ma_pred = float(np.mean(values[-3:]))

    blended = (lr_pred + ma_pred) / 2.0
    return round(blended, 2), round(last_val, 2)


def forecast_spending(
    df: pd.DataFrame,
    assumed_income: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Forecast next month's spending per category and calculate projected savings.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction DataFrame with 'date', 'amount', 'type', and 'category'.
    assumed_income : float, optional
        Fallback monthly income if no income transactions exist in df (default 25000.0).

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        forecast_df:
            DataFrame with columns: [category, last_month, predicted_next_month, change_percent]
        summary_metrics:
            Dictionary containing:
            - predicted_total_expense
            - last_month_total_expense
            - total_change_percent
            - average_monthly_income
            - projected_savings
            - projected_savings_rate
            - months_analyzed
            - monthly_pivot
    """
    if df.empty:
        empty_df = pd.DataFrame(columns=["category", "last_month", "predicted_next_month", "change_percent"])
        return empty_df, {
            "predicted_total_expense": 0.0,
            "last_month_total_expense": 0.0,
            "total_change_percent": 0.0,
            "average_monthly_income": assumed_income or 0.0,
            "projected_savings": assumed_income or 0.0,
            "projected_savings_rate": 0.0,
            "months_analyzed": 0,
            "monthly_pivot": pd.DataFrame(),
        }

    df_copy = df.copy()
    df_copy["date_dt"] = pd.to_datetime(df_copy["date"], errors="coerce")
    df_copy = df_copy.dropna(subset=["date_dt"])
    df_copy["month"] = df_copy["date_dt"].dt.strftime("%Y-%m")

    # Separate expenses and income
    if "type" in df_copy.columns:
        expenses = df_copy[df_copy["type"].astype(str).str.lower() == "expense"].copy()
        incomes = df_copy[df_copy["type"].astype(str).str.lower() == "income"].copy()
    else:
        expenses = df_copy.copy()
        incomes = pd.DataFrame()

    # Determine Average Monthly Income
    if not incomes.empty:
        monthly_income_series = incomes.groupby("month")["amount"].sum()
        avg_monthly_income = float(monthly_income_series.mean())
    elif assumed_income is not None:
        avg_monthly_income = float(assumed_income)
    else:
        avg_monthly_income = 0.0

    # Build monthly spending pivot per category
    if "category" not in expenses.columns:
        expenses["category"] = "General"

    pivot = expenses.pivot_table(
        index="month",
        columns="category",
        values="amount",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    months_count = len(pivot)
    results: List[Dict[str, Any]] = []

    for cat in pivot.columns:
        cat_values = pivot[cat].values.astype(float)
        pred_val, last_val = _forecast_series(cat_values)

        if last_val > 1e-4:
            pct_change = ((pred_val - last_val) / last_val) * 100.0
        elif pred_val > 0:
            pct_change = 100.0
        else:
            pct_change = 0.0

        results.append({
            "category": str(cat),
            "last_month": round(last_val, 2),
            "predicted_next_month": round(pred_val, 2),
            "change_percent": round(pct_change, 2),
        })

    forecast_df = pd.DataFrame(results).sort_values(
        by="predicted_next_month", ascending=False
    ).reset_index(drop=True)

    # Compute overall total expenditure prediction
    total_predicted = float(forecast_df["predicted_next_month"].sum())
    total_last_month = float(forecast_df["last_month"].sum())

    if total_last_month > 1e-4:
        total_change_pct = ((total_predicted - total_last_month) / total_last_month) * 100.0
    else:
        total_change_pct = 0.0

    # Projected Savings = Average Monthly Income - Predicted Total Expense
    projected_savings = avg_monthly_income - total_predicted
    projected_savings_rate = (
        (projected_savings / avg_monthly_income) * 100.0 if avg_monthly_income > 0 else 0.0
    )

    summary_metrics = {
        "predicted_total_expense": round(total_predicted, 2),
        "last_month_total_expense": round(total_last_month, 2),
        "total_change_percent": round(total_change_pct, 2),
        "average_monthly_income": round(avg_monthly_income, 2),
        "projected_savings": round(projected_savings, 2),
        "projected_savings_rate": round(projected_savings_rate, 2),
        "months_analyzed": months_count,
        "monthly_pivot": pivot,
    }

    return forecast_df, summary_metrics


def get_forecast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience accessor returning category forecast DataFrame directly."""
    forecast_df, _ = forecast_spending(df)
    return forecast_df


if __name__ == "__main__":
    print("=" * 65)
    print("  FinSight AI - Spending Forecast & Savings Projections")
    print("=" * 65)

    data_file = "data/transactions.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"'{data_file}' not found. Run src/data_generator.py first.")

    raw_df = pd.read_csv(data_file)
    forecast_df, metrics = forecast_spending(raw_df)

    print(f"\n[+] Historical Months Analyzed: {metrics['months_analyzed']}")
    print(f"[+] Average Monthly Income:     Rs. {metrics['average_monthly_income']:,.2f}")
    print(f"[+] Last Month Total Spend:     Rs. {metrics['last_month_total_expense']:,.2f}")
    print(f"[+] Predicted Next Month Spend: Rs. {metrics['predicted_total_expense']:,.2f} ({metrics['total_change_percent']:+.1f}%)")
    print(f"[+] Projected Net Savings:      Rs. {metrics['projected_savings']:,.2f} (Savings Rate: {metrics['projected_savings_rate']:.1f}%)")

    print("\n--- Category-Wise Forecast (Blended LinearRegression + 3M Moving Average) ---")
    print(forecast_df.to_string(index=False))

    # Test edge case: < 3 months of data
    print("\n--- Testing Edge Case: 2 Months of History ---")
    two_months_df = raw_df[pd.to_datetime(raw_df["date"]) >= "2026-09-01"]
    short_forecast_df, short_metrics = forecast_spending(two_months_df)
    print(f"Months Analyzed: {short_metrics['months_analyzed']}")
    print(f"Predicted Total Spend: Rs. {short_metrics['predicted_total_expense']:,.2f}")
    print(short_forecast_df.head(5).to_string(index=False))

    print("\n--- Testing Edge Case: 1 Month of History ---")
    one_month_df = raw_df[pd.to_datetime(raw_df["date"]) >= "2026-10-01"]
    one_forecast_df, one_metrics = forecast_spending(one_month_df)
    print(f"Months Analyzed: {one_metrics['months_analyzed']}")
    print(f"Predicted Total Spend: Rs. {one_metrics['predicted_total_expense']:,.2f}")
    print(one_forecast_df.head(5).to_string(index=False))
