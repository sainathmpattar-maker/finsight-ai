"""FinSight AI - Financial Data Utilities and KPI Engine.

This module provides data loading, schema validation, data cleaning,
formatting, and core financial KPI calculations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["date", "description", "amount"]
RECOMMENDED_COLUMNS = ["type", "category", "payment_mode"]


def format_inr(amount: float, symbol: str = "₹", decimals: int = 2) -> str:
    """Format a numeric amount into Indian Rupee representation.

    Parameters
    ----------
    amount : float
        Numeric value to format.
    symbol : str, optional
        Currency symbol prefix (default '₹').
    decimals : int, optional
        Decimal places (default 2).

    Returns
    -------
    str
        Formatted string (e.g., '₹25,000.00' or '-₹1,240.50').
    """
    if pd.isna(amount):
        return f"{symbol}0.00"

    is_negative = amount < 0
    abs_amt = abs(amount)

    if decimals == 0:
        formatted = f"{abs_amt:,.0f}"
    else:
        formatted = f"{abs_amt:,.{decimals}f}"

    prefix = f"-{symbol}" if is_negative else symbol
    return f"{prefix}{formatted}"


def parse_flexible_date(series: pd.Series) -> pd.Series:
    """Parse dates supporting YYYY-MM-DD, DD-MM-YYYY, and other standard formats."""
    parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    mask = parsed.isna()
    if mask.any():
        parsed_ddmmyyyy = pd.to_datetime(series[mask], format="%d-%m-%Y", errors="coerce")
        parsed.update(parsed_ddmmyyyy)
    mask = parsed.isna()
    if mask.any():
        parsed_slash = pd.to_datetime(series[mask], format="%d/%m/%Y", errors="coerce")
        parsed.update(parsed_slash)
    mask = parsed.isna()
    if mask.any():
        parsed_mixed = pd.to_datetime(series[mask], errors="coerce", dayfirst=True)
        parsed.update(parsed_mixed)
    return parsed


def validate_and_clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and clean input transaction records.

    Parameters
    ----------
    df : pd.DataFrame
        Input transaction DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with guaranteed column types and chronological sorting.
    """
    if df.empty:
        return df.copy()

    df_clean = df.copy()

    # Standardize column names (lowercase, stripped)
    df_clean.columns = [str(col).strip().lower() for col in df_clean.columns]

    # Check required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df_clean.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing mandatory columns: {missing}")

    # Parse dates flexibly
    df_clean["date"] = parse_flexible_date(df_clean["date"])
    df_clean = df_clean.dropna(subset=["date"])

    # Clean amount
    df_clean["amount"] = pd.to_numeric(df_clean["amount"], errors="coerce").fillna(0.0)

    # Clean description
    df_clean["description"] = df_clean["description"].fillna("Unknown Merchant").astype(str)

    # Default 'type' column if not provided (assume negative amounts or all are expense)
    if "type" not in df_clean.columns:
        df_clean["type"] = np.where(df_clean["amount"] < 0, "expense", "expense")
        df_clean["amount"] = df_clean["amount"].abs()
    else:
        df_clean["type"] = df_clean["type"].astype(str).str.strip().str.lower()
        # Normalize types
        df_clean["type"] = df_clean["type"].apply(
            lambda t: "income" if any(w in t for w in ["income", "credit", "salary", "stipend"]) else "expense"
        )
        df_clean["amount"] = df_clean["amount"].abs()

    # Default 'category' if missing
    if "category" not in df_clean.columns:
        df_clean["category"] = "Uncategorized"
    else:
        df_clean["category"] = df_clean["category"].fillna("Uncategorized").astype(str).str.strip()

    # Default 'payment_mode' if missing
    if "payment_mode" not in df_clean.columns:
        df_clean["payment_mode"] = "UPI"
    else:
        df_clean["payment_mode"] = df_clean["payment_mode"].fillna("UPI").astype(str).str.strip()

    # Sort chronologically
    df_clean = df_clean.sort_values(by="date", ascending=True).reset_index(drop=True)

    return df_clean


def validate_uploaded_csv(source: Any) -> Tuple[bool, Optional[pd.DataFrame], List[str]]:
    """Validate an uploaded CSV file with comprehensive checks and user-friendly error messages.

    Parameters
    ----------
    source : str or file-like object
        File path or Streamlit UploadedFile buffer.

    Returns
    -------
    tuple of (bool, pd.DataFrame or None, list of str)
        (is_valid, cleaned_dataframe, error_messages_list)
    """
    errors: List[str] = []

    if source is None:
        return False, None, ["No file was provided for validation."]

    # 1. Read CSV
    try:
        # If it's a file buffer, ensure pointer is at 0
        if hasattr(source, "seek"):
            source.seek(0)
        df_raw = pd.read_csv(source)
    except pd.errors.EmptyDataError:
        return False, None, ["The uploaded file is empty. Please provide a CSV file with transaction records."]
    except UnicodeDecodeError:
        return False, None, [
            "Encoding error: The file is not a valid UTF-8 text file. Please save your file as CSV UTF-8 and try again."
        ]
    except Exception as e:
        return False, None, [f"Failed to parse CSV file: {str(e)}"]

    # 2. Check if table is empty
    if df_raw.empty:
        return False, None, ["The uploaded CSV contains headers but has 0 data rows."]

    # Normalize column names
    raw_cols = [str(c).strip().lower() for c in df_raw.columns]
    df_raw.columns = raw_cols

    # 3. Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in raw_cols]
    if missing_cols:
        col_list_str = ", ".join([f"'{c}'" for c in missing_cols])
        errors.append(
            f"Missing required column(s): {col_list_str}. "
            f"Your CSV must include at least: 'date', 'description', and 'amount'."
        )
        return False, None, errors

    # 4. Check date column
    parsed_dates = parse_flexible_date(df_raw["date"])
    valid_date_count = parsed_dates.notna().sum()
    if valid_date_count == 0:
        errors.append(
            "None of the values in the 'date' column could be parsed into valid dates. "
            "Expected standard date formats like 'YYYY-MM-DD' (e.g. 2026-05-15) or 'DD-MM-YYYY'."
        )
        return False, None, errors

    # 5. Check amount column
    parsed_amounts = pd.to_numeric(df_raw["amount"], errors="coerce")
    valid_amount_count = parsed_amounts.notna().sum()
    if valid_amount_count == 0:
        errors.append(
            "None of the values in the 'amount' column could be parsed as numbers. "
            "Please ensure amounts are numeric (e.g., 250.00, 1200)."
        )
        return False, None, errors

    # 6. Clean and format through standard pipeline
    try:
        cleaned_df = validate_and_clean_transactions(df_raw)
        return True, cleaned_df, []
    except Exception as e:
        return False, None, [f"Data normalization error: {str(e)}"]


def load_transaction_csv(source: Any) -> pd.DataFrame:
    """Load and normalize transaction data from a file path or buffer.

    Parameters
    ----------
    source : str or file-like
        Filepath or UploadedFile object.

    Returns
    -------
    pd.DataFrame
        Validated and cleaned DataFrame.
    """
    df_raw = pd.read_csv(source)
    return validate_and_clean_transactions(df_raw)


def calculate_financial_kpis(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute high-level executive financial metrics and KPIs.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame.

    Returns
    -------
    dict
        Calculated financial indicators.
    """
    if df.empty:
        return {
            "total_income": 0.0,
            "total_expenses": 0.0,
            "net_savings": 0.0,
            "savings_rate_pct": 0.0,
            "avg_monthly_expense": 0.0,
            "avg_daily_expense": 0.0,
            "top_category": "None",
            "top_category_amount": 0.0,
            "transaction_count": 0,
        }

    incomes = df[df["type"] == "income"] if "type" in df.columns else pd.DataFrame()
    expenses = df[df["type"] == "expense"] if "type" in df.columns else df

    total_income = float(incomes["amount"].sum()) if not incomes.empty else 0.0
    total_expenses = float(expenses["amount"].sum()) if not expenses.empty else 0.0

    net_savings = total_income - total_expenses
    # Never divide by zero
    savings_rate = (net_savings / total_income * 100.0) if total_income > 0 else 0.0

    # Date range for daily/monthly averages
    min_date = df["date"].min()
    max_date = df["date"].max()
    days_span = max(1, (max_date - min_date).days + 1)
    months_span = max(1.0, days_span / 30.4)

    avg_daily = total_expenses / days_span if days_span > 0 else 0.0
    avg_monthly = total_expenses / months_span if months_span > 0 else 0.0

    # Top category
    if not expenses.empty and "category" in expenses.columns:
        cat_group = expenses.groupby("category")["amount"].sum()
        top_cat = str(cat_group.idxmax())
        top_cat_amt = float(cat_group.max())
    else:
        top_cat = "None"
        top_cat_amt = 0.0

    return {
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "net_savings": round(net_savings, 2),
        "savings_rate_pct": round(savings_rate, 1),
        "avg_monthly_expense": round(avg_monthly, 2),
        "avg_daily_expense": round(avg_daily, 2),
        "top_category": top_cat,
        "top_category_amount": round(top_cat_amt, 2),
        "transaction_count": len(df),
    }


def get_top_merchants(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Extract top spending merchants/descriptions with totals and frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame.
    top_n : int, optional
        Number of top merchants to return (default 10).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [merchant, total_spend, transaction_count, avg_spend].
    """
    expenses = df[df["type"] == "expense"] if "type" in df.columns else df
    if expenses.empty:
        return pd.DataFrame(columns=["merchant", "total_spend", "transaction_count", "avg_spend"])

    # Extract base merchant (strip ID suffixes like #4821 or -9923)
    def _extract_base_merchant(desc: str) -> str:
        d = str(desc).strip()
        # Remove trailing #1234 or -1234
        import re
        d = re.sub(r"[#\-]\s*\d+$", "", d)
        d = re.sub(r"\s+\d+$", "", d)
        return d.strip()

    merchants = expenses["description"].apply(_extract_base_merchant)
    grouped = expenses.groupby(merchants)["amount"].agg(
        total_spend="sum",
        transaction_count="count",
        avg_spend="mean",
    ).reset_index()

    grouped.rename(columns={"description": "merchant"}, inplace=True)
    grouped["total_spend"] = grouped["total_spend"].round(2)
    grouped["avg_spend"] = grouped["avg_spend"].round(2)

    return grouped.sort_values(by="total_spend", ascending=False).head(top_n).reset_index(drop=True)
