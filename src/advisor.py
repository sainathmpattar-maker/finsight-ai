"""FinSight AI - AI Financial Recommendation Engine.

This module powers personal finance advisory and conversational coaching
using Google's Gemini models via the `google-genai` SDK, with a resilient
fallback engine for offline or unauthenticated operation.

Features:
- Structured financial summary compilation.
- Strict JSON recommendation generation (Health score, 4-6 categorized tips, warnings).
- Grounded conversational coach for user queries.
- Rule-based fallback engine (50/30/20 rule, category audits, subscription optimization).
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
import numpy as np
import pandas as pd

# Load environment variables (.env)
load_dotenv()

# Current Gemini model configuration (easily configurable)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def get_gemini_api_key() -> Optional[str]:
    """Retrieve Gemini API key from st.secrets (Streamlit Cloud) or environment (.env locally)."""
    # 1. Try Streamlit Cloud secrets first
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", "").strip()
        if key and "your_" not in key.lower() and "<" not in key:
            return key
    except Exception:
        pass
    # 2. Fall back to .env / environment variable
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or "your_" in key.lower() or "<" in key:
        return None
    return key


def build_financial_summary(
    df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame] = None,
    anomalies_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Synthesize raw transactions, forecasts, and anomalies into a compact summary dict.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction DataFrame with ['date', 'amount', 'type', 'category'].
    forecast_df : pd.DataFrame, optional
        Forecast results from forecast_spending().
    anomalies_df : pd.DataFrame, optional
        Dataset with 'is_anomaly' and 'reason' columns.

    Returns
    -------
    dict
        Compact financial summary dict tailored for LLM prompt context.
    """
    if df.empty:
        return {
            "monthly_income": 0.0,
            "monthly_expenses": 0.0,
            "savings_rate_percent": 0.0,
            "top_5_categories": [],
            "month_over_month_change_percent": 0.0,
            "anomalies_count": 0,
            "recent_anomalies": [],
            "predicted_next_month_spend": 0.0,
            "predicted_next_month_savings": 0.0,
            "currency": "INR (Rs)",
        }

    df_copy = df.copy()
    df_copy["date_dt"] = pd.to_datetime(df_copy["date"], errors="coerce")
    df_copy["month"] = df_copy["date_dt"].dt.strftime("%Y-%m")

    # Incomes & Expenses split
    if "type" in df_copy.columns:
        expenses = df_copy[df_copy["type"].astype(str).str.lower() == "expense"].copy()
        incomes = df_copy[df_copy["type"].astype(str).str.lower() == "income"].copy()
    else:
        expenses = df_copy.copy()
        incomes = pd.DataFrame()

    # 1. Income Metrics
    if not incomes.empty:
        monthly_income_series = incomes.groupby("month")["amount"].sum()
        avg_monthly_income = float(monthly_income_series.mean())
    else:
        avg_monthly_income = 0.0

    # 2. Expense Metrics
    monthly_expense_series = expenses.groupby("month")["amount"].sum().sort_index()
    avg_monthly_expenses = (
        float(monthly_expense_series.mean()) if not monthly_expense_series.empty else 0.0
    )
    last_month_expense = (
        float(monthly_expense_series.iloc[-1]) if len(monthly_expense_series) > 0 else 0.0
    )

    # Month-over-month change
    if len(monthly_expense_series) >= 2:
        prev_month_expense = float(monthly_expense_series.iloc[-2])
        if prev_month_expense > 0:
            mom_change = ((last_month_expense - prev_month_expense) / prev_month_expense) * 100.0
        else:
            mom_change = 0.0
    else:
        mom_change = 0.0

    # Savings Rate - avoid divide by zero
    if avg_monthly_income > 0:
        net_savings = avg_monthly_income - avg_monthly_expenses
        savings_rate = (net_savings / avg_monthly_income) * 100.0
    else:
        net_savings = -avg_monthly_expenses
        savings_rate = 0.0

    # 3. Category Breakdown (Top 5)
    cat_spends = expenses.groupby("category")["amount"].sum().sort_values(ascending=False)
    total_spend_all = cat_spends.sum()
    top_5_categories: List[Dict[str, Any]] = []
    for cat, amt in cat_spends.head(5).items():
        share = (amt / total_spend_all * 100.0) if total_spend_all > 0 else 0.0
        top_5_categories.append({
            "category": str(cat),
            "total_spend": round(float(amt), 2),
            "share_percent": round(float(share), 1),
        })

    # 4. Anomalies
    if anomalies_df is not None and "is_anomaly" in anomalies_df.columns:
        flagged = anomalies_df[anomalies_df["is_anomaly"] == True]
        anomalies_count = len(flagged)
        sample_anomalies = []
        for _, row in flagged.sort_values(by="amount", ascending=False).head(3).iterrows():
            sample_anomalies.append({
                "description": str(row.get("description", "")),
                "amount": float(row.get("amount", 0.0)),
                "reason": str(row.get("reason", "")),
            })
    else:
        anomalies_count = 0
        sample_anomalies = []

    # 5. Forecast Projections
    if forecast_df is not None and "predicted_next_month" in forecast_df.columns:
        predicted_spend = float(forecast_df["predicted_next_month"].sum())
    else:
        predicted_spend = avg_monthly_expenses

    predicted_savings = avg_monthly_income - predicted_spend

    return {
        "monthly_income": round(avg_monthly_income, 2),
        "monthly_expenses": round(avg_monthly_expenses, 2),
        "last_month_expenses": round(last_month_expense, 2),
        "savings_rate_percent": round(savings_rate, 1),
        "month_over_month_change_percent": round(mom_change, 1),
        "top_5_categories": top_5_categories,
        "anomalies_count": anomalies_count,
        "recent_anomalies": sample_anomalies,
        "predicted_next_month_spend": round(predicted_spend, 2),
        "predicted_next_month_savings": round(predicted_savings, 2),
        "currency": "INR (Rs)",
    }


def _rule_based_recommendations(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Generate high-quality rule-based recommendations when Gemini API is unavailable."""
    income = summary.get("monthly_income", 25000.0)
    expenses = summary.get("monthly_expenses", 20000.0)
    savings_rate = summary.get("savings_rate_percent", 20.0)
    top_categories = {c["category"]: c["share_percent"] for c in summary.get("top_5_categories", [])}
    anomalies_cnt = summary.get("anomalies_count", 0)

    # Base Health Score Calculation
    score = 65
    if savings_rate >= 25.0:
        score += 20
    elif savings_rate >= 10.0:
        score += 10
    elif savings_rate < 0.0:
        score -= 25

    if anomalies_cnt > 5:
        score -= 10
    score = max(20, min(95, score))

    recommendations: List[Dict[str, Any]] = []
    warnings: List[str] = []

    # Check 1: Negative Savings / Deficit
    if savings_rate < 0:
        deficit = abs(income - expenses)
        warnings.append(
            f"You are spending more than your income by approximately Rs {deficit:,.0f}/month. Lifestyle inflation or one-off gadget purchases may be eroding your savings."
        )

    # Check 2: Food & Dining
    food_share = top_categories.get("Food & Dining", 0.0)
    if food_share >= 15.0:
        saving_est = round((food_share - 10.0) * expenses / 100.0, -1)
        saving_est = max(1200, min(4000, saving_est))
        recommendations.append({
            "title": "Cap Food Delivery & Weekend Outings",
            "action": f"Food & Dining accounts for {food_share:.1f}% of your total expenses. Cooking 2 extra meals a week or grouping Swiggy/Zomato orders can free up cash.",
            "estimated_monthly_saving_inr": int(saving_est),
            "priority": "high",
        })

    # Check 3: Shopping Spikes
    shopping_share = top_categories.get("Shopping", 0.0)
    if shopping_share >= 15.0:
        saving_est = round(shopping_share * expenses * 0.25 / 100.0, -1)
        saving_est = max(1500, min(5000, saving_est))
        recommendations.append({
            "title": "Implement a 48-Hour Cart Rule",
            "action": f"Shopping is consuming {shopping_share:.1f}% of your budget. Enforce a 48-hour wait period for non-essential Amazon, Flipkart, and fashion purchases.",
            "estimated_monthly_saving_inr": int(saving_est),
            "priority": "high",
        })

    # Check 4: Subscriptions Audit
    recommendations.append({
        "title": "Audit Recurring Subscriptions & Family Sharing",
        "action": "Switch single OTT/music subscriptions (Netflix, Spotify, YouTube) to shared family plans, or rotate memberships monthly.",
        "estimated_monthly_saving_inr": 450,
        "priority": "medium",
    })

    # Check 5: 50/30/20 Rule Alignment
    recommendations.append({
        "title": "Automate 20% Savings on Salary Day",
        "action": f"Set up an auto-debit of Rs {income * 0.20:,.0f} (20% of your Rs {income:,.0f} income) into a recurring deposit or liquid fund on the 2nd of each month before spending starts.",
        "estimated_monthly_saving_inr": int(income * 0.20),
        "priority": "medium",
    })

    # Check 6: Anomaly Guard
    if anomalies_cnt > 0:
        warnings.append(
            f"Detected {anomalies_cnt} unusual spending transactions or duplicate charges. Review the Anomaly Detection tab to verify all transactions."
        )
        recommendations.append({
            "title": "Verify Flagged Duplicate Charges",
            "action": "Raise chargeback requests for duplicate UPI and delivery charges detected within 24 hours.",
            "estimated_monthly_saving_inr": 900,
            "priority": "high",
        })

    summary_text = (
        f"Your current monthly burn is Rs {expenses:,.0f} against an inflow of Rs {income:,.0f}. "
        f"Savings rate is currently {savings_rate:.1f}%. With minor lifestyle tweaks in "
        f"Dining and Shopping, you can comfortably optimize your monthly surplus."
    )

    return {
        "health_score": int(score),
        "summary": summary_text,
        "recommendations": recommendations[:5],
        "warnings": warnings,
        "engine": "FinSight Rule Engine (Fallback)",
    }


def _clean_json_string(text: str) -> str:
    """Strip markdown backticks, code fences, and extraneous prefix text."""
    text = text.strip()
    # Strip markdown ```json ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    # Search for outermost JSON object { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def get_recommendations(
    summary: Dict[str, Any],
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> Dict[str, Any]:
    """Generate structured AI personal finance recommendations via Gemini API with fallback.

    Parameters
    ----------
    summary : dict
        Compact financial summary generated by build_financial_summary().
    model_name : str, optional
        Gemini model identifier (default DEFAULT_GEMINI_MODEL).

    Returns
    -------
    dict
        Structured dict with:
        - health_score: int (0-100)
        - summary: str
        - recommendations: list of {title, action, estimated_monthly_saving_inr, priority}
        - warnings: list of str
    """
    api_key = get_gemini_api_key()
    if not api_key:
        # Fallback gracefully
        return _rule_based_recommendations(summary)

    # Attempt to query Gemini API via google-genai
    prompt = f"""You are FinSight AI, a friendly, practical, and analytical Indian personal finance coach for a college student / young professional.

Review this financial snapshot:
- Average Monthly Income: Rs {summary.get('monthly_income', 0):,.2f}
- Average Monthly Expense: Rs {summary.get('monthly_expenses', 0):,.2f}
- Current Savings Rate: {summary.get('savings_rate_percent', 0):.1f}%
- Month-over-Month Expense Change: {summary.get('month_over_month_change_percent', 0):.1f}%
- Top Expense Categories: {json.dumps(summary.get('top_5_categories', []))}
- Flagged Anomalies / Duplicate Count: {summary.get('anomalies_count', 0)}
- Recent Flagged Anomalies: {json.dumps(summary.get('recent_anomalies', []))}
- Projected Next Month Outflow: Rs {summary.get('predicted_next_month_spend', 0):,.2f}
- Projected Next Month Savings: Rs {summary.get('predicted_next_month_savings', 0):,.2f}

Respond ONLY with a valid, strictly formatted JSON object. Do not include markdown code fences, commentary, or text outside the JSON.

Required JSON Structure:
{{
  "health_score": <integer from 0 to 100 based on savings rate, debt risk, and discipline>,
  "summary": "<2-3 sentence executive review tailored to the actual numbers above>",
  "recommendations": [
    {{
      "title": "<Concise action title>",
      "action": "<Specific, practical step mentioning realistic INR amounts or merchants>",
      "estimated_monthly_saving_inr": <realistic monthly saving amount as integer>,
      "priority": "<high|medium|low>"
    }}
  ],
  "warnings": [
    "<specific risk warnings regarding anomalies, negative cash-flow, or overspending>"
  ]
}}

Generate exactly 4 to 6 high-impact recommendations tailored to this Indian student profile."""

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )

        raw_text = response.text or ""
        cleaned_json = _clean_json_string(raw_text)
        data = json.loads(cleaned_json)

        # Basic schema validation
        if "health_score" in data and "recommendations" in data:
            data["engine"] = f"Gemini ({model_name})"
            return data
        else:
            return _rule_based_recommendations(summary)

    except Exception:
        # Graceful fallback so demos never fail
        return _rule_based_recommendations(summary)


def chat_with_coach(
    question: str,
    summary: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """Answer a user query with context-grounded financial advice.

    Parameters
    ----------
    question : str
        User's question in chat interface.
    summary : dict
        User's financial summary dictionary.
    conversation_history : list of dict, optional
        Previous turns [{'role': 'user'|'model', 'text': '...'}].
    model_name : str, optional
        Target Gemini model.

    Returns
    -------
    str
        Markdown response answering the query.
    """
    api_key = get_gemini_api_key()

    # Fallback response generator if API key is not present or API fails
    def _fallback_chat(query: str) -> str:
        q_lower = query.lower()
        inc = summary.get("monthly_income", 25000.0)
        exp = summary.get("monthly_expenses", 20000.0)
        proj_sav = summary.get("predicted_next_month_savings", 5000.0)

        if any(w in q_lower for w in ["afford", "buy", "purchase", "keyboard", "laptop", "phone"]):
            return (
                f"**Financial Feasibility Assessment:**\n\n"
                f"- Your estimated monthly income is **Rs {inc:,.0f}**, with average expenses around **Rs {exp:,.0f}**.\n"
                f"- Your projected surplus for next month is **Rs {proj_sav:,.0f}**.\n\n"
                f"**Coach Recommendation:** If this purchase is non-essential, ensure it does not exceed 50% of your projected surplus. "
                f"If the item costs more than Rs {max(1000.0, proj_sav * 0.5):,.0f}, consider delaying by one month or splitting it into two savings milestones."
            )
        elif any(w in q_lower for w in ["food", "swiggy", "zomato", "dining"]):
            return (
                f"**Food & Dining Insights:**\n\n"
                f"Your dining expenses are one of your primary outflow categories. "
                f"To optimize without compromising social life, try restricting food delivery orders to 2 days a week "
                f"and setting a weekly dining budget of Rs 1,500. This will save ~Rs 2,000 to Rs 3,500 monthly."
            )
        elif any(w in q_lower for w in ["invest", "sip", "save", "mutual fund"]):
            return (
                f"**Savings & Investment Strategy:**\n\n"
                f"Given your current profile, focus on this 3-step ladder:\n"
                f"1. **Emergency Buffer**: Accumulate 1 month of living expenses (approx Rs {exp:,.0f}) in a high-interest savings account or liquid fund.\n"
                f"2. **Start an Index SIP**: Once the buffer is ready, start an automated SIP of Rs 1,000–Rs 2,500/month in a Nifty 50 Index Fund.\n"
                f"3. **Avoid high-risk day trading** until your core emergency fund is fully capitalized."
            )
        else:
            return (
                f"Based on your 6-month transaction profile, your average monthly spend is **Rs {exp:,.0f}** against an inflow of **Rs {inc:,.0f}** "
                f"(Savings Rate: **{summary.get('savings_rate_percent', 0):.1f}%**). "
                f"Focus on keeping discretionary categories (Shopping and Dining) within 30% of total income to safeguard your cash runway."
            )

    if not api_key:
        return _fallback_chat(question)

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        system_context = f"""You are a friendly, savvy Indian personal finance coach for a student/young professional.
User's Financial Profile:
- Monthly Income: Rs {summary.get('monthly_income', 0):,.2f}
- Monthly Expenses: Rs {summary.get('monthly_expenses', 0):,.2f}
- Current Savings Rate: {summary.get('savings_rate_percent', 0):.1f}%
- Top Spending Categories: {json.dumps(summary.get('top_5_categories', []))}
- Flagged Anomalies: {summary.get('anomalies_count', 0)}
- Projected Next Month Savings: Rs {summary.get('predicted_next_month_savings', 0):,.2f}

Give practical, empathetic, and financially grounded advice in Indian Rupees (Rs). Keep answers concise, direct, and structured with bullet points where appropriate."""

        full_prompt = f"{system_context}\n\nUser Question: {question}"
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
        )
        return response.text or _fallback_chat(question)

    except Exception:
        return _fallback_chat(question)


if __name__ == "__main__":
    print("=" * 65)
    print("  FinSight AI - AI Advisor & Recommendation Engine")
    print("=" * 65)

    data_file = "data/transactions.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"'{data_file}' not found.")

    raw_df = pd.read_csv(data_file)

    # Import modules to build full context
    from src.anomaly import detect_anomalies
    from src.forecast import forecast_spending

    print("\n[+] Preparing Financial Summary Context...")
    anomalies_df = detect_anomalies(raw_df)
    forecast_df, _ = forecast_spending(raw_df)
    summary = build_financial_summary(raw_df, forecast_df, anomalies_df)

    print(f"    - Monthly Income:             Rs. {summary['monthly_income']:,.2f}")
    print(f"    - Monthly Outflow:            Rs. {summary['monthly_expenses']:,.2f}")
    print(f"    - Current Savings Rate:       {summary['savings_rate_percent']:.1f}%")
    print(f"    - Month-over-Month Change:    {summary['month_over_month_change_percent']:+.1f}%")
    print(f"    - Anomalies Detected:         {summary['anomalies_count']}")
    print(f"    - Projected Next Mo Savings:  Rs. {summary['predicted_next_month_savings']:,.2f}")

    print("\n--- Generating Financial Recommendations (Gemini / Rule Fallback) ---")
    recommendations = get_recommendations(summary)
    print(f"Engine: {recommendations.get('engine', 'Unknown')}")
    print(f"Health Score: {recommendations.get('health_score')}/100")
    print(f"Summary: {recommendations.get('summary')}")

    print("\nAction Items:")
    for i, rec in enumerate(recommendations.get("recommendations", []), 1):
        print(f"  {i}. [{rec.get('priority', '').upper()}] {rec.get('title')}")
        print(f"     Action: {rec.get('action')}")
        print(f"     Estimated Monthly Saving: Rs. {rec.get('estimated_monthly_saving_inr'):,}")

    if recommendations.get("warnings"):
        print("\nWarnings:")
        for w in recommendations["warnings"]:
            print(f"  ! {w}")

    print("\n--- Testing Conversational Coach Query ---")
    test_question = "Can I afford to buy a new mechanical keyboard for Rs 4,500 next month?"
    print(f"Question: '{test_question}'")
    answer = chat_with_coach(test_question, summary)
    print("\nCoach Response:")
    print(answer)
