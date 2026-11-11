"""FinSight AI - Streamlit Web Application Entry Point.

AI-Powered Personal Finance Management System developed for the 'AI for Finance' curriculum.
Starts completely empty with persistent SQLite storage (data/finsight.db).

Pages:
1. Add Transaction: Quick add shortcuts, validated manual entry, and recent transactions manager.
2. Dashboard: Real-time executive KPIs, monthly cashflow, category donut, spending trends, top merchants.
3. Smart Categorization: Cold-start rule-based keyword matcher + auto-trained TF-IDF Logistic Regression (unlocks at 30+ expenses).
4. Anomaly Alerts: 24h duplicate charge guard from start + Isolation Forest outlier radar (unlocks at 20+ expenses).
5. Forecast: Outflow forecasting & 'what-if' budget simulator (unlocks at 2+ months of data).
6. AI Coach: Gemini-powered health score and conversational coach (unlocks with >=1 income & >=1 expense).
7. About / How AI is Used: Technical documentation of AI/ML architectures and models.
"""

from datetime import date, datetime
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

# Local modules
from src.advisor import (
    DEFAULT_GEMINI_MODEL,
    build_financial_summary,
    chat_with_coach,
    get_gemini_api_key,
    get_recommendations,
)
from src.anomaly import detect_anomalies
from src.categorizer import (
    categorize_dataframe,
    is_ml_model_available,
    load_model,
    predict_category,
    train_model_from_db,
)
from src.db import (
    add_transaction,
    add_transactions_bulk,
    clear_all_data,
    delete_transaction,
    get_distinct_months,
    get_expense_count,
    get_transaction_count,
    get_transactions,
    has_income_and_expense,
    init_db,
    update_transaction,
)
from src.forecast import forecast_spending
from src.ui import (
    badge,
    empty_state,
    glass_card,
    hero_banner,
    inject_css,
    metric_card,
    render_footer,
    render_html,
    savings_goal_progress,
    section_header,
    spotify_insights_strip,
    style_fig,
    unlock_hint_card,
)
from src.utils import (
    calculate_financial_kpis,
    format_inr,
    get_top_merchants,
    parse_flexible_date,
    validate_and_clean_transactions,
    validate_uploaded_csv,
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="FinSight AI - Personal Finance Management",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database on startup
init_db()

# Inject Gen Z Fintech Design System & Animations
inject_css()

# Category definitions
EXPENSE_CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Transport",
    "Rent",
    "Bills & Utilities",
    "Shopping",
    "Entertainment",
    "Health",
    "Education",
    "Subscriptions",
    "Travel",
    "Other / General",
]

INCOME_CATEGORIES = [
    "Salary",
    "Freelance",
    "Investment",
    "Gift / Allowance",
    "Other Income",
]

ALL_CATEGORIES = sorted(list(set(EXPENSE_CATEGORIES + INCOME_CATEGORIES)))


# -----------------------------------------------------------------------------
# Navigation State Helper
# -----------------------------------------------------------------------------
def navigate_to(page_name: str) -> None:
    """Set target page in session state and rerun."""
    st.session_state["nav_target"] = page_name
    st.rerun()


# -----------------------------------------------------------------------------
# Sidebar Navigation & Global Controls
# -----------------------------------------------------------------------------
def render_sidebar() -> Tuple[str, str, pd.DataFrame]:
    """Render sidebar branding, data source toggle, filters, and page menu."""
    render_html("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 2px;">
        <span style="font-size: 1.85rem;">💳</span>
        <span class="genz-gradient-text" style="font-size: 1.65rem; font-weight: 800; letter-spacing: -0.02em;">FinSight AI</span>
    </div>
    <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 16px; font-weight: 500;">
        Personal Wealth Studio
    </div>
    """, sidebar=True)

    page_options = [
        "Add Transaction",
        "Dashboard",
        "Smart Categorization",
        "Anomaly Alerts",
        "Forecast",
        "AI Coach",
        "About / How AI is Used",
    ]
    page_icons = [
        "plus-circle-fill",
        "grid-1x2-fill",
        "tag-fill",
        "shield-exclamation",
        "graph-up-arrow",
        "stars",
        "info-circle-fill",
    ]

    default_idx = 0
    if "nav_target" in st.session_state and st.session_state["nav_target"] in page_options:
        default_idx = page_options.index(st.session_state["nav_target"])
        del st.session_state["nav_target"]

    with st.sidebar:
        selected_page = option_menu(
            menu_title=None,
            options=page_options,
            icons=page_icons,
            default_index=default_idx,
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "transparent",
                },
                "icon": {
                    "color": "#8B5CF6",
                    "font-size": "1.05rem",
                },
                "nav-link": {
                    "font-family": "Plus Jakarta Sans, sans-serif",
                    "font-size": "0.92rem",
                    "font-weight": "600",
                    "text-align": "left",
                    "margin": "3px 0px",
                    "padding": "9px 13px",
                    "border-radius": "14px",
                    "color": "#94A3B8",
                    "background-color": "rgba(20, 20, 43, 0.45)",
                    "border": "1px solid rgba(255, 255, 255, 0.05)",
                    "transition": "all 0.25s ease",
                    "--hover-color": "rgba(139, 92, 246, 0.12)",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%) !important",
                    "color": "#FFFFFF !important",
                    "font-weight": "700",
                    "box-shadow": "0 6px 20px rgba(139, 92, 246, 0.4)",
                    "border": "none",
                },
            },
        )

    render_html("<div style='margin-top: 8px;'></div>", sidebar=True)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Data Source")

    data_source_mode = st.sidebar.radio(
        "Source Mode",
        ["My data (manual entry)", "Import my CSV"],
        index=0,
        label_visibility="collapsed",
    )

    # Load live transactions from SQLite
    raw_df = get_transactions()
    total_txns = len(raw_df)

    st.sidebar.caption(f"💾 **Storage**: SQLite DB (`data/finsight.db`) | **Total Records**: {total_txns}")

    # Active filters if transactions exist
    if not raw_df.empty:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Filters")

        if "date" in raw_df.columns and raw_df["date"].notna().any():
            min_date = raw_df["date"].min().date()
            max_date = raw_df["date"].max().date()
            if min_date == max_date:
                st.sidebar.info(f"Showing date: {min_date}")
            else:
                date_range = st.sidebar.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_d, end_d = date_range
                    raw_df = raw_df[(raw_df["date"].dt.date >= start_d) & (raw_df["date"].dt.date <= end_d)]

        if "category" in raw_df.columns:
            all_cats = sorted(raw_df["category"].dropna().unique().tolist())
            if all_cats:
                selected_cats = st.sidebar.multiselect(
                    "Categories",
                    options=all_cats,
                    default=all_cats,
                )
                if selected_cats:
                    raw_df = raw_df[raw_df["category"].isin(selected_cats)]

    # Export & Reset Controls
    st.sidebar.markdown("---")
    full_df = get_transactions()
    if not full_df.empty:
        csv_buffer = io.StringIO()
        export_df = full_df.copy()
        if "date" in export_df.columns:
            export_df["date"] = export_df["date"].dt.strftime("%Y-%m-%d")
        export_df.to_csv(csv_buffer, index=False)
        st.sidebar.download_button(
            label="📥 Export My Data as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"finsight_export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.sidebar.expander("⚠️ Danger Zone: Clear Data"):
        confirm_clear = st.checkbox("I confirm I want to wipe all records", key="confirm_clear_chk")
        if st.button("🗑️ Clear All Data", type="secondary", disabled=not confirm_clear, use_container_width=True):
            clear_all_data()
            st.toast("All transactions wiped from database.", icon="🧹")
            st.rerun()

    # API Key Badge
    st.sidebar.markdown("---")
    api_key = get_gemini_api_key()
    if api_key:
        st.sidebar.success(f"Gemini API: Connected ({DEFAULT_GEMINI_MODEL})")
    else:
        st.sidebar.info("Gemini API: Offline / Rule Fallback")

    return selected_page, data_source_mode, raw_df


# -----------------------------------------------------------------------------
# Page 1: Add Transaction
# -----------------------------------------------------------------------------
def render_add_transaction():
    """Render manual entry form, quick add shortcuts, and recent transactions manager."""
    section_header(
        title="Add Real Transaction",
        subtitle="Record your genuine daily inflows and outflows. Data persists securely in your local SQLite database.",
        emoji="➕",
    )

    # Initialize prefill states if not present
    if "quick_desc" not in st.session_state:
        st.session_state.quick_desc = ""
    if "quick_amount" not in st.session_state:
        st.session_state.quick_amount = 0.0
    if "quick_type" not in st.session_state:
        st.session_state.quick_type = "Expense"
    if "quick_cat" not in st.session_state:
        st.session_state.quick_cat = "Food & Dining"
    if "quick_mode" not in st.session_state:
        st.session_state.quick_mode = "UPI"

    # Quick Add Buttons
    render_html("""
    <div style="font-size: 0.95rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
        ⚡ Quick Add Shortcuts:
    </div>
    """)
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)

    with qcol1:
        if st.button("☕ Chai (₹40)", use_container_width=True):
            st.session_state.quick_desc = "Tapri Chai & Bun Maska"
            st.session_state.quick_amount = 40.0
            st.session_state.quick_type = "Expense"
            st.session_state.quick_cat = "Food & Dining"
            st.session_state.quick_mode = "UPI"
            st.rerun()

    with qcol2:
        if st.button("🛺 Auto (₹80)", use_container_width=True):
            st.session_state.quick_desc = "Auto Rickshaw Fare UPI"
            st.session_state.quick_amount = 80.0
            st.session_state.quick_type = "Expense"
            st.session_state.quick_cat = "Transport"
            st.session_state.quick_mode = "UPI"
            st.rerun()

    with qcol3:
        if st.button("🍔 Swiggy (₹280)", use_container_width=True):
            st.session_state.quick_desc = "Swiggy Food Delivery"
            st.session_state.quick_amount = 280.0
            st.session_state.quick_type = "Expense"
            st.session_state.quick_cat = "Food & Dining"
            st.session_state.quick_mode = "UPI"
            st.rerun()

    with qcol4:
        if st.button("📱 Recharge (₹299)", use_container_width=True):
            st.session_state.quick_desc = "Mobile Prepaid Recharge"
            st.session_state.quick_amount = 299.0
            st.session_state.quick_type = "Expense"
            st.session_state.quick_cat = "Bills & Utilities"
            st.session_state.quick_mode = "UPI"
            st.rerun()

    render_html("<div style='margin-top: 14px;'></div>")

    # Clean glassmorphic container for the form
    with st.container():
        render_html("""
        <div class="genz-glass-card" style="padding: 24px 28px; margin-bottom: 16px;">
            <div style="font-size: 1.12rem; font-weight: 800; color: #FFFFFF; margin-bottom: 4px;">
                📝 Transaction Details
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8;">
                Fill out the fields below. Category is automatically suggested for expenses based on your description!
            </div>
        </div>
        """)

        with st.form("manual_entry_form", clear_on_submit=True):
            row1_col1, row1_col2, row1_col3 = st.columns([1, 1, 1])

            with row1_col1:
                txn_date = st.date_input(
                    "Date *",
                    value=date.today(),
                    max_value=date.today(),
                    help="Transaction date (cannot be in the future).",
                )

            with row1_col2:
                type_idx = 0 if st.session_state.quick_type == "Expense" else 1
                txn_type = st.radio(
                    "Type *",
                    ["Expense", "Income"],
                    index=type_idx,
                    horizontal=True,
                )

            with row1_col3:
                txn_amount = st.number_input(
                    "Amount (₹) *",
                    min_value=0.01,
                    value=float(st.session_state.quick_amount) if st.session_state.quick_amount > 0 else 50.0,
                    step=10.0,
                    format="%.2f",
                )

            row2_col1, row2_col2, row2_col3 = st.columns([1.5, 1.2, 1])

            with row2_col1:
                txn_desc = st.text_input(
                    "Description / Merchant *",
                    value=st.session_state.quick_desc,
                    placeholder="e.g. Swiggy, Metro Recharge, Salary Credit",
                )

            # Auto-suggest category for expenses
            suggested_cat = st.session_state.quick_cat
            if txn_type == "Expense":
                allowed_cats = EXPENSE_CATEGORIES
                if txn_desc:
                    pred_cat, _ = predict_category(txn_desc)
                    if pred_cat in allowed_cats:
                        suggested_cat = pred_cat
            else:
                allowed_cats = INCOME_CATEGORIES
                if suggested_cat not in allowed_cats:
                    suggested_cat = allowed_cats[0]

            cat_idx = allowed_cats.index(suggested_cat) if suggested_cat in allowed_cats else 0

            with row2_col2:
                txn_cat = st.selectbox(
                    "Category (Auto-Suggested)",
                    options=allowed_cats,
                    index=cat_idx,
                    help="Auto-suggested based on merchant description. You can override anytime.",
                )

            with row2_col3:
                modes = ["UPI", "Card", "Cash", "Net Banking"]
                mode_idx = modes.index(st.session_state.quick_mode) if st.session_state.quick_mode in modes else 0
                txn_mode = st.selectbox("Payment Mode", options=modes, index=mode_idx)

            submit_btn = st.form_submit_button("➕ Save Transaction to Database", type="primary", use_container_width=True)

    # Process Form Submission
    if submit_btn:
        # Validation
        errors = []
        if not txn_desc or not txn_desc.strip():
            errors.append("Description / Merchant name is required.")
        if txn_amount <= 0:
            errors.append("Amount must be greater than ₹0.")
        if txn_date > date.today():
            errors.append("Transaction date cannot be in the future.")

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            date_str = txn_date.strftime("%Y-%m-%d")
            add_transaction(
                date=date_str,
                description=txn_desc.strip(),
                amount=float(txn_amount),
                txn_type=txn_type.lower(),
                category=txn_cat,
                payment_mode=txn_mode,
            )
            # Reset quick-add states
            st.session_state.quick_desc = ""
            st.session_state.quick_amount = 0.0
            st.session_state.quick_cat = "Food & Dining"
            st.session_state.quick_mode = "UPI"

            # Check if ML model should now be auto-trained (threshold: 30+ categorized expenses)
            exp_count = get_expense_count()
            if exp_count >= 30 and not is_ml_model_available():
                try:
                    train_model_from_db()
                    st.toast("🎉 ML model trained automatically on your 30+ expenses!", icon="🤖")
                except Exception:
                    pass

            st.toast(f"Saved: {txn_desc} (₹{txn_amount:.2f})", icon="✅")
            st.rerun()

    # Recent Transactions Manager
    render_html("<div style='margin-top: 18px;'></div>")
    all_df = get_transactions()

    render_html(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
            <span>📋</span> Recent Transactions ({len(all_df)} Total)
        </div>
        <div style="font-size: 0.85rem; color: #94A3B8;">
            Direct SQLite records &bull; Inline editor
        </div>
    </div>
    """)

    if all_df.empty:
        empty_state(
            "📝✨",
            "No Transactions Recorded Yet",
            "Use the form above to add your first transaction, or hit one of the quick add buttons!",
        )
    else:
        # Display editable table
        display_df = all_df.sort_values(by=["date", "id"], ascending=[False, False]).copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

        with st.expander("✏️ Edit Transactions Table", expanded=True):
            st.caption("Double-click any cell to edit details directly, or select rows to delete.")
            edited_table = st.data_editor(
                display_df[["id", "date", "description", "amount", "type", "category", "payment_mode"]],
                column_config={
                    "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                    "date": st.column_config.TextColumn("Date (YYYY-MM-DD)", required=True),
                    "description": st.column_config.TextColumn("Description", required=True),
                    "amount": st.column_config.NumberColumn("Amount (₹)", min_value=0.01, format="₹%.2f", required=True),
                    "type": st.column_config.SelectboxColumn("Type", options=["expense", "income"], required=True),
                    "category": st.column_config.SelectboxColumn("Category", options=ALL_CATEGORIES, required=True),
                    "payment_mode": st.column_config.SelectboxColumn("Payment Mode", options=["UPI", "Card", "Cash", "Net Banking"]),
                },
                disabled=["id"],
                hide_index=True,
                use_container_width=True,
                key="recent_txns_editor",
            )

            col_save, col_del = st.columns([1, 1.5])
            with col_save:
                if st.button("💾 Save Table Edits", type="secondary", use_container_width=True):
                    # Compare and update modified rows
                    updated_cnt = 0
                    for _, row in edited_table.iterrows():
                        txn_id = int(row["id"])
                        update_transaction(
                            txn_id,
                            date=str(row["date"]),
                            description=str(row["description"]),
                            amount=float(row["amount"]),
                            type=str(row["type"]).lower(),
                            category=str(row["category"]),
                            payment_mode=str(row["payment_mode"]),
                        )
                        updated_cnt += 1
                    st.toast(f"Updated {updated_cnt} transactions!", icon="💾")
                    st.rerun()

            with col_del:
                # Direct Delete Selector
                del_id = st.selectbox(
                    "Delete Transaction",
                    options=display_df["id"].tolist(),
                    format_func=lambda x: f"ID #{x}: {display_df.loc[display_df['id'] == x, 'description'].values[0]} (₹{display_df.loc[display_df['id'] == x, 'amount'].values[0]})",
                    label_visibility="collapsed",
                )
                if st.button("🗑️ Delete Selected Transaction", type="secondary"):
                    delete_transaction(del_id)
                    st.toast(f"Deleted transaction #{del_id}", icon="🗑️")
                    st.rerun()


# -----------------------------------------------------------------------------
# Page 2: Executive Dashboard
# -----------------------------------------------------------------------------
def render_dashboard(df: pd.DataFrame):
    """Render executive financial KPIs, Spotify Wrapped insights, savings goal, and styled charts."""
    hero_banner(
        title="Hey, here's your money story 👋",
        tagline="Real-time cash inflows, burn velocity, and category concentration based exclusively on your real data.",
        badge_text="⚡ LIVE WEALTH SNAPSHOT",
    )

    unlock_hint_card(
        title="Dashboard & Cashflow Analytics",
        current=len(df),
        required=1,
        item_label="transactions",
        unlocked_msg="Analytics engine active! Charts dynamically reflect your recorded transactions.",
        locked_msg="Add at least 1 transaction to view spending trajectories and category distributions.",
        icon="📊",
    )

    kpis = calculate_financial_kpis(df)

    # 4 Glassmorphism KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(
            title="Total Inflow",
            value=format_inr(kpis["total_income"]),
            delta="+Income & Credits" if kpis["total_income"] > 0 else "₹0 Inflow",
            icon="💵",
            color="purple",
        )
    with col2:
        metric_card(
            title="Total Outflow",
            value=format_inr(kpis["total_expenses"]),
            delta=f"-Avg {format_inr(kpis['avg_monthly_expense'])}/mo" if kpis["total_expenses"] > 0 else "₹0 Spend",
            icon="💸",
            color="pink",
        )
    with col3:
        savings_delta = (
            f"+{kpis['savings_rate_pct']:.1f}% Savings Rate"
            if kpis["net_savings"] >= 0 and kpis["total_income"] > 0
            else (f"-{abs(kpis['savings_rate_pct']):.1f}% Deficit" if kpis["total_income"] > 0 else "0.0% Rate")
        )
        metric_card(
            title="Net Savings",
            value=format_inr(kpis["net_savings"]),
            delta=savings_delta,
            icon="💎",
            color="mint",
        )
    with col4:
        metric_card(
            title="Top Spend Category",
            value=kpis["top_category"],
            delta=format_inr(kpis["top_category_amount"]) if kpis["top_category_amount"] > 0 else "₹0.00",
            icon="🔥",
            color="blue",
        )

    if df.empty:
        empty_state(
            "📊✨",
            "Nothing here yet",
            "Add your first transaction to start your money story 🚀",
        )
        if st.button("➕ Add Your First Transaction", type="primary"):
            navigate_to("Add Transaction")
        return

    # Spotify Wrapped Insights Strip
    expenses = df[df["type"] == "expense"].copy() if "type" in df.columns else df.copy()

    top_cat = kpis["top_category"]
    top_cat_amt = format_inr(kpis["top_category_amount"])
    top_cat_share = (
        f"{(kpis['top_category_amount'] / kpis['total_expenses'] * 100):.1f}%"
        if kpis["total_expenses"] > 0
        else "0%"
    )

    if not expenses.empty and "date" in expenses.columns and expenses["date"].notna().any():
        daily_totals = expenses.groupby("date")["amount"].sum()
        max_day = daily_totals.idxmax()
        max_day_val = float(daily_totals.max())
        most_exp_day_str = pd.to_datetime(max_day).strftime("%d %b %Y")
        most_exp_day_amt = format_inr(max_day_val)
    else:
        most_exp_day_str = "N/A"
        most_exp_day_amt = "₹0.00"

    top_merch_df = get_top_merchants(df, top_n=1)
    if not top_merch_df.empty:
        biggest_m = str(top_merch_df.iloc[0]["merchant"])
        biggest_m_amt = format_inr(float(top_merch_df.iloc[0]["total_spend"]))
    else:
        biggest_m = "N/A"
        biggest_m_amt = "₹0.00"

    if not expenses.empty and "date" in expenses.columns and expenses["date"].notna().any():
        latest_period = expenses["date"].dt.to_period("M").max()
        month_expenses = expenses[expenses["date"].dt.to_period("M") == latest_period]
        days_in_month = pd.Period(latest_period, freq="M").days_in_month
        active_days_with_spend = month_expenses["date"].dt.day.nunique()
        no_spend_days = max(0, int(days_in_month - active_days_with_spend))
    else:
        no_spend_days = 0

    spotify_insights_strip(
        top_category=top_cat,
        top_cat_amount=top_cat_amt,
        top_cat_share=top_cat_share,
        most_expensive_day=most_exp_day_str,
        most_expensive_day_amt=most_exp_day_amt,
        biggest_merchant=biggest_m,
        biggest_merchant_amt=biggest_m_amt,
        no_spend_days=no_spend_days,
    )

    # Savings Goal Progress Bar
    savings_goal_progress(current_savings_rate=kpis["savings_rate_pct"], target_rate=20.0)

    # Charts Row 1: Monthly Cashflow Dynamics & Category Donut
    col_left, col_right = st.columns([1.2, 1.0])

    df_viz = df.copy()
    df_viz["month"] = df_viz["date"].dt.strftime("%Y-%m")

    with col_left:
        render_html("""
        <div style="margin-bottom: 8px;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">📅 Monthly Cashflow Dynamics</div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Chronological inflow versus expense burn across calendar months</div>
        </div>
        """)
        monthly_flows = df_viz.groupby(["month", "type"])["amount"].sum().unstack(fill_value=0.0).reset_index()

        fig_bar = go.Figure()
        if "income" in monthly_flows.columns:
            fig_bar.add_trace(go.Bar(
                x=monthly_flows["month"],
                y=monthly_flows["income"],
                name="Income",
                marker=dict(color="#10F0B0", cornerradius=8),
                hovertemplate="<b>Income</b>: ₹%{y:,.2f}<extra></extra>",
            ))
        if "expense" in monthly_flows.columns:
            fig_bar.add_trace(go.Bar(
                x=monthly_flows["month"],
                y=monthly_flows["expense"],
                name="Expense",
                marker=dict(color="#EC4899", cornerradius=8),
                hovertemplate="<b>Expense</b>: ₹%{y:,.2f}<extra></extra>",
            ))
        fig_bar.update_layout(
            barmode="group",
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(title="Amount (₹)", tickprefix="₹"),
        )
        style_fig(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        render_html("""
        <div style="margin-bottom: 8px;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">🍩 Category Distribution</div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Expense concentration split with total outlay in center</div>
        </div>
        """)
        if not expenses.empty and "category" in expenses.columns:
            cat_totals = expenses.groupby("category")["amount"].sum().reset_index()
            fig_donut = px.pie(
                cat_totals,
                names="category",
                values="amount",
                hole=0.65,
            )
            fig_donut.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Amount: ₹%{value:,.2f}<extra></extra>",
            )
            fig_donut.update_layout(
                height=340,
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            style_fig(fig_donut, is_donut=True, donut_total_text=format_inr(kpis["total_expenses"]))
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            empty_state("🍩✨", "No Expense Categories", "Add expense entries to view category proportions.")

    render_html("<div style='margin-top: 14px;'></div>")

    # Charts Row 2: Daily Spending Trend & Top Merchants
    col_trend, col_merch = st.columns([1.2, 1.0])

    with col_trend:
        render_html("""
        <div style="margin-bottom: 8px;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">📈 Daily Burn Trajectory</div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Daily expenditure with 7-day moving average</div>
        </div>
        """)
        if not expenses.empty and "date" in expenses.columns:
            daily_spend = expenses.groupby("date")["amount"].sum().reset_index().sort_values("date")
            daily_spend["rolling_7d"] = daily_spend["amount"].rolling(7, min_periods=1).mean()

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=daily_spend["date"],
                y=daily_spend["amount"],
                mode="lines",
                name="Daily Spend",
                line=dict(color="#3B82F6", width=2.2),
                fill="tozeroy",
                fillcolor="rgba(59, 130, 246, 0.08)",
                hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br>Spend: ₹%{y:,.2f}<extra></extra>",
            ))
            fig_trend.add_trace(go.Scatter(
                x=daily_spend["date"],
                y=daily_spend["rolling_7d"],
                mode="lines",
                name="7-Day Moving Avg",
                line=dict(color="#10F0B0", width=3, dash="solid"),
                hovertemplate="<b>7-Day Avg</b>: ₹%{y:,.2f}<extra></extra>",
            ))
            fig_trend.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(title="Amount (₹)", tickprefix="₹"),
            )
            style_fig(fig_trend)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            empty_state("📈✨", "No Expense History", "Daily trendline appears after adding expense transactions.")

    with col_merch:
        render_html("""
        <div style="margin-bottom: 8px;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">🏆 Top 10 Spending Merchants</div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Entities receiving highest cumulative payment volume</div>
        </div>
        """)
        top_merch_df = get_top_merchants(df, top_n=10)
        if not top_merch_df.empty:
            fig_merch = px.bar(
                top_merch_df.sort_values(by="total_spend", ascending=True),
                x="total_spend",
                y="merchant",
                orientation="h",
                color="total_spend",
                color_continuous_scale=["#3B82F6", "#8B5CF6", "#EC4899"],
            )
            fig_merch.update_layout(
                height=320,
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(title="Total Spend (₹)", tickprefix="₹"),
                yaxis=dict(title=""),
            )
            fig_merch.update_traces(hovertemplate="<b>%{y}</b><br>Outflow: ₹%{x:,.2f}<extra></extra>")
            style_fig(fig_merch)
            st.plotly_chart(fig_merch, use_container_width=True)
        else:
            empty_state("🏆✨", "No Merchants Yet", "Top merchants rank here automatically.")


# -----------------------------------------------------------------------------
# Page 3: Smart Categorization
# -----------------------------------------------------------------------------
def render_categorization(df: pd.DataFrame):
    """Render smart categorization with cold-start rule matcher and auto-trained ML model."""
    section_header(
        title="Smart Transaction Categorization",
        subtitle="Automatic classification from keyword heuristics to trained TF-IDF + Logistic Regression NLP models.",
        emoji="🏷️",
    )

    exp_count = get_expense_count()
    ml_ready = is_ml_model_available()

    # Progress hint card
    if ml_ready:
        unlock_hint_card(
            title="Categorization Engine",
            current=exp_count,
            required=30,
            item_label="categorized expenses",
            unlocked_msg="🚀 ML Model Active! Trained on your personal transactions with TF-IDF and Logistic Regression.",
            icon="🤖",
        )
    else:
        unlock_hint_card(
            title="ML Model Training Progress",
            current=exp_count,
            required=30,
            item_label="categorized expenses",
            unlocked_msg="30+ categorized expenses reached! Ready to train custom ML model.",
            locked_msg=f"Currently using cold-start keyword matcher. Add {max(0, 30 - exp_count)} more categorized expenses to automatically train your customized ML model.",
            icon="🏷️",
        )

    # Retrain button if 30+ expenses
    if exp_count >= 30:
        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("🔄 Retrain ML Model on My Data", use_container_width=True):
                with st.spinner("Retraining NLP pipeline on your real transactions..."):
                    try:
                        metrics = train_model_from_db()
                        st.toast(f"Model retrained with {metrics['accuracy'] * 100:.1f}% accuracy! 🎯", icon="✅")
                        st.success(f"Trained on {metrics['total_samples']} expenses! Test Accuracy: {metrics['accuracy'] * 100:.1f}%")
                    except Exception as e:
                        st.error(f"Retraining note: {e}")
                st.rerun()

    if df.empty:
        empty_state(
            "🏷️📂",
            "No Transactions to Categorize",
            "Add transactions manually or import a CSV to evaluate category classification.",
        )
        if st.button("➕ Go to Add Transaction"):
            navigate_to("Add Transaction")
        return

    # Run categorization on user data
    classified_df = categorize_dataframe(df, description_col="description")

    # Metrics summary
    total_recs = len(classified_df)
    confident_recs = int((classified_df["confidence"] >= 0.50).sum())
    review_recs = total_recs - confident_recs
    avg_conf = float(classified_df["confidence"].mean())

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        metric_card("Total Classified", f"{total_recs:,}", delta="Your Data", icon="📂", color="purple")
    with mcol2:
        metric_card("Confident Matches", f"{confident_recs:,}", delta=f"+{confident_recs/total_recs*100:.1f}% Ready", icon="🎯", color="mint")
    with mcol3:
        metric_card("Needs Review", f"{review_recs:,}", delta=f"{review_recs} Low Conf" if review_recs > 0 else "0 Low Conf", icon="⚠️", color="pink")
    with mcol4:
        engine_label = "ML Pipeline" if ml_ready else "Rule Matcher"
        metric_card("Active Engine", engine_label, delta=f"{avg_conf * 100:.0f}% Avg Conf", icon="⚡", color="blue")

    render_html("<div style='margin-top: 18px;'></div>")

    # Filter view
    filter_choice = st.radio(
        "Filter View",
        ["All Transactions", "Only Low-Confidence / Needs Review", "High Confidence"],
        horizontal=True,
    )

    if filter_choice == "Only Low-Confidence / Needs Review":
        display_df = classified_df[classified_df["confidence"] < 0.50]
    elif filter_choice == "High Confidence":
        display_df = classified_df[classified_df["confidence"] >= 0.50]
    else:
        display_df = classified_df

    if display_df.empty:
        empty_state("🏷️✨", "No Records in this Filter", "Switch to 'All Transactions' to view all categorized entries.")
    else:
        df_show = display_df.copy()
        if "date" in df_show.columns:
            df_show["date"] = pd.to_datetime(df_show["date"]).dt.strftime("%Y-%m-%d")

        st.dataframe(
            df_show[["date", "description", "amount", "predicted_category", "confidence"]],
            column_config={
                "date": st.column_config.TextColumn("Date", width="small"),
                "description": st.column_config.TextColumn("Merchant / Description", width="large"),
                "amount": st.column_config.NumberColumn("Amount (₹)", format="₹%.2f", width="small"),
                "predicted_category": st.column_config.TextColumn("Predicted Category", width="medium"),
                "confidence": st.column_config.ProgressColumn(
                    "Model Confidence",
                    format="%.0f%%",
                    min_value=0.0,
                    max_value=1.0,
                    width="medium",
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

    # Interactive Category Correction Editor (saves to DB)
    render_html("<div style='margin-top: 16px;'></div>")
    with st.expander("✏️ Correct Categories & Teach the Model", expanded=True):
        st.caption("Change any category below and click 'Save Category Corrections'. Every correction is saved to your database so the ML model learns from your feedback!")

        editor_cols = ["id", "description", "amount", "category"]
        valid_editor_cols = [c for c in editor_cols if c in display_df.columns]
        edited_cats = st.data_editor(
            display_df[valid_editor_cols],
            column_config={
                "id": st.column_config.NumberColumn("ID", disabled=True),
                "description": st.column_config.TextColumn("Description", disabled=True),
                "amount": st.column_config.NumberColumn("Amount (₹)", disabled=True, format="₹%.2f"),
                "category": st.column_config.SelectboxColumn("Category", options=ALL_CATEGORIES, required=True),
            },
            hide_index=True,
            use_container_width=True,
            key="cat_correction_editor",
        )

        if st.button("💾 Save Category Corrections to My Data", type="primary"):
            saved_cnt = 0
            for _, row in edited_cats.iterrows():
                update_transaction(int(row["id"]), category=str(row["category"]))
                saved_cnt += 1
            st.toast(f"Saved {saved_cnt} category updates to SQLite!", icon="✅")
            st.rerun()


# -----------------------------------------------------------------------------
# Page 4: Anomaly Alerts
# -----------------------------------------------------------------------------
def render_anomaly_alerts(df: pd.DataFrame):
    """Render unsupervised anomaly detection with 24h duplicate guard and Isolation Forest."""
    section_header(
        title="Spending Anomaly & Outlier Radar",
        subtitle="Detect duplicate debits and irregular spending spikes using Isolation Forest and deterministic guards.",
        emoji="🚨",
    )

    exp_count = get_expense_count()
    is_unlocked = unlock_hint_card(
        title="Isolation Forest Outlier Radar",
        current=exp_count,
        required=20,
        item_label="expense transactions",
        unlocked_msg="Isolation Forest multi-feature outlier radar active!",
        locked_msg=f"Currently running 24h duplicate charge guard. Add {max(0, 20 - exp_count)} more expense transactions to unlock full Isolation Forest statistical outlier radar.",
        icon="🚨",
    )

    if df.empty:
        empty_state(
            "🚨📂",
            "No Transactions to Analyze",
            "Add transactions in the 'Add Transaction' page to monitor for duplicate charges and unusual spikes.",
        )
        if st.button("➕ Go to Add Transaction"):
            navigate_to("Add Transaction")
        return

    # Sensitivity control if unlocked
    contamination = 0.035
    if is_unlocked:
        c1, _ = st.columns([1.5, 2])
        with c1:
            contamination = st.slider(
                "Outlier Sensitivity (Contamination %)",
                min_value=0.01,
                max_value=0.10,
                value=0.035,
                step=0.005,
                help="Higher values flag more subtle outliers; lower values flag only extreme spikes.",
            )

    # Run detection
    anomaly_df = detect_anomalies(df, contamination=contamination)
    flagged_df = anomaly_df[anomaly_df["is_anomaly"] == True].sort_values(by="anomaly_score", ascending=False)

    total_anomalies = len(flagged_df)
    total_amt = float(flagged_df["amount"].sum())
    duplicate_count = int(flagged_df["reason"].str.contains("duplicate", case=False, na=False).sum())
    max_anomaly_amt = float(flagged_df["amount"].max()) if not flagged_df.empty else 0.0

    # Summary cards
    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        metric_card("Flagged Anomalies", f"{total_anomalies}", delta="Detected", icon="🚨", color="pink")
    with scol2:
        metric_card("Flagged Exposure", format_inr(total_amt), delta="Unusual Outflow", icon="⚠️", color="purple")
    with scol3:
        metric_card("Duplicate Charges", f"{duplicate_count}", delta="Potential Refunds" if duplicate_count > 0 else "0 Detected", icon="🔁", color="blue")
    with scol4:
        metric_card("Peak Outlier Spike", format_inr(max_anomaly_amt), delta="Single Largest", icon="⚡", color="mint")

    render_html("<div style='margin-top: 24px;'></div>")

    if is_unlocked:
        # Scatter Plot
        render_html("""
        <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
            📍 Interactive Outlier Radar Map
        </div>
        """)

        normal_df = anomaly_df[anomaly_df["is_anomaly"] == False]
        anom_plot_df = anomaly_df[anomaly_df["is_anomaly"] == True]

        fig_scatter = go.Figure()
        if not normal_df.empty:
            fig_scatter.add_trace(go.Scatter(
                x=normal_df["date"],
                y=normal_df["amount"],
                mode="markers",
                name="Normal Spending",
                marker=dict(color="#10F0B0", size=7, opacity=0.6),
                customdata=np.stack((normal_df["description"], normal_df["category"]), axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>Category: %{customdata[1]}<br>Date: %{x|%d %b %Y}<br>Amount: ₹%{y:,.2f}<extra>Normal</extra>",
            ))

        if not anom_plot_df.empty:
            reasons = anom_plot_df["reason"].fillna("Statistically irregular outlier")
            scores = anom_plot_df["anomaly_score"].round(2)
            fig_scatter.add_trace(go.Scatter(
                x=anom_plot_df["date"],
                y=anom_plot_df["amount"],
                mode="markers",
                name="Anomaly Alert",
                marker=dict(color="#F43F5E", size=13, line=dict(color="#FFA4B6", width=2.5)),
                customdata=np.stack((anom_plot_df["description"], anom_plot_df["category"], reasons, scores), axis=-1),
                hovertemplate="<b>🚨 %{customdata[0]}</b><br>Category: %{customdata[1]}<br>Date: %{x|%d %b %Y}<br>Amount: ₹%{y:,.2f}<br>Reason: <i>%{customdata[2]}</i><extra>ALERT</extra>",
            ))

        fig_scatter.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(title="Amount (₹)", tickprefix="₹"),
        )
        style_fig(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Flagged Anomaly Cards
    st.markdown("##### 📋 Flagged Anomaly Cards")
    if not flagged_df.empty:
        for i in range(0, len(flagged_df), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(flagged_df):
                    row = flagged_df.iloc[i + j]
                    is_dup = "duplicate" in str(row.get("reason", "")).lower()
                    date_str = pd.to_datetime(row["date"]).strftime("%d %b %Y") if pd.notna(row["date"]) else "N/A"
                    amt_str = format_inr(float(row["amount"]))
                    reason_text = str(row.get("reason", "Statistically irregular pattern."))

                    with cols[j]:
                        render_html(f"""
                        <div class="genz-anomaly-card {'anomaly-card-red' if is_dup else 'anomaly-card-orange'}">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 10px;">
                                <div>
                                    <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF;">
                                        {row['description']}
                                    </div>
                                    <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 2px;">
                                        📅 {date_str} &bull; 🏷️ {row['category']}
                                    </div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.25rem; font-weight: 800; color: #F43F5E;">
                                        {amt_str}
                                    </div>
                                    <div>{badge('DUPLICATE' if is_dup else 'SPIKE', 'high' if is_dup else 'warning')}</div>
                                </div>
                            </div>
                            <div style="background: rgba(0,0,0,0.3); border-radius: 10px; padding: 8px 12px; margin-top: 10px; border-left: 3px solid #F43F5E; font-size: 0.85rem; color: #E2E8F0;">
                                {reason_text}
                            </div>
                        </div>
                        """)
    else:
        empty_state(
            "😇✨",
            "Zero Anomalies Detected",
            "No duplicate charges or irregular spikes found in your transactions.",
        )


# -----------------------------------------------------------------------------
# Page 5: Forecast
# -----------------------------------------------------------------------------
def render_forecast(df: pd.DataFrame):
    """Render next-month expenditure forecasting with interactive 'what-if' simulation."""
    section_header(
        title="Predictive Spending Forecast & Budget Simulator",
        subtitle="Linear Regression on time series indices blended with moving averages to project future burn.",
        emoji="🔮",
    )

    distinct_months = get_distinct_months()
    is_unlocked = unlock_hint_card(
        title="Predictive Outflow Forecaster",
        current=distinct_months,
        required=2,
        item_label="distinct months",
        unlocked_msg="Ensemble forecasting active!",
        locked_msg=f"Add data from {max(0, 2 - distinct_months)} more month(s) to unlock forecasting. Time-series models require at least 2 distinct calendar months of history.",
        icon="🔮",
    )

    if distinct_months < 2 or df.empty:
        empty_state(
            "🔮⏳",
            "Forecasting Locked",
            f"You currently have data across {distinct_months} month(s). Outflow forecasting unlocks once transactions span at least 2 distinct months.",
        )
        if st.button("➕ Add More Transactions"):
            navigate_to("Add Transaction")
        return

    # Run forecast
    forecast_df, metrics = forecast_spending(df)

    if forecast_df.empty:
        empty_state("🔮📉", "Insufficient Expense History", "Add expense entries to project category outlays.")
        return

    # Top KPI cards
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        metric_card(
            title="Avg Monthly Inflow",
            value=format_inr(metrics["average_monthly_income"]),
            delta="Real Inflow",
            icon="💵",
            color="purple",
        )
    with fcol2:
        metric_card(
            title="Predicted Outflow",
            value=format_inr(metrics["predicted_total_expense"]),
            delta=f"{metrics['total_change_percent']:+.1f}% vs last mo",
            icon="📊",
            color="pink",
        )
    with fcol3:
        savings_delta = (
            f"+{metrics['projected_savings_rate']:.1f}% Savings Rate"
            if metrics["projected_savings"] >= 0 and metrics["average_monthly_income"] > 0
            else "Negative Margin"
        )
        metric_card(
            title="Projected Savings",
            value=format_inr(metrics["projected_savings"]),
            delta=savings_delta,
            icon="💎",
            color="mint",
        )
    with fcol4:
        metric_card(
            title="Months Analyzed",
            value=f"{metrics['months_analyzed']}",
            delta="Time Horizon",
            icon="⏳",
            color="blue",
        )

    render_html("<div style='margin-top: 24px;'></div>")

    # Bar chart: Last Month Actual vs Predicted Next Month
    render_html("""
    <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
        📊 Category Outflow: Actual vs. Predictive Projection
    </div>
    """)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Bar(
        x=forecast_df["category"],
        y=forecast_df["last_month"],
        name="Last Month Actual",
        marker=dict(color="#3B82F6", cornerradius=8),
        hovertemplate="<b>%{x}</b><br>Last Month: ₹%{y:,.2f}<extra></extra>",
    ))
    fig_fc.add_trace(go.Bar(
        x=forecast_df["category"],
        y=forecast_df["predicted_next_month"],
        name="Predicted Next Month",
        marker=dict(color="#8B5CF6", cornerradius=8),
        hovertemplate="<b>%{x}</b><br>Predicted: ₹%{y:,.2f}<extra></extra>",
    ))
    fig_fc.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(title="Spend (₹)", tickprefix="₹"),
    )
    style_fig(fig_fc)
    st.plotly_chart(fig_fc, use_container_width=True)

    # What-If Simulator
    render_html("<div style='margin-top: 24px;'></div>")
    render_html("""
    <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; margin-bottom: 4px;">
        🎛️ Interactive 'What-If' Spending Optimizer
    </div>
    <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 14px;">
        Simulate how trimming a discretionary category boosts your monthly savings.
    </div>
    """)

    sim_col1, sim_col2 = st.columns([1.1, 1.9])
    with sim_col1:
        target_category = st.selectbox("Category to Trim", options=forecast_df["category"].tolist(), index=0)
        cut_percent = st.slider(f"Reduce {target_category} Spend by", min_value=0, max_value=60, value=25, step=5, format="%d%%")

    orig_cat_spend = float(forecast_df.loc[forecast_df["category"] == target_category, "predicted_next_month"].values[0])
    saved_amount = orig_cat_spend * (cut_percent / 100.0)
    new_total_expense = metrics["predicted_total_expense"] - saved_amount
    new_projected_savings = metrics["average_monthly_income"] - new_total_expense
    new_savings_rate = (
        (new_projected_savings / metrics["average_monthly_income"]) * 100.0
        if metrics["average_monthly_income"] > 0
        else 0.0
    )

    with sim_col2:
        render_html(f"""
        <div style="text-align: center; padding: 20px 22px; background: linear-gradient(135deg, rgba(139, 92, 246, 0.18) 0%, rgba(20, 20, 43, 0.85) 100%); border: 1px solid rgba(139, 92, 246, 0.4); border-radius: 20px; margin-bottom: 14px;">
            <div style="font-size: 0.78rem; font-weight: 800; color: #C084FC; text-transform: uppercase;">
                ✨ Simulated Monthly Savings
            </div>
            <div class="genz-gradient-text" style="font-size: 2.6rem; font-weight: 900; margin: 4px 0;">
                +{format_inr(saved_amount)} / mo
            </div>
            <div style="font-size: 0.92rem; color: #10F0B0; font-weight: 600;">
                Adjusted Savings Rate: {new_savings_rate:.1f}% (Total Outflow: {format_inr(new_total_expense)})
            </div>
        </div>
        """)


# -----------------------------------------------------------------------------
# Page 6: AI Coach
# -----------------------------------------------------------------------------
def render_ai_coach(df: pd.DataFrame):
    """Render Gemini-powered financial plan generator and conversational chat coach."""
    hero_banner(
        title="AI Personal Finance Coach",
        tagline="Personalized financial health scoring, prioritized action cards, and conversational coach powered by Gemini.",
        badge_text="🤖 GEMINI COACH",
    )

    has_both = has_income_and_expense()
    inc_count = len(df[df["type"] == "income"]) if "type" in df.columns else 0
    exp_count = len(df[df["type"] == "expense"]) if "type" in df.columns else 0

    unlock_hint_card(
        title="AI Financial Coach",
        current=1 if has_both else (1 if (inc_count > 0 or exp_count > 0) else 0),
        required=2,
        item_label="record types (1 income + 1 expense)",
        unlocked_msg="AI Coach active! Powered exclusively by your genuine numbers.",
        locked_msg=f"AI Coach requires at least 1 income entry (currently {inc_count}) and 1 expense entry (currently {exp_count}) to evaluate cashflow velocity.",
        icon="🤖",
    )

    if not has_both or df.empty:
        empty_state(
            "🤖💡",
            "Income & Expense Entries Needed",
            f"The AI Coach requires both income and expense records to calculate your financial health score. Currently: {inc_count} income, {exp_count} expense entries.",
        )
        if st.button("➕ Add Transaction Now"):
            navigate_to("Add Transaction")
        return

    # Compute context inputs
    forecast_df, _ = forecast_spending(df)
    anomalies_df = detect_anomalies(df)
    summary = build_financial_summary(df, forecast_df, anomalies_df)

    if "ai_plan" not in st.session_state:
        st.session_state.ai_plan = None
    if "coach_messages" not in st.session_state:
        st.session_state.coach_messages = []

    c_btn, _ = st.columns([1.2, 2.0])
    with c_btn:
        generate_clicked = st.button("✨ Generate My Financial Plan", type="primary", use_container_width=True)

    if generate_clicked or st.session_state.ai_plan is None:
        with st.spinner("Analyzing your real numbers with Gemini AI..."):
            st.session_state.ai_plan = get_recommendations(summary)
            st.toast("AI financial health plan generated!", icon="🤖")

    plan = st.session_state.ai_plan
    health_score = int(plan.get("health_score", 50))

    if health_score >= 80:
        level_label = "Money Level: Legend 👑"
        bar_color = "#10F0B0"
    elif health_score >= 65:
        level_label = "Money Level: Pro 🔥"
        bar_color = "#8B5CF6"
    elif health_score >= 45:
        level_label = "Money Level: Rising ⚡"
        bar_color = "#3B82F6"
    else:
        level_label = "Money Level: Beginner 🌱"
        bar_color = "#F43F5E"

    render_html("<div style='margin-top: 18px;'></div>")
    col_gauge, col_sum = st.columns([1.1, 2.0])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={"text": f"<b>{level_label}</b>", "font": {"size": 17, "color": "#FFFFFF"}},
            number={"suffix": "/100", "font": {"size": 34, "color": "#FFFFFF"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748B", "tickfont": {"color": "#94A3B8"}},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.03)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(244, 63, 94, 0.2)"},
                    {"range": [50, 70], "color": "rgba(251, 146, 60, 0.2)"},
                    {"range": [70, 85], "color": "rgba(139, 92, 246, 0.2)"},
                    {"range": [85, 100], "color": "rgba(16, 240, 176, 0.2)"},
                ],
            },
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
        style_fig(fig_gauge)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_sum:
        summary_text = plan.get("summary", "Reviewing your financial standing...")
        warnings = plan.get("warnings", [])
        warnings_html = "".join([f'<div style="color: #FFA4B6; font-size: 0.85rem; margin-top: 4px;">⚠️ {w}</div>' for w in warnings])

        render_html(f"""
        <div class="genz-glass-card" style="border-left: 5px solid {bar_color}; padding: 20px 24px; min-height: 260px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-bottom: 8px;">
                📌 Executive Assessment
            </div>
            <div style="color: #CBD5E1; line-height: 1.6; font-size: 0.92rem;">
                {summary_text}
            </div>
            <div style="margin-top: 10px;">{warnings_html}</div>
        </div>
        """)

    # Action Items
    render_html("<h4 style='font-weight: 700; color: #FFFFFF; margin-top: 16px; margin-bottom: 12px;'>🎯 Targeted Action Items</h4>")
    recs = plan.get("recommendations", [])
    if recs:
        for i in range(0, len(recs), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(recs):
                    rec = recs[i + j]
                    p = str(rec.get("priority", "medium")).lower()
                    saving_val = rec.get("estimated_monthly_saving_inr", 0)
                    with cols[j]:
                        render_html(f"""
                        <div class="genz-rec-card rec-border-{p}">
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 8px;">
                                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF;">
                                        {rec.get('title')}
                                    </div>
                                    <div>{badge(f"{p.upper()} PRIORITY", p)}</div>
                                </div>
                                <div style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.5;">
                                    {rec.get('action')}
                                </div>
                            </div>
                            <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px; margin-top: 12px; display: flex; justify-content: flex-end;">
                                <span class="glow-save-chip">⚡ Save {format_inr(saving_val)}/mo</span>
                            </div>
                        </div>
                        """)

    # Chat Coach
    render_html("<h4 style='font-weight: 700; color: #FFFFFF; margin-top: 20px; margin-bottom: 8px;'>💬 Chat with Your Coach</h4>")

    for msg in st.session_state.coach_messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["text"])

    col_q1, col_q2, col_q3 = st.columns(3)
    quick_query = None
    with col_q1:
        if st.button("💡 How can I save more?", use_container_width=True):
            quick_query = "How can I save more?"
    with col_q2:
        if st.button("⚠️ Am I overspending?", use_container_width=True):
            quick_query = "Am I overspending?"
    with col_q3:
        if st.button("🎯 Plan my next month", use_container_width=True):
            quick_query = "Plan my next month"

    user_query = st.chat_input("Ask your coach anything based on your real numbers...")
    active_prompt = user_query or quick_query

    if active_prompt:
        st.session_state.coach_messages.append({"role": "user", "text": active_prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(active_prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing your profile..."):
                reply = chat_with_coach(active_prompt, summary, st.session_state.coach_messages)
                st.markdown(reply)
                st.session_state.coach_messages.append({"role": "assistant", "text": reply})


# -----------------------------------------------------------------------------
# Page: CSV Import / Export Studio
# -----------------------------------------------------------------------------
def render_csv_import():
    """Render CSV file uploader, validation, preview, and batch database ingestion."""
    section_header(
        title="Import & Export Transactions",
        subtitle="Bulk import your existing bank, UPI, or card statements via CSV into SQLite.",
        emoji="📥",
    )

    col_temp, col_exp = st.columns([1, 1])
    with col_temp:
        # Blank template download
        template_csv = "date,description,amount,type,category,payment_mode\n2026-05-15,Swiggy Order,350.00,expense,Food & Dining,UPI\n"
        st.download_button(
            label="📄 Download Blank CSV Template",
            data=template_csv,
            file_name="finsight_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_exp:
        all_txns = get_transactions()
        if not all_txns.empty:
            buf = io.StringIO()
            exp_df = all_txns.copy()
            if "date" in exp_df.columns:
                exp_df["date"] = exp_df["date"].dt.strftime("%Y-%m-%d")
            exp_df.to_csv(buf, index=False)
            st.download_button(
                label="📤 Export All My Data as CSV",
                data=buf.getvalue(),
                file_name="finsight_all_transactions.csv",
                mime="text/csv",
                use_container_width=True,
            )

    render_html("<div style='margin-top: 14px;'></div>")

    uploaded_file = st.file_uploader(
        "Upload CSV file (supports YYYY-MM-DD and DD-MM-YYYY dates)",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            raw_cols = [str(c).strip().lower() for c in df_raw.columns]
            df_raw.columns = raw_cols

            required = ["date", "description", "amount"]
            missing = [c for c in required if c not in raw_cols]

            if missing:
                st.error(f"❌ Missing required column(s): {', '.join(missing)}. File must include at least: date, description, amount.")
                return

            # Parse dates
            df_raw["parsed_date"] = parse_flexible_date(df_raw["date"])
            df_raw["parsed_amount"] = pd.to_numeric(df_raw["amount"], errors="coerce")

            invalid_date_count = int(df_raw["parsed_date"].isna().sum())
            invalid_amt_count = int(df_raw["parsed_amount"].isna().sum())
            valid_mask = df_raw["parsed_date"].notna() & df_raw["parsed_amount"].notna() & (df_raw["parsed_amount"] > 0)

            valid_df = df_raw[valid_mask].copy()
            error_count = len(df_raw) - len(valid_df)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                metric_card("Valid Rows", f"{len(valid_df):,}", delta="Ready to Import", icon="✅", color="mint")
            with col_m2:
                metric_card("Rows with Errors", f"{error_count:,}", delta=f"{invalid_date_count} bad date, {invalid_amt_count} bad amt", icon="⚠️", color="pink")

            st.markdown("##### 🔍 Preview of Valid Transactions to Import:")
            preview_display = valid_df[["parsed_date", "description", "parsed_amount"]].copy()
            preview_display["parsed_date"] = preview_display["parsed_date"].dt.strftime("%Y-%m-%d")
            st.dataframe(preview_display.head(20), use_container_width=True)

            if len(valid_df) > 0:
                if st.button("📥 Import to My Data (Skip Exact Duplicates)", type="primary", use_container_width=True):
                    # Prepare batch records
                    records = []
                    for _, row in valid_df.iterrows():
                        t_val = str(row.get("type", "expense")).lower()
                        txn_type = "income" if any(w in t_val for w in ["income", "credit", "salary", "stipend"]) else "expense"
                        cat_val = str(row.get("category", "Uncategorized")) if pd.notna(row.get("category")) else "Uncategorized"
                        mode_val = str(row.get("payment_mode", "UPI")) if pd.notna(row.get("payment_mode")) else "UPI"

                        records.append({
                            "date": row["parsed_date"].strftime("%Y-%m-%d"),
                            "description": str(row["description"]).strip(),
                            "amount": float(row["parsed_amount"]),
                            "type": txn_type,
                            "category": cat_val,
                            "payment_mode": mode_val,
                        })

                    inserted = add_transactions_bulk(records)
                    skipped = len(records) - inserted
                    st.toast(f"Imported {inserted} new transactions (skipped {skipped} duplicates)!", icon="🎉")
                    st.success(f"✓ Successfully imported {inserted} transactions into your database! (Skipped {skipped} exact duplicates).")
                    st.rerun()

        except Exception as e:
            st.error(f"Failed to process CSV file: {e}")


# -----------------------------------------------------------------------------
# Page 7: About / How AI is Used
# -----------------------------------------------------------------------------
def render_about():
    """Render technical documentation of AI and ML models used."""
    section_header(
        title="System Architecture & AI Specifications",
        subtitle="Comprehensive technical breakdown of FinSight AI's progressive machine learning architecture.",
        emoji="ℹ️",
    )

    render_html("""
    <div class="genz-glass-card" style="margin-bottom: 24px;">
        <span style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">⚡ Zero-Synthetic Persistent Architecture</span>
        <p style="color: #CBD5E1; margin-top: 8px; margin-bottom: 0; line-height: 1.6;">
            <b>FinSight AI</b> is engineered with a strict <b>zero synthetic data</b> policy. The application starts
            completely empty and persists all user-entered transactions inside a local SQLite database (<code>data/finsight.db</code>).
            AI/ML features progressively unlock as your genuine financial footprint expands.
        </p>
    </div>
    """)

    col1, col2 = st.columns(2)

    with col1:
        render_html(r"""
        <div class="genz-glass-card" style="border-left: 5px solid #8B5CF6; margin-bottom: 20px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-bottom: 8px;">
                🏷️ 1. Progressive Categorization
            </div>
            <div style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;">
                <p><b>Phase 1 (0-29 Expenses)</b>: Deterministic keyword matcher instantly categorizes popular merchants (Swiggy, Zomato, Uber, Metro, etc.).</p>
                <p><b>Phase 2 (30+ Expenses)</b>: Automatically trains a <code>TfidfVectorizer</code> (word & char n-grams) + <code>LogisticRegression</code> model on your personal data.</p>
                <p><b>Continuous Learning</b>: User category overrides are saved to SQLite to retrain and refine the model.</p>
            </div>
        </div>
        """)

        render_html(r"""
        <div class="genz-glass-card" style="border-left: 5px solid #3B82F6; margin-bottom: 20px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-bottom: 8px;">
                🔮 3. Expenditure Forecaster
            </div>
            <div style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;">
                <p><b>Unlock Threshold</b>: Requires at least 2 distinct calendar months of transaction history.</p>
                <p><b>Model</b>: Single-variable Linear Regression on historical monthly sequence averaged with a 3-month moving average.</p>
                <p><b>What-If Simulator</b>: Simulates discretionary cuts to calculate exact rupee savings and adjusted savings rates.</p>
            </div>
        </div>
        """)

    with col2:
        render_html(r"""
        <div class="genz-glass-card" style="border-left: 5px solid #F43F5E; margin-bottom: 20px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-bottom: 8px;">
                🚨 2. Anomaly Detection
            </div>
            <div style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;">
                <p><b>Duplicate Guard (Instant)</b>: Scans for duplicate transactions (same merchant & amount within 24 hours) from transaction #1.</p>
                <p><b>Isolation Forest (20+ Expenses)</b>: Unsupervised multi-feature outlier scoring with 3-sigma thresholds.</p>
                <p><b>Human-Readable Reasons</b>: Every flagged anomaly includes an explainable root-cause diagnosis.</p>
            </div>
        </div>
        """)

        render_html(r"""
        <div class="genz-glass-card" style="border-left: 5px solid #10F0B0; margin-bottom: 20px;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-bottom: 8px;">
                🤖 4. Generative AI Coach
            </div>
            <div style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;">
                <p><b>Unlock Threshold</b>: Activates once you have at least 1 income and 1 expense record.</p>
                <p><b>Model</b>: Google Gemini 2.5 Flash via official <code>google-genai</code> SDK.</p>
                <p><b>Strict Grounding</b>: Given only your real numbers. Zero fake assumptions.</p>
                <p><b>Fallback Guarantee</b>: Deterministic 50/30/20 rule engine when offline.</p>
            </div>
        </div>
        """)


# -----------------------------------------------------------------------------
# Main Application Router
# -----------------------------------------------------------------------------
def main():
    """Application entry point and routing coordinator."""
    selected_page, data_source_mode, df = render_sidebar()

    if data_source_mode == "Import my CSV":
        render_csv_import()
    else:
        if selected_page == "Add Transaction":
            render_add_transaction()
        elif selected_page == "Dashboard":
            render_dashboard(df)
        elif selected_page == "Smart Categorization":
            render_categorization(df)
        elif selected_page == "Anomaly Alerts":
            render_anomaly_alerts(df)
        elif selected_page == "Forecast":
            render_forecast(df)
        elif selected_page == "AI Coach":
            render_ai_coach(df)
        elif selected_page == "About / How AI is Used":
            render_about()

    render_footer()


if __name__ == "__main__":
    main()
