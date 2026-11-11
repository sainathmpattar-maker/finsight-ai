"""FinSight AI - Synthetic Financial Data Generator.

This module generates realistic synthetic personal finance transactions over the
last 6 months for an Indian college student / young professional.

Outputs:
1. data/transactions.csv (Full labeled dataset with categories)
2. data/transactions_uncategorized.csv (Dataset with 'category' dropped for ML classification demo)
"""

import os
import random
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


# Category taxonomy
CATEGORIES = [
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
]

# Realistic Indian Merchant templates
MERCHANT_TEMPLATES: Dict[str, List[Tuple[str, float, float, List[str]]]] = {
    # format: (description_pattern, min_amount, max_amount, allowed_payment_modes)
    "Food & Dining": [
        ("SWIGGY ORDER {id}", 110.0, 480.0, ["UPI", "Card"]),
        ("UPI-ZOMATO-{id}", 140.0, 520.0, ["UPI", "Card"]),
        ("ZOMATO RESTAURANT #{id}", 180.0, 650.0, ["UPI", "Card"]),
        ("CHAI POINT {id}", 40.0, 160.0, ["UPI", "Cash"]),
        ("TAPRI CHAI & BUN MASKA", 25.0, 80.0, ["UPI", "Cash"]),
        ("COLLEGE CANTEEN SNACKS", 35.0, 120.0, ["UPI", "Cash"]),
        ("DOMINOS PIZZA #{id}", 280.0, 750.0, ["UPI", "Card"]),
        ("MCDONALDS BURGER-{id}", 190.0, 490.0, ["UPI", "Card"]),
        ("UDUPI SAGAR VEG TIFFIN", 60.0, 180.0, ["UPI", "Cash"]),
        ("STARBUCKS COFFEE IND-{id}", 250.0, 480.0, ["Card", "UPI"]),
    ],
    "Groceries": [
        ("BLINKIT QUICK GROCERY-{id}", 95.0, 680.0, ["UPI", "Card"]),
        ("ZEPTO DELIVERY #{id}", 80.0, 550.0, ["UPI", "Card"]),
        ("INSTAMART SWIGGY-{id}", 110.0, 720.0, ["UPI", "Card"]),
        ("BIGBASKET DAILY-{id}", 250.0, 1400.0, ["UPI", "Card"]),
        ("LOCAL KIRANA STORE", 50.0, 450.0, ["UPI", "Cash"]),
        ("NATURE FRESH VEGETABLES", 40.0, 220.0, ["UPI", "Cash"]),
        ("MORE RETAIL SUPERMARKET", 180.0, 850.0, ["Card", "UPI"]),
    ],
    "Transport": [
        ("UBER TRIP HELP-{id}", 65.0, 360.0, ["UPI", "Card"]),
        ("OLA CABS RIDE-{id}", 70.0, 340.0, ["UPI", "Cash"]),
        ("RAPIDO BIKE TAXI-{id}", 35.0, 110.0, ["UPI"]),
        ("METRO SMART CARD RECHARGE", 100.0, 400.0, ["UPI", "Card"]),
        ("AUTO RICKSHAW FARE UPI", 40.0, 150.0, ["UPI", "Cash"]),
        ("NMIT CAMPUS SHUTTLE PASS", 150.0, 300.0, ["UPI", "Cash"]),
    ],
    "Shopping": [
        ("AMAZON PAY RETAIL-{id}", 249.0, 2400.0, ["Card", "UPI"]),
        ("AMAZON.IN ORDER #{id}", 350.0, 1899.0, ["Card", "UPI"]),
        ("FLIPKART INTERNET-{id}", 299.0, 2200.0, ["UPI", "Card"]),
        ("MYNTRA FASHION STORE-{id}", 499.0, 2500.0, ["Card", "UPI"]),
        ("ZUDIO APPARELS", 399.0, 1299.0, ["Card", "UPI"]),
        ("DECATHLON SPORTS IND", 299.0, 1800.0, ["Card", "UPI"]),
        ("MEESHO ONLINE SHOPPING", 199.0, 750.0, ["UPI"]),
    ],
    "Entertainment": [
        ("BOOKMYSHOW CINEMA-{id}", 220.0, 650.0, ["UPI", "Card"]),
        ("PVR INOX TICKETS-{id}", 250.0, 700.0, ["UPI", "Card"]),
        ("STEAM DIGITAL GAME-{id}", 249.0, 899.0, ["Card"]),
        ("COLLEGE CULTURAL FEST PASS", 150.0, 500.0, ["UPI", "Cash"]),
        ("SMASH GAMING & BOWLING", 300.0, 850.0, ["Card", "UPI"]),
    ],
    "Health": [
        ("APOLLO PHARMACY-{id}", 60.0, 650.0, ["UPI", "Card", "Cash"]),
        ("MEDPLUS DRUG STORE-{id}", 80.0, 580.0, ["UPI", "Card", "Cash"]),
        ("TATA 1MG ORDER #{id}", 150.0, 890.0, ["UPI", "Card"]),
        ("CLINIC DOCTOR CONSULTATION", 300.0, 600.0, ["UPI", "Cash"]),
        ("EYE PLUS OPTICALS", 200.0, 950.0, ["Card", "UPI"]),
    ],
    "Education": [
        ("UDEMY ONLINE COURSE #{id}", 389.0, 599.0, ["Card", "UPI"]),
        ("COURSERA LEARNING PASS", 499.0, 1200.0, ["Card"]),
        ("COLLEGE XEROX & STATIONERY", 20.0, 180.0, ["UPI", "Cash"]),
        ("SAPNA BOOK HOUSE BLR", 150.0, 650.0, ["UPI", "Card"]),
        ("PROJECT PRINTING & BINDING", 60.0, 250.0, ["UPI", "Cash"]),
    ],
    "Subscriptions": [
        ("YOUTUBE PREMIUM INDIVIDUAL", 129.0, 149.0, ["Card", "UPI"]),
        ("GOOGLE ONE 100GB STORAGE", 130.0, 130.0, ["Card"]),
        ("APPLE ICLOUD 50GB", 75.0, 75.0, ["Card"]),
        ("CHATGPT PLUS SUBSCRIPTION", 1650.0, 1650.0, ["Card"]),
    ],
    "Bills & Utilities": [
        ("HP GAS CYLINDER REFILL", 850.0, 920.0, ["UPI", "Card"]),
        ("LAUNDRY DRYCLEAN SERVICES", 120.0, 380.0, ["UPI", "Cash"]),
        ("WATER CAN REFILL 20L", 40.0, 90.0, ["Cash", "UPI"]),
    ],
    "Travel": [
        ("IRCTC TRAIN TICKET-{id}", 240.0, 1450.0, ["UPI", "Card"]),
        ("REDBUS TRAVELS BOOKING", 350.0, 1200.0, ["UPI", "Card"]),
        ("ABHIBUS INTERCITY TICKET", 300.0, 950.0, ["UPI"]),
    ],
}


def _generate_random_id(length: int = 4) -> str:
    """Generate numeric string suffix for transaction reference numbers."""
    return "".join(random.choices("0123456789", k=length))


def generate_synthetic_transactions(
    start_date: str = "2026-05-01",
    end_date: str = "2026-10-31",
    target_count: int = 600,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic financial transactions for an Indian student/young professional.

    Parameters
    ----------
    start_date : str, optional
        Start date 'YYYY-MM-DD' (default "2026-05-01", 6-month window).
    end_date : str, optional
        End date 'YYYY-MM-DD' (default "2026-10-31").
    target_count : int, optional
        Target number of records (default ~600).
    random_seed : int, optional
        Fixed seed for reproducibility (default 42).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [date, description, amount, type, category, payment_mode]
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    total_days = (end_dt - start_dt).days + 1

    records: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. Monthly Recurring Income (Stipend / Salary on 1st of month)
    # -------------------------------------------------------------
    current_month_cursor = date(start_dt.year, start_dt.month, 1)
    while current_month_cursor <= end_dt:
        if current_month_cursor >= start_dt:
            stipend_amount = 25000.0 + random.choice([0.0, 500.0, 1000.0])
            records.append({
                "date": current_month_cursor.strftime("%Y-%m-%d"),
                "description": f"SALARY CREDIT - TECHCORP STIPEND #{_generate_random_id(3)}",
                "amount": round(stipend_amount, 2),
                "type": "income",
                "category": "Salary",
                "payment_mode": "Card",
            })
        # Advance 1 month
        next_month = current_month_cursor.month + 1
        next_year = current_month_cursor.year
        if next_month > 12:
            next_month = 1
            next_year += 1
        current_month_cursor = date(next_year, next_month, 1)

    # -------------------------------------------------------------
    # 2. Occasional Freelance Income (1-2 per month)
    # -------------------------------------------------------------
    freelance_clients = [
        "UPI-FREELANCE-WEBDEV",
        "UPWORK ESCROW PAYOUT",
        "FREELANCE UI/UX CLIENT PAY",
        "DATA ANNOTATION STIPEND",
        "PYTHON TUTORING REMITTANCE",
    ]
    # Spread ~8-10 freelance gigs over the 6 months
    num_freelance = 9
    freelance_days = sorted(random.sample(range(5, total_days - 2), num_freelance))
    for day_offset in freelance_days:
        f_date = start_dt + timedelta(days=day_offset)
        client = random.choice(freelance_clients)
        f_amount = round(random.uniform(3200.0, 8500.0), -2)  # round to nearest 100
        records.append({
            "date": f_date.strftime("%Y-%m-%d"),
            "description": f"{client} #{_generate_random_id(4)}",
            "amount": float(f_amount),
            "type": "income",
            "category": "Freelance",
            "payment_mode": "UPI",
        })

    # -------------------------------------------------------------
    # 3. Monthly Fixed Recurring Expenses
    # -------------------------------------------------------------
    month_cursor = date(start_dt.year, start_dt.month, 1)
    while month_cursor <= end_dt:
        y, m = month_cursor.year, month_cursor.month

        # Rent on 5th
        rent_date = date(y, m, 5)
        if start_dt <= rent_date <= end_dt:
            records.append({
                "date": rent_date.strftime("%Y-%m-%d"),
                "description": "UPI-RENT-FLAT502-KUMAR",
                "amount": 8500.0,
                "type": "expense",
                "category": "Rent",
                "payment_mode": "UPI",
            })

        # WiFi Broadband on 10th
        wifi_date = date(y, m, 10)
        if start_dt <= wifi_date <= end_dt:
            records.append({
                "date": wifi_date.strftime("%Y-%m-%d"),
                "description": "ACT FIBERNET BROADBAND BILL",
                "amount": 799.0,
                "type": "expense",
                "category": "Bills & Utilities",
                "payment_mode": "UPI",
            })

        # Electricity BESCOM on 12th
        elec_date = date(y, m, 12)
        if start_dt <= elec_date <= end_dt:
            records.append({
                "date": elec_date.strftime("%Y-%m-%d"),
                "description": "BESCOM ELECTRICITY BILLPAY",
                "amount": round(random.uniform(620.0, 940.0), 2),
                "type": "expense",
                "category": "Bills & Utilities",
                "payment_mode": "UPI",
            })

        # Netflix on 15th
        netflix_date = date(y, m, 15)
        if start_dt <= netflix_date <= end_dt:
            records.append({
                "date": netflix_date.strftime("%Y-%m-%d"),
                "description": "NETFLIX RECURRING SUBSCRIPTION",
                "amount": 199.0,
                "type": "expense",
                "category": "Subscriptions",
                "payment_mode": "Card",
            })

        # Spotify on 15th
        spotify_date = date(y, m, 15)
        if start_dt <= spotify_date <= end_dt:
            records.append({
                "date": spotify_date.strftime("%Y-%m-%d"),
                "description": "SPOTIFY INDIA AUTO-DEBIT",
                "amount": 119.0,
                "type": "expense",
                "category": "Subscriptions",
                "payment_mode": "Card",
            })

        # Mobile Recharge on 20th
        mobile_date = date(y, m, 20)
        if start_dt <= mobile_date <= end_dt:
            records.append({
                "date": mobile_date.strftime("%Y-%m-%d"),
                "description": "AIRTEL PREPAID 84D PLAN",
                "amount": 299.0,
                "type": "expense",
                "category": "Bills & Utilities",
                "payment_mode": "UPI",
            })

        # Advance month
        next_m = m + 1
        next_y = y
        if next_m > 12:
            next_m = 1
            next_y += 1
        month_cursor = date(next_y, next_m, 1)

    # -------------------------------------------------------------
    # 4. Inject 12 Deliberate Anomalies (for ML anomaly demonstration)
    # -------------------------------------------------------------
    # We will pick specific dates within the range for each anomaly.
    anomalies: List[Dict[str, Any]] = [
        # 1. Massive Shopping purchase (Gadget spike)
        {
            "date": (start_dt + timedelta(days=22)).strftime("%Y-%m-%d"),
            "description": "AMAZON.IN - APPLE IPAD AIR 64GB PURCHASE",
            "amount": 49999.0,
            "type": "expense",
            "category": "Shopping",
            "payment_mode": "Card",
        },
        # 2. Luxury Nightlife / Pub VIP Lounge spike
        {
            "date": (start_dt + timedelta(days=45)).strftime("%Y-%m-%d"),
            "description": "LATE NIGHT SOCIAL PUB & BREWERY - VIP LOUNGE",
            "amount": 14850.0,
            "type": "expense",
            "category": "Food & Dining",
            "payment_mode": "Card",
        },
        # 3 & 4. Accidental Duplicate Swiggy charge (same day, same amount, same order ID)
        {
            "date": (start_dt + timedelta(days=68)).strftime("%Y-%m-%d"),
            "description": "SWIGGY ORDER 8842",
            "amount": 540.0,
            "type": "expense",
            "category": "Food & Dining",
            "payment_mode": "UPI",
        },
        {
            "date": (start_dt + timedelta(days=68)).strftime("%Y-%m-%d"),
            "description": "SWIGGY ORDER 8842",
            "amount": 540.0,
            "type": "expense",
            "category": "Food & Dining",
            "payment_mode": "UPI",
        },
        # 5. Outlier Flight booking to Goa
        {
            "date": (start_dt + timedelta(days=85)).strftime("%Y-%m-%d"),
            "description": "MAKEMYTRIP GOA ROUNDTRIP FLIGHT TICKET",
            "amount": 16400.0,
            "type": "expense",
            "category": "Travel",
            "payment_mode": "Card",
        },
        # 6 & 7. Accidental Duplicate Uber ride charge
        {
            "date": (start_dt + timedelta(days=102)).strftime("%Y-%m-%d"),
            "description": "UBER TRIP HELP-9133",
            "amount": 345.0,
            "type": "expense",
            "category": "Transport",
            "payment_mode": "UPI",
        },
        {
            "date": (start_dt + timedelta(days=102)).strftime("%Y-%m-%d"),
            "description": "UBER TRIP HELP-9133",
            "amount": 345.0,
            "type": "expense",
            "category": "Transport",
            "payment_mode": "UPI",
        },
        # 8. Unusually massive bulk grocery hoard
        {
            "date": (start_dt + timedelta(days=118)).strftime("%Y-%m-%d"),
            "description": "BIGBASKET BULK WHOLESALE PANTRY ORDER",
            "amount": 9250.0,
            "type": "expense",
            "category": "Groceries",
            "payment_mode": "Card",
        },
        # 9. Expensive certification / tech software package
        {
            "date": (start_dt + timedelta(days=135)).strftime("%Y-%m-%d"),
            "description": "JETBRAINS / AWS SOLUTIONS ARCHITECT EXAM VOUCHER",
            "amount": 12800.0,
            "type": "expense",
            "category": "Education",
            "payment_mode": "Card",
        },
        # 10. High-value gift jewelry purchase
        {
            "date": (start_dt + timedelta(days=150)).strftime("%Y-%m-%d"),
            "description": "TANISHQ JEWELLERY FESTIVAL GIFT",
            "amount": 21500.0,
            "type": "expense",
            "category": "Shopping",
            "payment_mode": "Card",
        },
        # 11. Emergency medical care charge
        {
            "date": (start_dt + timedelta(days=162)).strftime("%Y-%m-%d"),
            "description": "MANIPAL HOSPITAL EMERGENCY CARE & TESTS",
            "amount": 11500.0,
            "type": "expense",
            "category": "Health",
            "payment_mode": "UPI",
        },
        # 12 & 13. Duplicate Broadband Bill Payment in same billing cycle
        {
            "date": (start_dt + timedelta(days=171)).strftime("%Y-%m-%d"),
            "description": "ACT FIBERNET BROADBAND BILLPAY-DUP",
            "amount": 799.0,
            "type": "expense",
            "category": "Bills & Utilities",
            "payment_mode": "UPI",
        },
        {
            "date": (start_dt + timedelta(days=171)).strftime("%Y-%m-%d"),
            "description": "ACT FIBERNET BROADBAND BILLPAY-DUP",
            "amount": 799.0,
            "type": "expense",
            "category": "Bills & Utilities",
            "payment_mode": "UPI",
        },
    ]
    records.extend(anomalies)

    # -------------------------------------------------------------
    # 5. Variable Day-to-Day Expenses to reach ~600 records
    # -------------------------------------------------------------
    # Category weights reflecting an Indian student / young professional lifestyle
    category_weights = {
        "Food & Dining": 0.35,     # Swiggy/Zomato/canteen/chai most frequent
        "Transport": 0.18,         # Metro, autos, Rapido, Uber
        "Groceries": 0.14,         # Zepto, Blinkit, local kirana
        "Shopping": 0.10,          # Amazon, clothes, stationery
        "Entertainment": 0.08,     # Movies, gaming, outings
        "Health": 0.04,            # Pharmacy, minor meds
        "Education": 0.04,         # Xerox, books, small courses
        "Travel": 0.03,            # Train / weekend bus trips
        "Bills & Utilities": 0.02, # Incidental bills (laundry, gas)
        "Subscriptions": 0.02,     # YouTube premium, etc.
    }

    variable_target = target_count - len(records)
    if variable_target < 100:
        variable_target = 520

    weighted_categories = list(category_weights.keys())
    probs = [category_weights[c] for c in weighted_categories]

    # Generate variable transactions distributed across dates
    for _ in range(variable_target):
        # Pick random day within span
        day_offset = random.randint(0, total_days - 1)
        tx_date = start_dt + timedelta(days=day_offset)

        cat = np.random.choice(weighted_categories, p=probs)
        templates = MERCHANT_TEMPLATES[cat]
        template, min_amt, max_amt, modes = random.choice(templates)

        # Build description
        rand_id = _generate_random_id(4)
        desc = template.format(id=rand_id)

        # Amount with realistic distribution (log-normal skew toward lower end)
        # using uniform between min and max for simplicity, rounded
        amt = round(random.uniform(min_amt, max_amt), 2)
        if amt > 200:
            # Round higher amounts to nice numbers
            amt = round(amt / 10.0) * 10.0

        mode = random.choice(modes)

        records.append({
            "date": tx_date.strftime("%Y-%m-%d"),
            "description": desc,
            "amount": float(amt),
            "type": "expense",
            "category": cat,
            "payment_mode": mode,
        })

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Sort chronologically by date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Reorder columns as requested: date, description, amount, type, category, payment_mode
    cols = ["date", "description", "amount", "type", "category", "payment_mode"]
    df = df[cols]

    return df


def generate_and_save_datasets(
    output_dir: str = "data",
    target_count: int = 600,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and save both the labeled and uncategorized transaction CSVs.

    Parameters
    ----------
    output_dir : str, optional
        Target directory to save CSVs (default "data").
    target_count : int, optional
        Total target number of transactions (default 600).
    random_seed : int, optional
        Fixed seed for reproducibility (default 42).

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (labeled_df, uncategorized_df)
    """
    os.makedirs(output_dir, exist_ok=True)

    df_labeled = generate_synthetic_transactions(
        start_date="2026-05-01",
        end_date="2026-10-31",
        target_count=target_count,
        random_seed=random_seed,
    )

    # 1. Save labeled dataset
    labeled_path = os.path.join(output_dir, "transactions.csv")
    df_labeled.to_csv(labeled_path, index=False)

    # 2. Save uncategorized dataset (dropping 'category' column)
    df_uncategorized = df_labeled.drop(columns=["category"])
    uncategorized_path = os.path.join(output_dir, "transactions_uncategorized.csv")
    df_uncategorized.to_csv(uncategorized_path, index=False)

    return df_labeled, df_uncategorized


if __name__ == "__main__":
    print("=" * 65)
    print("  FinSight AI - Generating Synthetic Financial Datasets")
    print("=" * 65)

    df_full, df_un = generate_and_save_datasets()

    print(f"\n[+] Successfully generated {len(df_full)} total transactions.")
    print(f"    - Date Range: {df_full['date'].min()} to {df_full['date'].max()}")
    print(f"    - Total Income Records: {(df_full['type'] == 'income').sum()} | Amount: Rs {df_full[df_full['type'] == 'income']['amount'].sum():,.2f}")
    print(f"    - Total Expense Records: {(df_full['type'] == 'expense').sum()} | Amount: Rs {df_full[df_full['type'] == 'expense']['amount'].sum():,.2f}")
    print(f"\n[+] Saved labeled dataset to: data/transactions.csv")
    print(f"[+] Saved uncategorized dataset to: data/transactions_uncategorized.csv")

    print("\n--- Category Breakdown (Expense) ---")
    cat_summary = (
        df_full[df_full["type"] == "expense"]
        .groupby("category")
        .agg(count=("amount", "count"), total_spend=("amount", "sum"))
        .sort_values(by="count", ascending=False)
    )
    print(cat_summary.to_string())

    print("\n--- Payment Mode Distribution ---")
    pm_summary = df_full["payment_mode"].value_counts()
    print(pm_summary.to_string())

    print("\n--- Sample First 10 Transactions ---")
    print(df_full.head(10).to_string(index=False))
