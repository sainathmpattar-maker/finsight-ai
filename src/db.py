"""FinSight AI - SQLite Persistent Storage Layer.

Provides CRUD operations for the transactions table backed by a local SQLite
database (data/finsight.db).  The database and table are created automatically
on first run so the app starts with a completely empty slate.
"""

import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "finsight.db")


def _get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database, creating dir if needed."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create the transactions table if it does not already exist."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT NOT NULL,
            description TEXT NOT NULL,
            amount      REAL NOT NULL,
            type        TEXT NOT NULL CHECK(type IN ('income', 'expense')),
            category    TEXT NOT NULL DEFAULT 'Uncategorized',
            payment_mode TEXT NOT NULL DEFAULT 'UPI',
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


def add_transaction(
    date: str,
    description: str,
    amount: float,
    txn_type: str,
    category: str = "Uncategorized",
    payment_mode: str = "UPI",
) -> int:
    """Insert a single transaction and return its row id."""
    conn = _get_connection()
    cur = conn.execute(
        """
        INSERT INTO transactions (date, description, amount, type, category, payment_mode, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (date, description.strip(), round(amount, 2), txn_type.lower(),
         category.strip(), payment_mode.strip(), datetime.now().isoformat()),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def add_transactions_bulk(rows: List[Dict[str, Any]]) -> int:
    """Insert multiple transactions at once, skipping exact duplicates.

    A duplicate is defined as same date + description + amount + type.
    Returns the count of newly inserted rows.
    """
    conn = _get_connection()
    inserted = 0
    for r in rows:
        # Check for exact duplicate
        existing = conn.execute(
            """
            SELECT 1 FROM transactions
            WHERE date = ? AND description = ? AND amount = ? AND type = ?
            LIMIT 1
            """,
            (r["date"], r["description"].strip(), round(r["amount"], 2), r["type"].lower()),
        ).fetchone()
        if existing:
            continue
        conn.execute(
            """
            INSERT INTO transactions (date, description, amount, type, category, payment_mode, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["date"],
                r["description"].strip(),
                round(r["amount"], 2),
                r["type"].lower(),
                r.get("category", "Uncategorized").strip(),
                r.get("payment_mode", "UPI").strip(),
                datetime.now().isoformat(),
            ),
        )
        inserted += 1
    conn.commit()
    conn.close()
    return inserted


def get_transactions() -> pd.DataFrame:
    """Return all transactions as a cleaned pandas DataFrame, sorted by date."""
    conn = _get_connection()
    df = pd.read_sql_query("SELECT * FROM transactions ORDER BY date ASC, id ASC", conn)
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=[
            "id", "date", "description", "amount", "type", "category", "payment_mode", "created_at",
        ])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df


def update_transaction(txn_id: int, **fields) -> None:
    """Update specific fields of a transaction by its id.

    Accepted keyword arguments: date, description, amount, type, category, payment_mode.
    """
    allowed = {"date", "description", "amount", "type", "category", "payment_mode"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [txn_id]
    conn = _get_connection()
    conn.execute(f"UPDATE transactions SET {set_clause} WHERE id = ?", values)
    conn.commit()
    conn.close()


def delete_transaction(txn_id: int) -> None:
    """Delete a single transaction by its id."""
    conn = _get_connection()
    conn.execute("DELETE FROM transactions WHERE id = ?", (txn_id,))
    conn.commit()
    conn.close()


def clear_all_data() -> None:
    """Delete ALL transactions from the database."""
    conn = _get_connection()
    conn.execute("DELETE FROM transactions")
    conn.commit()
    conn.close()


def get_transaction_count() -> int:
    """Return the total number of transactions."""
    conn = _get_connection()
    count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    conn.close()
    return count


def get_expense_count() -> int:
    """Return the number of expense transactions."""
    conn = _get_connection()
    count = conn.execute("SELECT COUNT(*) FROM transactions WHERE type = 'expense'").fetchone()[0]
    conn.close()
    return count


def get_distinct_months() -> int:
    """Return the number of distinct year-month periods in the data."""
    conn = _get_connection()
    count = conn.execute(
        "SELECT COUNT(DISTINCT strftime('%Y-%m', date)) FROM transactions"
    ).fetchone()[0]
    conn.close()
    return count


def has_income_and_expense() -> bool:
    """Return True if the DB has at least 1 income AND 1 expense transaction."""
    conn = _get_connection()
    inc = conn.execute("SELECT 1 FROM transactions WHERE type='income' LIMIT 1").fetchone()
    exp = conn.execute("SELECT 1 FROM transactions WHERE type='expense' LIMIT 1").fetchone()
    conn.close()
    return inc is not None and exp is not None

