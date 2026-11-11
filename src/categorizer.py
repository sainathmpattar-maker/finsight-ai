"""FinSight AI - Machine Learning Transaction Categorization.

This module automates the classification of raw bank/card/UPI transaction descriptions
into structured budget categories using an NLP TF-IDF feature pipeline and
Supervised Logistic Regression.

Features:
- Rule-based keyword matcher as a cold-start fallback (works from 0 data).
- Character n-grams (2-5) + Word n-grams (1-2) via FeatureUnion.
- Text cleaning removing transaction IDs, digits, and noise.
- Auto-trains ML model once user has 30+ categorized expenses.
- Confidence scoring with low-confidence fallback (< 0.5 -> "Uncategorized / Needs Review").
- Serialized model persistence via joblib using native scikit-learn pipeline components.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

# Module-level model cache to avoid repeated disk reads
_MODEL_CACHE: Dict[str, Any] = {}
UNCATEGORIZED_LABEL = "Uncategorized / Needs Review"

# ---------------------------------------------------------------------------
# Rule-Based Keyword Matcher (cold-start, works with 0 training data)
# ---------------------------------------------------------------------------
KEYWORD_RULES: Dict[str, List[str]] = {
    "Food & Dining": [
        "swiggy", "zomato", "dominos", "pizza", "mcdonalds", "burger", "kfc",
        "starbucks", "cafe", "coffee", "chai", "canteen", "restaurant", "biryani",
        "subway", "taco", "ice cream", "bakery", "mess", "tiffin", "dhaba",
        "food", "dining", "barbeque", "dine", "eat", "meal", "lunch", "dinner",
        "breakfast", "snack", "tapri", "udupi", "haldiram", "momo",
    ],
    "Groceries": [
        "blinkit", "zepto", "instamart", "bigbasket", "grocery", "kirana",
        "vegetable", "fruit", "dmart", "more retail", "supermarket", "provisions",
        "nature fresh", "pantry", "milk", "atta", "dal",
    ],
    "Transport": [
        "uber", "ola", "rapido", "metro", "auto", "rickshaw", "taxi", "cab",
        "shuttle", "petrol", "fuel", "parking", "toll", "bike", "ride",
        "transport", "namma", "yulu",
    ],
    "Rent": [
        "rent", "flat", "pg ", "hostel", "room", "landlord", "lease",
    ],
    "Bills & Utilities": [
        "electricity", "bescom", "water", "gas cylinder", "broadband", "wifi",
        "fibernet", "laundry", "dry clean", "billpay", "recharge", "airtel",
        "jio", "vodafone", "vi ", "postpaid", "prepaid", "mobile plan",
        "internet", "act ", "bsnl",
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "meesho", "ajio", "zudio", "decathlon",
        "fashion", "apparel", "shoes", "clothing", "mall", "retail",
        "shopping", "h&m", "lifestyle", "westside", "jewel", "tanishq",
    ],
    "Entertainment": [
        "bookmyshow", "pvr", "inox", "cinema", "movie", "game", "gaming",
        "steam", "playstation", "xbox", "bowling", "amusement", "concert",
        "fest", "ticket", "smash",
    ],
    "Health": [
        "pharmacy", "apollo", "medplus", "1mg", "hospital", "clinic", "doctor",
        "medical", "health", "medicine", "opticals", "diagnostic", "lab test",
        "pathology", "dental",
    ],
    "Education": [
        "udemy", "coursera", "course", "college", "university", "xerox",
        "stationery", "book", "education", "tuition", "coaching", "exam",
        "library", "printing", "binding",
    ],
    "Subscriptions": [
        "netflix", "spotify", "youtube premium", "prime", "hotstar", "disney",
        "subscription", "chatgpt", "google one", "icloud", "apple music",
        "jio cinema", "zee5", "recurring",
    ],
    "Travel": [
        "irctc", "train", "flight", "airline", "makemytrip", "goibibo",
        "redbus", "abhibus", "travel", "hotel", "oyo", "airbnb", "trip",
    ],
}


def rule_based_categorize(description: str) -> Tuple[str, float]:
    """Categorize a description using keyword matching rules.

    Returns
    -------
    tuple of (str, float)
        (predicted_category, confidence)
        Confidence is 0.70 for a match, 0.0 for no match.
    """
    desc_lower = description.lower()
    # Clean noise for matching
    desc_clean = re.sub(r"[#\-_/\\:,.]\d*", " ", desc_lower)
    desc_clean = re.sub(r"\b\d+\b", " ", desc_clean)
    desc_clean = re.sub(r"\s+", " ", desc_clean).strip()

    for category, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw in desc_clean:
                return category, 0.70
    return UNCATEGORIZED_LABEL, 0.0


def preprocess_description(text: str) -> str:
    """Normalize transaction description by removing numbers, IDs, and punctuation noise.

    Parameters
    ----------
    text : str
        Raw transaction description string (e.g. 'UPI-ZOMATO-9923', 'SWIGGY ORDER 4821').

    Returns
    -------
    str
        Cleaned, lowercased string containing only semantic tokens.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Strip reference numbers, hashes, and digits (e.g. #4821, 9923, 502)
    text = re.sub(r"#\d+", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\d+", " ", text)

    # Replace hyphens, underscores, slashes, punctuation with spaces
    text = re.sub(r"[_\-/\\#:,.]\s*", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse multi-spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_categorization_pipeline(
    max_iter: int = 1000,
    c_reg: float = 5.0,
    random_state: int = 42,
) -> Pipeline:
    """Construct an NLP + LogisticRegression pipeline with word & char n-grams.

    Parameters
    ----------
    max_iter : int, optional
        Maximum iterations for LogisticRegression optimizer (default 1000).
    c_reg : float, optional
        Inverse regularization strength (default 5.0).
    random_state : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    Pipeline
        Configured scikit-learn Pipeline.
    """
    # Combine Word N-grams (1-2) with Character N-grams (2-5)
    union_features = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                sublinear_tf=True,
                min_df=1,
            ),
        ),
    ])

    pipeline = Pipeline([
        ("features", union_features),
        (
            "classifier",
            LogisticRegression(
                C=c_reg,
                max_iter=max_iter,
                random_state=random_state,
                solver="lbfgs",
            ),
        ),
    ])

    return pipeline


def train_model_from_db(
    model_path: str = "models/categorizer.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train the categorizer from the user's own DB transactions.

    Requires at least 30 categorized expense transactions.

    Returns
    -------
    dict
        Evaluation metrics including accuracy, sample counts, categories, model_path.
    """
    from src.db import get_transactions

    df = get_transactions()
    if df.empty:
        raise ValueError("No transactions in database to train from.")

    # Filter to expenses with known categories
    if "type" in df.columns:
        df_expenses = df[df["type"] == "expense"].copy()
    else:
        df_expenses = df.copy()

    # Drop uncategorized
    df_expenses = df_expenses[
        ~df_expenses["category"].isin(["Uncategorized", UNCATEGORIZED_LABEL, ""])
    ]
    df_expenses = df_expenses.dropna(subset=["description", "category"])

    if len(df_expenses) < 30:
        raise ValueError(
            f"Need at least 30 categorized expenses to train ML model. "
            f"Currently have {len(df_expenses)}."
        )

    X = df_expenses["description"].astype(str).apply(preprocess_description)
    y = df_expenses["category"].astype(str)

    # Need at least 2 samples per class for stratified split
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]

    if len(X) < 10:
        raise ValueError("Not enough data with sufficient class diversity to train.")

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Build and fit pipeline on training split
    pipeline = build_categorization_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    # Evaluate on test split
    y_pred = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    # Retrain on full dataset for deployment
    full_pipeline = build_categorization_pipeline(random_state=random_state)
    full_pipeline.fit(X, y)

    # Persist
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(full_pipeline, model_path)
    _MODEL_CACHE[model_path] = full_pipeline

    return {
        "accuracy": accuracy,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": len(X),
        "categories": sorted(y.unique().tolist()),
        "model_path": model_path,
    }


def train_model(
    data_path: str = "data/transactions.csv",
    model_path: str = "models/categorizer.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train the transaction categorizer on expense records from a CSV and save the model artifact.

    This function is kept as a developer testing tool.

    Parameters
    ----------
    data_path : str, optional
        Path to labeled transaction CSV (default 'data/transactions.csv').
    model_path : str, optional
        Filepath to persist trained pipeline (default 'models/categorizer.joblib').
    test_size : float, optional
        Test split proportion (default 0.2).
    random_state : int, optional
        Seed for split and classifier (default 42).

    Returns
    -------
    dict
        Evaluation results containing accuracy, classification_report, and train/test counts.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found at '{data_path}'")

    df = pd.read_csv(data_path)

    # Filter to expenses only as requested
    if "type" in df.columns:
        df_expenses = df[df["type"] == "expense"].copy()
    else:
        df_expenses = df.copy()

    # Drop missing categories or descriptions
    df_expenses = df_expenses.dropna(subset=["description", "category"])

    # Clean text features using our noise-filtering preprocessor
    X = df_expenses["description"].astype(str).apply(preprocess_description)
    y = df_expenses["category"].astype(str)

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Build and fit pipeline on training split
    pipeline = build_categorization_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    # Evaluate on test split
    y_pred = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, zero_division=0)

    # Retrain on full expense dataset for maximum real-world generalization
    full_pipeline = build_categorization_pipeline(random_state=random_state)
    full_pipeline.fit(X, y)

    # Ensure output directory exists and save model
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(full_pipeline, model_path)

    # Update in-memory cache
    _MODEL_CACHE[model_path] = full_pipeline

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": len(X),
        "categories": sorted(y.unique().tolist()),
        "model_path": model_path,
    }


def is_ml_model_available(model_path: str = "models/categorizer.joblib") -> bool:
    """Check if a trained ML model exists on disk or in cache."""
    return model_path in _MODEL_CACHE or os.path.exists(model_path)


def load_model(model_path: str = "models/categorizer.joblib") -> Any:
    """Load the trained categorizer pipeline from disk with in-memory caching.

    Parameters
    ----------
    model_path : str, optional
        Path to serialized model (default 'models/categorizer.joblib').

    Returns
    -------
    Pipeline
        Loaded scikit-learn Pipeline.
    """
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. Call train_model() first."
        )

    pipeline = joblib.load(model_path)
    _MODEL_CACHE[model_path] = pipeline
    return pipeline


def predict_category(
    description: str,
    model_path: str = "models/categorizer.joblib",
    threshold: float = 0.5,
) -> Tuple[str, float]:
    """Predict category and confidence score for a single transaction description.

    Uses the ML model if available, otherwise falls back to rule-based matching.

    Parameters
    ----------
    description : str
        Merchant or transaction string.
    model_path : str, optional
        Path to serialized model (default 'models/categorizer.joblib').
    threshold : float, optional
        Minimum confidence threshold (default 0.5).

    Returns
    -------
    tuple of (str, float)
        (predicted_category, confidence_score)
    """
    # Try ML model first
    if is_ml_model_available(model_path):
        try:
            pipeline = load_model(model_path)
            cleaned = preprocess_description(description)

            probs = pipeline.predict_proba([cleaned])[0]
            classes = pipeline.classes_

            best_idx = int(np.argmax(probs))
            confidence = float(probs[best_idx])

            if confidence < threshold:
                predicted_category = UNCATEGORIZED_LABEL
            else:
                predicted_category = str(classes[best_idx])

            return predicted_category, round(confidence, 4)
        except Exception:
            pass

    # Fall back to rule-based
    return rule_based_categorize(description)


def categorize_dataframe(
    df: pd.DataFrame,
    description_col: str = "description",
    model_path: str = "models/categorizer.joblib",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Categorize an entire DataFrame of transactions and append prediction columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input transaction DataFrame.
    description_col : str, optional
        Column name holding descriptions (default 'description').
    model_path : str, optional
        Path to model artifact (default 'models/categorizer.joblib').
    threshold : float, optional
        Minimum confidence score (default 0.5).

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with added 'predicted_category' and 'confidence' columns.
    """
    df_out = df.copy()
    if df_out.empty or description_col not in df_out.columns:
        df_out["predicted_category"] = []
        df_out["confidence"] = []
        return df_out

    # Try ML model
    if is_ml_model_available(model_path):
        try:
            pipeline = load_model(model_path)
            cleaned_descriptions = (
                df_out[description_col]
                .fillna("")
                .astype(str)
                .apply(preprocess_description)
                .tolist()
            )

            probs = pipeline.predict_proba(cleaned_descriptions)
            classes = pipeline.classes_

            best_indices = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)

            predicted_categories: List[str] = []
            for idx, conf in zip(best_indices, confidences):
                if conf < threshold:
                    predicted_categories.append(UNCATEGORIZED_LABEL)
                else:
                    predicted_categories.append(str(classes[idx]))

            df_out["predicted_category"] = predicted_categories
            df_out["confidence"] = np.round(confidences, 4)
            return df_out
        except Exception:
            pass

    # Fallback: rule-based for each row
    results = df_out[description_col].fillna("").astype(str).apply(
        lambda d: rule_based_categorize(d)
    )
    df_out["predicted_category"] = [r[0] for r in results]
    df_out["confidence"] = [r[1] for r in results]
    return df_out


if __name__ == "__main__":
    print("=" * 65)
    print("  FinSight AI - Training Transaction Categorizer")
    print("=" * 65)

    results = train_model()

    print(f"\n[+] Total Expense Records: {results['total_samples']}")
    print(f"    - Training Samples: {results['train_samples']}")
    print(f"    - Test Samples:     {results['test_samples']}")
    print(f"\n[+] Out-of-Sample Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"[+] Model saved to: {results['model_path']}")

    # Verification on sample descriptions
    print("\n--- Test Predictions on Arbitrary Unseen Descriptions ---")
    test_cases = [
        "SWIGGY ORDER 4821",
        "UPI-ZOMATO-9923",
        "METRO SMART CARD RECHARGE",
        "UBER TRIP HELP-4921",
        "AMAZON PAY RETAIL 7712",
        "NETFLIX RECURRING SUBSCRIPTION",
    ]

    for desc in test_cases:
        cat, conf = predict_category(desc)
        print(f"  • '{desc}' -> [{cat}] (Confidence: {conf:.2%})")
