# FinSight AI: AI-Powered Personal Finance Management System

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finsight-ai-ytwfcgtro8vcm4uwwkmmx9.streamlit.app/)

> **Live Demo:** 🚀 **[https://finsight-ai-ytwfcgtro8vcm4uwwkmmx9.streamlit.app/](https://finsight-ai-ytwfcgtro8vcm4uwwkmmx9.streamlit.app/)**  
> **GitHub:** [sainathmpattar-maker/finsight-ai](https://github.com/sainathmpattar-maker/finsight-ai)  
> **Course Project:** AI for Finance  
> **Target Persona:** Indian College Students & Early-Career Professionals  
> **Status:** Production-Ready & Evaluated  

**FinSight AI** is a privacy-first, zero-synthetic personal finance studio that transforms your genuine bank, UPI, and credit card statements into actionable financial intelligence. The app **starts completely empty** with no fake, synthetic, or sample records. All transactions are persisted in a local **SQLite database** (`data/finsight.db`), and advanced AI/ML capabilities progressively unlock as your real transaction history grows.

---

## Architecture Diagram

```text
+-------------------------------------------------------------------------------------------------+
|                                    USER FINANCIAL INGESTION                                     |
|  - Manual Entry Form (Add Transaction page with Quick Add shortcuts: Chai, Auto, Swiggy, etc.)  |
|  - CSV Statement Import (Flexible date parser: YYYY-MM-DD / DD-MM-YYYY, duplicate skipping)     |
+-------------------------------------------------------------------------------------------------+
                                                |
                                                v
+-------------------------------------------------------------------------------------------------+
|                                 PERSISTENT STORAGE & CRUD LAYER                                 |
|  - SQLite Database: data/finsight.db via src/db.py                                              |
|  - Transactions schema: id, date, description, amount, type, category, payment_mode, created_at |
+-------------------------------------------------------------------------------------------------+
                                                |
             +----------------------------------+----------------------------------+
             |                                  |                                  |
             v                                  v                                  v
+--------------------------+       +--------------------------+       +--------------------------+
|   SMART CATEGORIZATION   |       |    ANOMALY DETECTION     |       |    SPENDING FORECAST     |
|   (src/categorizer.py)   |       |     (src/anomaly.py)     |       |    (src/forecast.py)     |
| - Phase 1: Keyword rule  |       | - 24-Hour Duplicate      |       | - Unlocks at 2+ months   |
|   matcher (0-29 txns)    |       |   Charge Guard (from #1) |       | - Linear Regression on   |
| - Phase 2: TF-IDF N-gram |       | - Isolation Forest       |       |   month index (t=0..N-1) |
|   Logistic Regression    |       |   Outlier Radar (20+     |       | - 3-Month Moving Average |
|   (auto-trains at 30+)   |       |   expense transactions)  |       | - Blended Ensemble       |
| - Continuous learning    |       | - 3-Sigma Outlier Rule   |       | - Projected Savings Rate |
|   from user corrections  |       | - Explainable Reasons    |       | - "What-If" Simulation   |
+--------------------------+       +--------------------------+       +--------------------------+
             |                                  |                                  |
             +----------------------------------+----------------------------------+
                                                |
                                                v
+-------------------------------------------------------------------------------------------------+
|                             FINANCIAL CONTEXT BUILDER & ADVISOR                                 |
|                                     (src/advisor.py)                                            |
|  - Unlocks with at least 1 income and 1 expense entry (grounded strictly on user's real numbers)|
|  - Primary Engine: Google Gemini 2.5 Flash via `google-genai` SDK (Strict JSON Payload)         |
|  - Resilient Fallback: FinSight Rule Engine (50/30/20 checks, category audits, dispute alerts)   |
+-------------------------------------------------------------------------------------------------+
                                                |
                                                v
+-------------------------------------------------------------------------------------------------+
|                                STREAMLIT MULTIPAGE WEB INTERFACE                                |
|                                           (app.py)                                              |
|  1. Add Transaction: Quick Add buttons, validated form, category auto-suggest, inline table edit |
|  2. Executive Dashboard: Plotly KPI Cards, Monthly Cash Flows, Donut Share, Daily Spend Trend  |
|  3. Smart Categorization: Rule matcher -> ML model, interactive table editor & model retrainer |
|  4. Anomaly Alerts: 24h Duplicate detector & Isolation Forest radar with explainable cards     |
|  5. Forecast & What-If: Actual vs Predicted Bar Chart, Category Cut Simulator, Net Savings      |
|  6. AI Financial Coach: Health Gauge (0-100), Action Cards with INR Savings, Multi-Turn Chat   |
|  7. About & Architecture: Course Documentation, Model Performance, and Manifest                |
+-------------------------------------------------------------------------------------------------+
```

---

## Key Features & Progressive Unlocks

1. **Add Transaction Page (New First Page)**
   - Clean validated entry form with Date, Type (Income/Expense), Amount, Description, Category, and Payment Mode.
   - For expenses, category is **auto-suggested** using the categorizer as you enter the description.
   - **Quick Add Shortcuts**: 1-click buttons for frequent items (*Chai ₹40*, *Auto ₹80*, *Swiggy ₹280*, *Recharge ₹299*).
   - **Recent Transactions Manager**: Interactive inline editor (`st.data_editor`) and single-click delete.

2. **Executive Financial Dashboard (Unlocks from 1 Transaction)**
   - High-impact metric cards: Total Inflow, Total Outflow, Net Savings, and Top Category.
   - Zero-divide safe: Shows clean ₹0 states when no transactions are recorded.
   - Grouped Plotly bar chart comparing monthly income against expenditures with INR tooltips.
   - Interactive category donut chart with hole=0.65 and center total outlay.
   - Daily spending trend area chart with an integrated 7-day rolling moving average.
   - Top 10 spending merchants ranked horizontal bar chart.

3. **Smart Categorization (Progressive AI)**
   - **Phase 1 (Cold Start, 0–29 expenses)**: Rule-based keyword dictionary classifies merchants (Swiggy, Zomato, Uber, Metro, etc.).
   - **Phase 2 (30+ expenses)**: Automatically trains an NLP pipeline (`TfidfVectorizer` word & character n-grams + `LogisticRegression`) on your personal data.
   - **Continuous Learning**: Every time you correct a category in the table editor, it persists to SQLite and can be used to retrain the model.

4. **Spending Anomaly & Fraud Detection**
   - **24-Hour Duplicate Charge Detector**: Flags identical amounts and merchants within 1 day (active from transaction #1).
   - **Isolation Forest Radar (Unlocks at 20+ expenses)**: Unsupervised multi-feature outlier scoring combined with 3-sigma category thresholds.
   - Plain-language explainable root causes for flagged transactions.

5. **Predictive Spending Forecast & "What-If" Simulator (Unlocks at 2+ Months)**
   - Blended ensemble model: Linear Regression on historical monthly sequence averaged with a 3-month moving average:
     $$\hat{y} = \frac{\max(0, \hat{y}_{\text{LR}}) + \hat{y}_{\text{MA}}}{2}$$
   - Calculates **Projected Monthly Savings** and **Projected Savings Rate %**.
   - Interactive slider allows simulating a $0\%$ to $60\%$ cut in any selected category, recalculating adjusted savings.

6. **AI Financial Coach (Unlocks with $\ge 1$ Income and $\ge 1$ Expense)**
   - Computes a holistic **Financial Health Score (0–100)** displayed via an interactive gauge.
   - Prioritized action cards with **HIGH / MEDIUM / LOW** badges, clear actions, and estimated monthly savings in INR.
   - Conversational chat assistant (`st.chat_input`) grounded on your real financial numbers.
   - **Zero-Downtime Guarantee**: Seamlessly falls back to a deterministic 50/30/20 rule engine when offline.

7. **CSV Import & Export Studio**
   - Upload bank/card/UPI CSV files with columns `date`, `description`, `amount`, and optional `type`, `category`, `payment_mode`.
   - Flexible date parsing supporting `YYYY-MM-DD` and `DD-MM-YYYY`.
   - Error detection preview (counts bad dates and bad amounts).
   - Exact duplicate skipping on import.
   - Downloadable blank CSV template and 1-click **Export My Data as CSV** button.

---

## Technology Stack

| Layer | Library / Tool | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core runtime environment |
| **Database** | SQLite3 (`data/finsight.db`) | Persistent local transactional storage |
| **Web Framework** | Streamlit ($\ge 1.35$) | Reactive multipage web interface |
| **Visualization** | Plotly Express & Graph Objects | Interactive charts with INR formatting |
| **Data Processing** | Pandas, NumPy | Schema cleaning, aggregation, resample |
| **Machine Learning** | Scikit-learn, Scipy | TF-IDF, Logistic Regression, Isolation Forest, Linear Regression |
| **Model Persistence**| Joblib | Fast binary pipeline serialization |
| **Generative AI** | Google GenAI SDK (`google-genai`) | Gemini 2.5 Flash API client |
| **Environment** | Python-Dotenv | Secure environment key management |

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher installed.
- *(Optional)* Gemini API Key from [Google AI Studio](https://aistudio.google.com/).

### 2. Clone and Navigate
```bash
git clone https://github.com/sainathmpattar-maker/finsight-ai.git
cd finsight-ai
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Secrets (.env)
Duplicate `.env.example` to create `.env`:
```bash
cp .env.example .env
```
*(On Windows: `Copy-Item .env.example .env`)*

Open `.env` and insert your Gemini API Key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```
> **Note:** If you leave `GEMINI_API_KEY` blank or omitted, FinSight AI automatically runs in **Rule-Based Fallback Mode** with 100% functionality.

---

## Running the Application

Launch the Streamlit web dashboard:
```bash
python -m streamlit run app.py
```

The application will launch in your browser at:
```text
http://localhost:8501
```

The app starts completely empty. Add transactions on the **Add Transaction** page, or import your CSV statement to begin!
