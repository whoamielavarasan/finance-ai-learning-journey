# Personal Finance AI — 7-Day Learning Journey

A documented, day-by-day learning project building a Personal Finance AI application
from scratch, transitioning from 10+ years of C# and SQL experience into Python and
Machine Learning.

---

## Background

**Starting point:** Expert in C#, SQL, .NET — zero Python, zero ML experience.

**Goal:** Build, understand, and deploy a real Personal Finance AI application within
7 days, with complete understanding of every line of code produced.

**Approach:** Every concept mapped to C#/SQL equivalents. Real bank data used instead
of synthetic data (3 accounts, 18 months, 712 transactions). Every bug found and fixed
by understanding the code, not just running it.

---

## The 7-Day Roadmap

| Day | Focus | Key Deliverable |
|-----|-------|-----------------|
| 1 | Python for C#/SQL Developers | Syntax, virtual environments, Pandas DataFrames |
| 2 | Data Ingestion & Cleaning | CSV parsing, null handling, deduplication pipeline |
| 3 | Analytics Engine | Categorisation, efficiency scoring, spending patterns |
| 4 | Feature Engineering | Monthly aggregation, lag features, rolling averages |
| 5 | ML Models | Linear Regression vs Random Forest, MAE/RMSE evaluation |
| 6 | Insights Engine | Z-score anomaly detection, Anthropic API integration |
| 7 | CLI Deployment | argparse interface, joblib model persistence |

---

## Files

```
finance-ai-learning-journey/
├── day1_finance.py          # Python fundamentals — DataFrames as SQL tables
├── day2_finance.py          # Ingestion pipeline — CSV → clean typed DataFrame
├── day3_finance.py          # Analytics engine — categories, scores, reports
├── day4_finance.py          # Feature engineering — monthly ML feature matrix
├── day5_finance.py          # ML training — Linear Regression + Random Forest
├── day6_finance.py          # Insights — anomaly detection + AI text generation
├── finance_app.py           # Production CLI — argparse + model persistence
├── categorize_merchants.py  # Merchant classification rules (40+ regex patterns)
└── requirements.txt         # Python dependencies
```

> **Note:** Real financial data files (`transactions.csv`, bank statements, parsed
> CSVs) are git-ignored and never committed. The production-ready version of this
> project with mock data is available at
> [personal-finance-ai](https://github.com/whoamielavarasan/personal-finance-ai).

---

## Key Learnings

### Python vs C# — The Mental Model Shifts

| Concept | C# | Python/Pandas |
|---|---|---|
| Typed collection | `List<Transaction>` | `pd.DataFrame` |
| SQL GROUP BY | LINQ `.GroupBy()` | `.groupby().agg()` |
| SQL WHERE | LINQ `.Where()` | Boolean masking `df[df["col"] > x]` |
| SQL OVER (PARTITION BY) | Manual loop | `.transform()` |
| SQL LAG() | Manual index offset | `.shift(1)` |
| try/catch | `try { } catch (Exception ex)` | `try: except Exception as e:` |
| null | `null` | `None`, `NaN`, `NaT` |
| Dictionary<K,V> | `Dictionary<string, object>` | `dict` |

### Real Bugs Caught By Understanding the Code

Because every line was understood rather than just copy-pasted, 9 consequential
bugs were identified and fixed during development:

1. **SoFi loan disbursement ($28,200)** miscategorised as an expense
2. **Comcast refund ($13.13)** counted as a utility expense
3. **BofA credit card December rows** duplicated across two statement files
4. **Citi duplicate $2.19 Costco purchase** wrongly deleted by deduplication
5. **`spend_type` column overwritten** by `flag_transactions()` on real data
6. **Grouping by month number alone** merged Jan 2025 with Jan 2026
7. **`is_month_end` check** incorrectly flagged 13 of 18 complete months as partial
8. **Removing `num_transactions`** degraded Linear Regression performance (Run 3)
9. **Americas Best $323.97** miscategorised as Shopping instead of Medical

### ML Model Results (Real Data — 18 Months)

```
Dataset:        712 transactions, 3 bank accounts, Jan 2025 – Jun 2026
Monthly rows:   17 usable (after partial month detection)
Train/Test:     12 train / 3 test (chronological split, shuffle=False)

Random Forest:
  MAE:  $2,202  (~27% error on typical ~$8,300/month spend)
  RMSE: $3,381  (inflated by outlier month prediction failure)

Linear Regression:
  MAE:  $3,161
  RMSE: $3,343
  Issue: multicollinearity between necessary_spend and total_spend

Winner: Random Forest — immune to multicollinearity, better on normal months.
Limitation: both models fail on outlier months (large irregular cash withdrawals).
Root cause: 12 training rows is too few for reliable outlier prediction.
```

### Multicollinearity Discovery

Running a correlation matrix revealed:
- `total_spend` vs `necessary_spend`: ~0.99 correlation (96% of spend is Necessary)
- `num_transactions` vs `avoidable_spend`: 0.92 correlation

Three-run experiment to fix it:
- **Run 1** (11 features): LR MAE $3,509 — severe multicollinearity
- **Run 2** (removed total_spend + lag_total_spend): LR MAE $3,161 — improved
- **Run 3** (also removed num_transactions): LR MAE $5,191 — degraded

Conclusion: with only 15 training rows, removing correlated features can hurt more
than help. Empirical testing beats theoretical rules on small datasets.

---

## Tech Stack

- **Python 3.11** with type hints throughout
- **Pandas 3.x** — DataFrames, groupby, transform, rolling, shift
- **Scikit-Learn** — RandomForestRegressor, LinearRegression, train_test_split
- **joblib** — model serialisation
- **Anthropic Python SDK** — AI financial insights
- **python-dotenv** — secure API key management
- **argparse** — CLI interface
- **pytest** — unit testing (18 tests)

---

## Production Version

The refactored, production-quality version of this project with:
- Layered architecture (infrastructure / data / features / analytics / models / insights)
- Dual console/file logging
- 18 unit tests
- Privacy-safe mock dataset
- Full GitHub documentation

→ [personal-finance-ai](https://github.com/whoamielavarasan/personal-finance-ai)

---

## Author

**Elavarasan Dhayalan**
10+ years C# and SQL | Python and ML — learned in 7 days
