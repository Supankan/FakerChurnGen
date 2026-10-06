# FakerChurnGen

> **Realistic Synthetic Customer Churn Dataset Generator with Domain Invariants, Causal Relationships, and ML Benchmarking.**

FakerChurnGen is an end-to-end synthetic data synthesis engine designed to generate high-fidelity, ML-ready customer churn datasets. Unlike naive random generators, FakerChurnGen enforces **hierarchical service gating**, **real-world physical bounds**, and a **rank-preserving correlation engine (Iman-Conover)** to produce datasets that obey domain invariants while reflecting true causal dynamics.

---

## Key Features

- **9 Production Business Presets**:
  - **Telecom**: Contract terms, internet & voice add-on hierarchies, usage and tenure charges.
  - **SaaS**: ARR/MRR tiers, seat licensing, product activity metrics, support ticket SLAs.
  - **Banking**: Account tenure, credit limits, multi-product balances, transaction flags.
  - **E-Commerce**: Order history, order chronology, basket economics, return constraints.
  - **Gaming**: Matchmaking latency, battle pass monetization, play streaks, session lengths.
  - **Streaming**: Content consumption hours, device counts, offline downloads, subscription tiers.
  - **Ride-Hailing**: Ride frequency, completion rates, surge elasticity, lifetime ride bounds.
  - **Fitness**: Facility visit recency, membership tiers, synchronized attendance drop ratios.
  - **EdTech**: Course enrollments vs. completions, streak tracking, certificate verification.

- **Iman-Conover Rank-Preserving Correlation Engine**:
  - Employs distribution-free rank permutation: preserves **100.0% of unique continuous values** and marginal distribution shapes without resampling collapse.
  - **Joint Connected-Component Solver**: Resolves overlapping multi-column correlation graphs simultaneously using Positive Semi-Definite (PSD) eigenvalue projection and Cholesky factorization.

- **Strict Domain Invariant Enforcement**:
  - Hierarchical service gates (e.g., no internet add-ons without internet service; no multi-line features without phone service).
  - Arithmetic and chronological consistency (e.g., total spend bounded by order count × average order value; lifetime rides $\ge$ last 30d rides; courses completed $\le$ courses enrolled).

- **Powerful Causal Rule Engine**:
  - Composite multi-condition triggers with typed comparison operators.
  - Non-linear transforms (Bathtub hazard curves, Inactivity decay, Step functions).
  - Telemetry badges displaying active rule status, row matches, and target vs. realized Pearson & Spearman correlations.

- **In-Browser Baseline ML Diagnostics**:
  - Instant training with Scikit-Learn (Histogram Gradient Boosting & Logistic Regression).
  - Evaluates ROC-AUC, classification metrics, and permutation feature importances directly against the simulated causal factors.

- **Craftsman Dark UI**:
  - Human-centered dark aesthetic with warm slate, amber, and sage accents.
  - Searchable categorized column selector, row count presets (1k to 500k), and selectable preview samples (5 to 200 rows).

---

## Getting Started

### Prerequisites

- Python 3.10+
- `pip`

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Supankan/FakerChurnGen.git
   cd FakerChurnGen
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## Running the Test Suites

The test suite validates schema boundaries, reflection security, typed rule evaluation, Iman-Conover rank preservation, and invariant compliance:

```bash
# Run Phase 1 & 2 tests (security boundary enforcement & typed rule matching)
python scratch/test_phase1_phase2.py

# Run Phase 3 & 4 tests (domain invariants & rank-preserving correlation engine)
python scratch/test_phase3_phase4.py

# Run all 9 presets integration test
python scratch/test_all_presets.py
```

---

## Project Architecture

```
FakerChurnGen/
├── app.py                      # Flask REST API & Web Server
├── requirements.txt            # Python dependencies
├── generator/
│   ├── engine.py               # Vectorized dataset generation pipeline
│   ├── domain_invariants.py    # Hierarchical gating & physical invariant rules
│   ├── relationships.py        # Iman-Conover rank permutation & joint PSD correlation solver
│   ├── churn_logic.py          # Calibrated logit scoring & churn assignment
│   ├── ml_evaluator.py         # HistGB / Logistic Regression baseline evaluator
│   ├── presets.py              # 9 comprehensive industry schema presets
│   └── schema.py               # Column definitions & datatype validation
├── static/
│   ├── css/style.css           # Craftsman dark theme styles
│   └── js/app.js               # Reactive frontend controller & telemetry renderer
├── templates/
│   └── index.html              # Interactive web dashboard
└── scratch/                    # Verification & automated test suites
```

---

## License

MIT License. Feel free to use, modify, and distribute for ML research, synthetic data benchmarking, or demonstration purposes.
