# Corporate Bankruptcy Prediction System

An end-to-end, containerized Machine Learning service designed to identify financially distressed companies at high risk of bankruptcy. Built with a focus on **minimizing False Negatives (Type-2 errors)** on severely imbalanced financial data, the system pairs a leak-free training pipeline with a production-ready **FastAPI** and **Docker** inference service.

---

## Business Objective & Problem Framing
In corporate credit and investment risk assessment, failing to flag a bankrupt company (False Negative) carries a significantly higher financial cost than flagging a healthy company for manual review (False Positive). 

* **Primary Evaluation Metric:** **Recall** (Sensitivity) to maximize the detection of actual bankruptcy cases.
* **Core Challenge:** Severe class imbalance (~3.7% positive bankruptcy class) and extreme multicollinearity across 95 financial ratios and indicators.

---

## Exploratory Data Analysis & Methodology (`notebook/Bankruptcy.ipynb`)
Before building the modular production scripts, thorough experimentation was conducted in the Jupyter notebook:

1. **Class Imbalance & Distribution Profiling:**
   * Quantified severe class imbalance between non-bankrupt (`0`) and bankrupt (`1`) companies.
   * Analyzed skewness (`|skew| > 2`), value ranges, and `max-to-Q3` ratios across all 95 features to identify extreme financial outliers.
   * Conducted bivariate mean-difference and target-correlation profiling using boxen plots to isolate top predictive financial indicators.
2. **Multicollinearity Reduction (2 Approaches Tested):**
   * *Correlation Filtering (Notebook Baseline):* Removed redundant features with pairwise correlation `> 0.85`, reducing the feature space from 95 to 45 (`0.8636` recall).
   * *Iterative Variance Inflation Factor (Production Pipeline):* Systematically dropped features with `VIF > 10` directly on the training set (eliminating 24 collinear ratios such as *Operating Gross Margin* with `VIF = 7.28e7`), retaining **71 core features** and boosting test Recall to **0.8824**.
3. **Leak-Free Pipeline & Benchmarking:**
   * Performed stratified train/test splitting (`80/20`) prior to VIF calculation, scaling, and resampling.
   * Wrapped `StandardScaler` and `SMOTE` inside an `imblearn` pipeline during 5-fold cross-validation (`GridSearchCV`, optimal `n_neighbors = 19`) to guarantee zero data leakage.

### Model Evaluation & Benchmark Summary

| Model | Feature Selection | Recall (Primary) | Accuracy | False Negatives | Key Takeaway |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **KNN Tuned (`k=19`)** | **VIF (71 features)** | **0.8824** | **0.8160** | **6 / 51** | **Selected for production: highest recall (45/51 bankruptcies caught) and principled multicollinearity removal.** |
| KNN Tuned | Correlation (45 features) | 0.8636 | — | 6 | Strong baseline, but relied on manual pairwise correlation thresholds. |
| Logistic Regression | VIF (71 features) | ~0.7950 | — | — | Solid linear baseline, lower sensitivity than distance-based KNN. |
| Tree / Ensemble Models | Correlation (45 features) | < 0.7900 | — | — | Underperformed KNN after SMOTE synthetic oversampling. |

**Production Test Confusion Matrix (`n = 1,364`):**
* **True Positives (Bankruptologies Caught):** `45`
* **False Negatives (Missed Bankruptcies):** `6`
* **True Negatives (Healthy Companies Cleared):** `1,068`
* **False Positives (Flagged for Review):** `245`

---

## Project Architecture

```text
Bankruptcy_Project/
├── app/
│   └── api.py                  # FastAPI application for health checks & batch predictions
├── datasets/
│   └── Dataset.csv             # Raw financial dataset
├── inputs/
│   └── input.json              # Sample JSON payload for API testing
├── models/
│   ├── bankruptcy_model.pkl    # Serialized imblearn pipeline (Scaler + SMOTE + KNN k=19)
│   └── vif_columns.txt         # 71 selected VIF features enforced at inference
├── notebook/
│   └── Bankruptcy.ipynb        # Full EDA, statistical profiling & model benchmarking
├── scripts/
│   ├── extract_features.py     # Iterative VIF multicollinearity reducer
│   ├── model_train.py          # GridSearchCV + SMOTE training & evaluation logic
│   └── main.py                 # End-to-end training & artifact serialization orchestrator
├── .dockerignore
├── Dockerfile
└── requirements.txt
```

---

## Quickstart & Deployment

### 1. Train the Model Pipeline
Run the training orchestrator to execute VIF feature selection, tune the KNN pipeline, and serialize `bankruptcy_model.pkl` and `vif_columns.txt`:
```bash
python scripts/main.py
```

### 2. Run with Docker (Recommended)
Build and start the containerized FastAPI service:
```bash
docker build -t bankruptcy-api .
docker run -p 8000:8000 bankruptcy-api
```

### 3. Run Locally
```bash
pip install -r requirements.txt
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Usage

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### Health Check
```bash
curl -X GET http://localhost:8000/
```
**Response:**
```json
{
  "status": "online",
  "expected_features_count": 71
}
```

### Batch Risk Prediction (`POST /predict-file`)
Upload a `.csv` or `.json` file containing company financial metrics. The API automatically validates that all 71 required VIF columns are present before running inference:
```bash
curl -X POST "http://localhost:8000/predict-file" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@inputs/input.json"
```
**Sample Response:**
```json
{
  "filename": "input.json",
  "rows_processed": 1,
  "results": [
    {
      "row": 1,
      "bankrupt_prediction": 1,
      "risk_label": "High Risk (Bankrupt)",
      "bankruptcy_probability": 0.8421
    }
  ]
}
```
