# Indian Air Quality Analytics & AQI Classification

**Author:** Arnab Kundu
**Project type:** Academic Data Analytics + Machine Learning  
**Application:** Streamlit single-file dashboard  
**Dataset**[`city_day.csv`]https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
---

## Project Overview

This project performs an end-to-end data analytics and machine learning study on India's ambient air quality using daily city-level measurements from 2015 to 2020. The goal is to classify each day's air quality into one of six AQI (Air Quality Index) categories — *Good*, *Satisfactory*, *Moderate*, *Poor*, *Very Poor*, or *Severe* — using pollutant concentration measurements from ground-level monitoring stations.

The project deliberately avoids using the numeric AQI score as a predictor (which would constitute target leakage) and instead learns the classification directly from raw pollutant measurements.

---

## Problem Statement

India's urban air quality poses significant public health risks. While the numeric AQI is useful, a category classification model trained on raw pollutant readings can:
- provide real-time air quality categorisation without needing a pre-computed AQI score
- identify which pollutants are most predictive of poor air quality
- generalise to new monitoring stations not seen during training

---

## Objectives

1. Inspect, profile, and document the quality of the air quality dataset
2. Perform exploratory data analysis (EDA) including missing-value analysis, outlier detection, and temporal trend analysis
3. Design a leakage-free, time-aware preprocessing pipeline
4. Train and compare multiple classification algorithms
5. Evaluate models using class-imbalance-aware metrics (macro-F1, balanced accuracy)
6. Deploy a polished academic Streamlit dashboard

---

## Dataset Description

| Property | Value |
|---|---|
| File | `city_day.csv` |
| Raw rows | 29,531 |
| Columns | 16 |
| Cities | 26 (India) |
| Date range | 2015-01-01 to 2020-07-01 |
| Granularity | One row per city per calendar day |

**Dataset source:** Kaggle — India Air Quality dataset (CPCB monitoring data).  
The dataset was used in read-only mode. Original files were never modified.

### Additional files (read-only, not used in primary modelling)
| File | Description |
|---|---|
| `city_hour.csv` | Hourly city-level readings |
| `stations.csv` | Station metadata (city, state, status) |
| `station_day.csv` | Daily station-level readings |
| `station_hour.csv` | Hourly station-level readings |

---

## Features Used

The model uses **11 pollutant concentration features**:

| Feature | Unit | Description |
|---|---|---|
| PM2.5 | µg/m³ | Fine particulate matter (≤ 2.5 µm) |
| PM10 | µg/m³ | Coarse particulate matter (≤ 10 µm) |
| NO | µg/m³ | Nitric oxide |
| NO2 | µg/m³ | Nitrogen dioxide |
| NOx | µg/m³ | Total nitrogen oxides |
| NH3 | µg/m³ | Ammonia |
| CO | mg/m³ | Carbon monoxide |
| SO2 | µg/m³ | Sulphur dioxide |
| O3 | µg/m³ | Ozone |
| Benzene | µg/m³ | Benzene (VOC) |
| Toluene | µg/m³ | Toluene (VOC) |

### Excluded features

| Feature | Reason |
|---|---|
| `AQI` | **Target leakage** — AQI_Bucket is deterministically derived from AQI |
| `Xylene` | **61.3% global missing**; 100% absent in 16/26 cities |

---

## Target Variable

`AQI_Bucket` — 6 ordinal categories following the CPCB (India) scale:

| Category | AQI range | Label encoding |
|---|---|---|
| Good | 0–50 | 0 |
| Satisfactory | 51–100 | 1 |
| Moderate | 101–200 | 2 |
| Poor | 201–300 | 3 |
| Very Poor | 301–400 | 4 |
| Severe | 401+ | 5 |

---

## Data Preprocessing

1. **Date parsing** — `Date` column parsed to `datetime64`
2. **Chronological sort** by `[Date, City]`
3. **Drop missing-target rows** — 4,681 rows with null `AQI_Bucket` dropped (target imputation is prohibited)
4. **Time-aware train/test split** at `2019-05-27`
5. **City-specific median imputation** — computed from training rows only; global training median used as fallback for cold-start cities
6. **log1p transformation** — applied to all 11 features (reduces right skew; Benzene skew drops from 27.6 to 0.9)
7. **Label encoding** — Good=0, Satisfactory=1, ..., Severe=5

---

## Handling of Missing Data

| Column | Missing % | Treatment |
|---|---|---|
| Xylene | 61.3% | Excluded from model |
| PM10 | 37.7% | City-specific median imputation |
| NH3 | 35.0% | City-specific median imputation |
| Toluene | 27.2% | City-specific median imputation |
| Benzene | 19.0% | City-specific median imputation |
| PM2.5 | 15.6% | City-specific median imputation |
| AQI_Bucket | 15.9% | Rows dropped |

Imputation statistics are computed **from the training set only** and stored in `artifacts/imputation_stats.json`. The same statistics are used at inference time to prevent data leakage.

---

## Model Used

**XGBoost** (extreme gradient boosting, 6-class multiclass)

Selected from a comparison of 6 models on the held-out chronological test set.

Configuration:
- `n_estimators=300`, `learning_rate=0.1`, `max_depth=6`
- `subsample=0.8`, `colsample_bytree=0.8`
- `sample_weight` (balanced class weights)
- Feature transformation: `log1p` applied before training

---

## Evaluation Methodology

- **Primary metric: Macro-F1** — treats all 6 classes equally regardless of class size
- **Secondary metrics:** Balanced Accuracy, Weighted-F1, Accuracy
- **Split method:** Chronological (time-aware) — no random shuffling
- **Class imbalance:** Addressed via `sample_weight='balanced'` (14.5× imbalance ratio)
- **SMOTE / oversampling:** Intentionally deferred to preserve temporal integrity

---

## Results

| Model | Macro-F1 (test) | Balanced Acc | Accuracy |
|---|---|---|---|
| DummyClassifier (baseline) | 0.154 | 0.165 | 0.256 |
| Logistic Regression | 0.612 | 0.680 | 0.607 |
| Random Forest | 0.751 | 0.745 | 0.779 |
| Gradient Boosting | 0.739 | 0.730 | 0.781 |
| **XGBoost ⭐** | **0.760** | **0.757** | **0.789** |
| LightGBM | 0.744 | 0.732 | 0.782 |

**XGBoost per-class performance (test set):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Good | 0.73 | 0.63 | 0.68 |
| Satisfactory | 0.81 | 0.86 | 0.83 |
| Moderate | 0.82 | 0.77 | 0.80 |
| Poor | 0.63 | 0.72 | 0.67 |
| Very Poor | 0.77 | 0.70 | 0.74 |
| Severe | 0.83 | 0.86 | 0.85 |

**Top feature importances (permutation):** PM2.5 (0.248) > CO (0.173) > PM10 (0.154) > O3 (0.047)

---

## Streamlit Application Features

The application (`ArnabKundu_IndianAirQualityAnalytics.py`) includes 7 pages:

| Page | Description |
|---|---|
| 🏠 Home | Project overview, objectives, key metrics |
| 🔮 AQI Prediction | Interactive prediction from pollutant inputs |
| 📊 Data Explorer | Charts from the original dataset (read-only) |
| 🤖 Model Information | Full evaluation metrics, confusion matrix, preprocessing |
| 📈 Feature Importance | Built-in and permutation importance charts |
| ⚠️ Data Quality | Missing data, outliers, Ahmedabad analysis, known issues |
| 📖 Methodology | Full project workflow and academic rationale |

---

## Project Structure

```
.
├── ArnabKundu_IndianAirQualityAnalytics.py   # Single-file Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── ArnabKundu_ProjectReport.docx    # Academic project report
├── phase3_preprocessing.py         # Reproducible preprocessing script
├── phase4_modeling.py              # Full training + evaluation script
├── phase3_imputation_stats.json    # Imputation backup
├── artifacts/
│   ├── best_model.pkl              # Trained XGBoost model (5.8 MB)
│   ├── imputation_stats.json       # City-specific median imputation stats
│   ├── model_metadata.json         # Metrics, labels, feature lists, importance
│   └── robust_scaler.pkl           # RobustScaler (for LR pipeline)
├── city_day.csv                    # Primary dataset (READ-ONLY)
├── city_hour.csv                   # Not used in modelling (READ-ONLY)
├── stations.csv                    # Station metadata (READ-ONLY)
├── station_day.csv                 # Not used in modelling (READ-ONLY)
└── station_hour.csv                # Not used in modelling (READ-ONLY)
```

---

## Installation Instructions

```bash
# 1. Clone or download the project folder
# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run the Streamlit App

```bash
streamlit run ArnabKundu_IndianAirQualityAnalytics.py
```

The app will open at `http://localhost:8501` in your browser.

> **Note:** The `artifacts/` directory must be present and contain the four artifact files. If they are missing, run `python phase4_modeling.py` first to regenerate them (requires the original CSV files to be present).

---

## Example Usage

1. Open the application: `streamlit run ArnabKundu_IndianAirQualityAnalytics.py`
2. Navigate to **🔮 AQI Prediction** in the sidebar
3. (Optional) Select a city for imputation defaults
4. Adjust the pollutant sliders to your desired values
5. Click **Predict AQI Category**
6. The predicted AQI_Bucket and class probabilities are displayed immediately

---

## Limitations

1. **AQI_Bucket predictions are advisory only** — not medical or regulatory guidance
2. The model was trained on 2015–2019 data; performance may degrade for conditions far outside that range
3. **Xylene is excluded** — the model has no Xylene signal
4. **Ahmedabad's unique CO-driven AQI profile** is not well-generalised; per-city macro-F1 for Ahmedabad is 0.558
5. All pollutant inputs must be non-negative and within physically plausible ranges
6. The model does not account for weather variables (wind speed, temperature, humidity, rainfall) which strongly affect pollutant dispersion

---

## Future Improvements

1. Add temporal features (Month, Season, Year) to capture seasonal patterns (mean Jan AQI ≈ 232, Jul AQI ≈ 112)
2. Include weather variables as additional predictors
3. Extend to station-level data (`station_day.csv`) for finer spatial resolution
4. Implement SMOTE or time-aware oversampling for the Good and Severe minority classes
5. Add City as a high-cardinality feature (target encoding) to capture city-level fixed effects
6. Retrain Benzene from features (near-zero permutation importance = −0.002)
7. Cross-validate using `TimeSeriesSplit` for more robust generalisation estimates

---

## Academic Disclaimer

This project was developed as an academic submission. All data analytics results, model metrics, confusion matrices, and feature importance values are derived directly from the original Kaggle dataset. No results have been fabricated or post-hoc adjusted. The model is for educational and demonstrative purposes only.

---

*Dataset: India Air Quality Data, Central Pollution Control Board (CPCB) via Kaggle*
