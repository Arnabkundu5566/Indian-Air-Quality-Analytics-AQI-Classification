"""
Indian Air Quality Analytics & AQI Classification
Single-file Streamlit application — ArnabKundu_IndianAirQualityAnalytics.py

Author : Arnab Kundu
Dataset: city_day.csv (India Air Quality, 2015–2020)
Model  : XGBoost (trained in Phase 4, loaded from artifacts/)

Run    : streamlit run ArnabKundu_IndianAirQualityAnalytics.py
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

# Ensure user site-packages are on path (for XGBoost / LightGBM)
_user_site = os.path.join(os.path.expanduser("~"), "AppData", "Roaming",
                          "Python", "Python314", "site-packages")
if os.path.isdir(_user_site) and _user_site not in sys.path:
    sys.path.insert(0, _user_site)

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — derived from artifacts/model_metadata.json at load time
# ─────────────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = "artifacts"
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
IMP_PATH      = os.path.join(ARTIFACTS_DIR, "imputation_stats.json")
META_PATH     = os.path.join(ARTIFACTS_DIR, "model_metadata.json")
SCALER_PATH   = os.path.join(ARTIFACTS_DIR, "robust_scaler.pkl")
DATA_PATH     = "city_day.csv"

BUCKET_COLORS = {
    "Good":         "#16a34a",
    "Satisfactory": "#3b82f6",
    "Moderate":     "#f59e0b",
    "Poor":         "#f97316",
    "Very Poor":    "#dc2626",
    "Severe":       "#7c3aed",
}
BUCKET_EMOJI = {
    "Good": "🟢", "Satisfactory": "🔵", "Moderate": "🟡",
    "Poor": "🟠", "Very Poor": "🔴", "Severe": "🟣",
}
BUCKET_ADVICE = {
    "Good":         "Air quality is considered satisfactory. Enjoy outdoor activities.",
    "Satisfactory": "Air quality is acceptable. Unusually sensitive individuals should limit prolonged outdoor exertion.",
    "Moderate":     "Members of sensitive groups may experience health effects. Reduce prolonged outdoor exertion.",
    "Poor":         "Everyone may begin to experience health effects. Limit prolonged outdoor exertion.",
    "Very Poor":    "Health alert: everyone may experience more serious health effects. Avoid outdoor exertion.",
    "Severe":       "Health warning of emergency conditions. Everyone should avoid all outdoor exertion.",
}

FEATURE_UNITS = {
    "PM2.5": "µg/m³", "PM10": "µg/m³", "NO": "µg/m³",
    "NO2": "µg/m³",   "NOx": "µg/m³",  "NH3": "µg/m³",
    "CO": "mg/m³",    "SO2": "µg/m³",  "O3": "µg/m³",
    "Benzene": "µg/m³", "Toluene": "µg/m³",
}
FEATURE_DEFAULTS = {
    "PM2.5": 50.0, "PM10": 95.0, "NO": 10.0, "NO2": 22.0, "NOx": 24.0,
    "NH3": 16.0,   "CO": 0.9,    "SO2": 9.0, "O3": 31.0,
    "Benzene": 1.1, "Toluene": 3.0,
}
FEATURE_MAX = {
    "PM2.5": 500.0, "PM10": 600.0, "NO": 400.0, "NO2": 400.0, "NOx": 500.0,
    "NH3": 400.0,   "CO": 100.0,   "SO2": 200.0, "O3": 300.0,
    "Benzene": 50.0, "Toluene": 100.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCE LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner=False)
def load_metadata():
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_imputation_stats():
    with open(IMP_PATH, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"])
    return df  # read-only — never written back

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING & INFERENCE — exact Phase 4 pipeline
# ─────────────────────────────────────────────────────────────────────────────
def impute_input(raw_values: dict, city: str, imp_stats: dict) -> dict:
    """Apply city-specific (or global-fallback) median imputation."""
    city_meds   = imp_stats.get("city_medians", {}).get(city, {})
    global_meds = imp_stats.get("global_fallback_medians", {})
    imputed = {}
    for feat, val in raw_values.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            imputed[feat] = city_meds.get(feat) or global_meds.get(feat, 0.0)
        else:
            imputed[feat] = float(val)
    return imputed

def build_feature_vector(imputed: dict, feature_order: list) -> np.ndarray:
    """Build ordered float array and apply log1p — exact Phase 4 transform."""
    raw = np.array([imputed[f] for f in feature_order], dtype=float)
    return np.log1p(raw).reshape(1, -1)

def predict_aqi_bucket(model, X: np.ndarray, meta: dict):
    """Return (predicted_label, probabilities_dict)."""
    pred_int  = int(model.predict(X)[0])
    label     = meta["inv_label_map"][str(pred_int)]
    proba     = model.predict_proba(X)[0]
    proba_dict = {meta["bucket_order"][i]: float(proba[i])
                  for i in range(len(meta["bucket_order"]))}
    return label, proba_dict

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, delta: str = ""):
    st.markdown(
        f"""<div style='background:#f7f8fa;border:1px solid #e5e7eb;
        border-radius:8px;padding:14px 18px;margin-bottom:8px;'>
        <div style='font-size:11px;color:#57606a;'>{label}</div>
        <div style='font-size:22px;font-weight:700;color:#3b82d4;'>{value}</div>
        <div style='font-size:12px;color:#57606a;'>{delta}</div></div>""",
        unsafe_allow_html=True,
    )

def bucket_badge(label: str):
    color = BUCKET_COLORS.get(label, "#888")
    emoji = BUCKET_EMOJI.get(label, "")
    st.markdown(
        f"""<div style='display:inline-block;background:{color};color:#fff;
        font-size:28px;font-weight:700;padding:12px 28px;border-radius:10px;
        margin:8px 0;'>{emoji} {label}</div>""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    st.title("🌫️ Indian Air Quality Analytics & AQI Classification")
    st.markdown(
        "**An end-to-end academic Data Analytics + Machine Learning project** "
        "that analyses India's ambient air quality data (2015–2020) and "
        "predicts the AQI category (*AQI_Bucket*) from pollutant concentrations."
    )
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Dataset period", "2015 – 2020", "city_day.csv")
    with c2: metric_card("Cities covered", "26", "India")
    with c3: metric_card("Training rows", "16,018", "2015-01-01 → 2019-05-26")
    with c4: metric_card("Test rows", "8,832", "2019-05-27 → 2020-07-01")

    st.subheader("Project Objective")
    st.markdown(
        "Classify a day's ambient air quality into one of **six AQI categories** "
        "(Good → Satisfactory → Moderate → Poor → Very Poor → Severe) "
        "using pollutant concentration measurements from ground-level monitoring stations."
    )

    st.subheader("Machine-Learning Approach")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
- **Algorithm:** XGBoost gradient boosting (6-class multiclass)
- **Features:** 11 pollutant measurements (log₁p-transformed)
- **Target:** `AQI_Bucket` — 6 ordinal classes
- **Split:** Chronological (time-aware), no random shuffling
- **Imputation:** City-specific median from training data only
- **Class imbalance:** Addressed via `sample_weight` (balanced)
        """)
    with col2:
        st.markdown("""
- **Macro-F1 (test):** 0.760
- **Balanced Accuracy:** 0.757
- **Accuracy:** 0.789
- **Severe class recall:** 0.863
- **Train period:** Jan 2015 – May 2019
- **Test period:** May 2019 – Jul 2020
        """)

    st.subheader("Important Limitations")
    st.warning(
        "**AQI is NOT an input feature.** AQI_Bucket is derived directly from the "
        "numeric AQI score — using AQI to predict AQI_Bucket would be target leakage. "
        "The model learns the category from raw pollutant concentrations only.\n\n"
        "**Xylene is excluded** due to 61.3% global missingness (100% absent in 16/26 cities).\n\n"
        "**Original datasets are read-only** and were never modified."
    )

    st.subheader("Navigation")
    st.markdown("""
Use the **sidebar** to navigate between sections:

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview (this page) |
| 🔮 AQI Prediction | Enter pollutant values and get an AQI category prediction |
| 📊 Data Explorer | Interactive charts from the original dataset |
| 🤖 Model Information | Model config, metrics, preprocessing methodology |
| 📈 Feature Importance | Which pollutants drive the model's decisions |
| ⚠️ Data Quality & Limits | Missing data, outliers, known limitations |
| 📖 Methodology | Full project workflow and academic context |
    """)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AQI PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def page_prediction(model, meta, imp_stats):
    st.title("🔮 AQI Category Prediction")
    st.markdown(
        "Enter pollutant concentration values below. "
        "The model will predict the **AQI_Bucket** (air quality category) "
        "using the exact preprocessing pipeline applied during training."
    )
    st.info(
        "**Tip:** Select a city to auto-fill median values as starting points "
        "(based on training-set medians). You can then adjust individual values."
    )

    ALL_FEATURES = meta["all_features"]
    cities_available = sorted(imp_stats.get("city_medians", {}).keys())
    city_options = ["(Global average — no city selected)"] + cities_available

    col_city, _ = st.columns([1, 2])
    with col_city:
        city_sel = st.selectbox("City (for imputation defaults)", city_options, index=0)

    selected_city = city_sel if city_sel != "(Global average — no city selected)" else None

    # Determine defaults for sliders
    if selected_city:
        city_meds = imp_stats["city_medians"].get(selected_city, {})
    else:
        city_meds = {}
    global_meds = imp_stats.get("global_fallback_medians", {})

    def get_default(feat):
        v = city_meds.get(feat) or global_meds.get(feat) or FEATURE_DEFAULTS[feat]
        return float(v) if v is not None else FEATURE_DEFAULTS[feat]

    st.subheader("Pollutant Concentrations")
    st.caption("⚠️ Do not enter AQI or Xylene — they are excluded from this model.")

    input_vals = {}
    cols = st.columns(3)
    for idx, feat in enumerate(ALL_FEATURES):
        default = min(get_default(feat), FEATURE_MAX[feat])
        with cols[idx % 3]:
            input_vals[feat] = st.number_input(
                f"{feat} ({FEATURE_UNITS[feat]})",
                min_value=0.0,
                max_value=FEATURE_MAX[feat],
                value=round(default, 2),
                step=0.1,
                key=f"inp_{feat}",
            )

    st.divider()
    predict_btn = st.button("🔍 Predict AQI Category", type="primary", use_container_width=True)

    if predict_btn:
        # Validate
        warnings_list = []
        if input_vals.get("PM2.5", 0) > 450:
            warnings_list.append("PM2.5 > 450 µg/m³ — extremely high; please verify.")
        if input_vals.get("CO", 0) > 50:
            warnings_list.append("CO > 50 mg/m³ — unusually high; please verify.")
        if all(v == 0.0 for v in input_vals.values()):
            warnings_list.append("All values are zero — results will not be meaningful.")

        if warnings_list:
            for w in warnings_list:
                st.warning(w)

        # Preprocessing — exact Phase 4 pipeline
        imputed    = impute_input(input_vals, selected_city or "", imp_stats)
        X_vec      = build_feature_vector(imputed, ALL_FEATURES)
        pred_label, proba_dict = predict_aqi_bucket(model, X_vec, meta)

        st.subheader("Prediction Result")
        bucket_badge(pred_label)
        st.markdown(f"**{BUCKET_ADVICE[pred_label]}**")

        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            "AQI Category": list(proba_dict.keys()),
            "Probability":  [f"{v:.1%}" for v in proba_dict.values()],
            "Confidence":   list(proba_dict.values()),
        })
        st.bar_chart(
            pd.DataFrame(proba_dict, index=["Probability"]).T.rename(columns={"Probability": "Probability"}),
            use_container_width=True,
        )
        st.dataframe(
            prob_df[["AQI Category", "Probability"]].set_index("AQI Category"),
            use_container_width=True,
        )

        with st.expander("Preprocessing details (transparency)"):
            log_rows = []
            for feat in ALL_FEATURES:
                raw = input_vals[feat]
                imp = imputed[feat]
                log1p_val = np.log1p(imp)
                used_imp = "✓ imputed" if (raw == 0.0 and imp != 0.0) else ""
                log_rows.append({"Feature": feat, "Input": raw,
                                 "After imputation": round(imp, 4),
                                 "After log1p": round(log1p_val, 4),
                                 "Note": used_imp})
            st.dataframe(pd.DataFrame(log_rows).set_index("Feature"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
def page_data_explorer():
    st.title("📊 Data Explorer")
    st.markdown(
        "Interactive analytics from the original **city_day.csv** dataset "
        "(read-only — original file is never modified)."
    )

    df = load_dataset()
    BUCKET_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    POLLUTANTS   = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
                    "CO", "SO2", "O3", "Benzene", "Toluene"]

    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Explorer Filters")
    cities = sorted(df["City"].unique())
    sel_cities = st.sidebar.multiselect("Filter cities", cities, default=cities[:5])
    year_range = st.sidebar.slider("Year range", 2015, 2020, (2015, 2020))

    dff = df[
        df["City"].isin(sel_cities) &
        (df["Date"].dt.year >= year_range[0]) &
        (df["Date"].dt.year <= year_range[1])
    ].copy()
    st.caption(f"Showing {len(dff):,} rows for {len(sel_cities)} cities, {year_range[0]}–{year_range[1]}")

    tab1, tab2, tab3, tab4 = st.tabs(["AQI Distribution", "Pollutant Profiles",
                                       "City Comparison", "Seasonal Trends"])

    # ── TAB 1: AQI Distribution ──────────────────────────────────────────────
    with tab1:
        st.subheader("AQI_Bucket Class Distribution")
        bucket_df = (
            dff["AQI_Bucket"].dropna()
            .value_counts()
            .reindex(BUCKET_ORDER, fill_value=0)
            .reset_index()
        )
        bucket_df.columns = ["AQI_Bucket", "Count"]
        st.bar_chart(bucket_df.set_index("AQI_Bucket"), use_container_width=True)

        st.subheader("AQI Score Distribution (numeric)")
        aqi_valid = dff["AQI"].dropna()
        st.markdown(f"**n = {len(aqi_valid):,}** | Mean = {aqi_valid.mean():.1f} | Median = {aqi_valid.median():.0f} | Max = {aqi_valid.max():.0f}")
        hist_data = pd.cut(aqi_valid, bins=40).value_counts().sort_index()
        hist_df = pd.DataFrame({
            "AQI range": [str(i) for i in hist_data.index],
            "Count": hist_data.values,
        })
        st.bar_chart(hist_df.set_index("AQI range"), use_container_width=True)

    # ── TAB 2: Pollutant Profiles ─────────────────────────────────────────────
    with tab2:
        st.subheader("Pollutant Summary Statistics (selected cities & years)")
        stats_rows = []
        for p in POLLUTANTS:
            s = dff[p].dropna()
            if len(s) > 0:
                stats_rows.append({
                    "Pollutant": p,
                    "Unit": FEATURE_UNITS[p],
                    "n": len(s),
                    "Mean": round(s.mean(), 2),
                    "Median": round(s.median(), 2),
                    "Std": round(s.std(), 2),
                    "Max": round(s.max(), 2),
                    "Missing %": f"{dff[p].isna().mean()*100:.1f}%",
                })
        st.dataframe(pd.DataFrame(stats_rows).set_index("Pollutant"), use_container_width=True)

        st.subheader("Pollutant Medians by City")
        sel_poll = st.selectbox("Select pollutant", POLLUTANTS, index=0)
        city_med = (
            dff.groupby("City")[sel_poll]
            .median()
            .dropna()
            .sort_values(ascending=False)
        )
        st.bar_chart(city_med, use_container_width=True)

    # ── TAB 3: City Comparison ───────────────────────────────────────────────
    with tab3:
        st.subheader("Mean AQI by City (selected filter)")
        city_aqi = (
            dff.groupby("City")["AQI"]
            .mean()
            .dropna()
            .sort_values(ascending=False)
            .reset_index()
        )
        city_aqi.columns = ["City", "Mean AQI"]
        st.bar_chart(city_aqi.set_index("City"), use_container_width=True)
        st.dataframe(city_aqi, use_container_width=True)

        st.subheader("AQI_Bucket proportions by City")
        if len(sel_cities) > 0:
            pivot = (
                dff.dropna(subset=["AQI_Bucket"])
                .groupby(["City", "AQI_Bucket"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=BUCKET_ORDER, fill_value=0)
            )
            pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).round(3)
            st.dataframe(pivot_pct.style.format("{:.1%}"), use_container_width=True)

    # ── TAB 4: Seasonal Trends ───────────────────────────────────────────────
    with tab4:
        st.subheader("Monthly Mean AQI (selected filter)")
        dff["Month"] = dff["Date"].dt.month
        monthly = dff.groupby("Month")["AQI"].mean().reset_index()
        monthly["Month_name"] = monthly["Month"].map({
            1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
            7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
        st.bar_chart(monthly.set_index("Month_name")["AQI"], use_container_width=True)
        st.caption("Highest AQI in Nov–Jan (post-harvest burning + cold inversion); lowest in Jul–Sep (monsoon rainfall).")

        st.subheader("Yearly AQI trend")
        dff["Year"] = dff["Date"].dt.year
        yearly = dff.groupby("Year")["AQI"].agg(["mean", "median"]).reset_index()
        yearly.columns = ["Year", "Mean AQI", "Median AQI"]
        st.line_chart(yearly.set_index("Year"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL INFORMATION
# ─────────────────────────────────────────────────────────────────────────────
def page_model_info(meta):
    st.title("🤖 Model Information")

    st.subheader("Deployment Model")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Algorithm", meta["best_model_name"])
    with c2: metric_card("Target", meta["target"])
    with c3: metric_card("Train rows", f"{meta['train_rows']:,}")
    with c4: metric_card("Test rows", f"{meta['test_rows']:,}")

    st.subheader("Feature Sets")
    tab_a, tab_b = st.tabs(["XGBoost (11 features)", "Logistic Regression (7 features)"])
    with tab_a:
        st.markdown("**Model:** XGBoost (selected for deployment)")
        st.dataframe(
            pd.DataFrame({"Feature": meta["all_features"],
                          "Unit": [FEATURE_UNITS[f] for f in meta["all_features"]]}),
            use_container_width=True,
        )
    with tab_b:
        st.markdown("**Model:** Logistic Regression (low-VIF 7-feature set)")
        st.dataframe(
            pd.DataFrame({"Feature": meta["lr_features"],
                          "Unit": [FEATURE_UNITS[f] for f in meta["lr_features"]]}),
            use_container_width=True,
        )

    st.subheader("Excluded Variables")
    excl = pd.DataFrame([
        {"Column": "AQI", "Reason": "Target leakage — AQI_Bucket is deterministically derived from AQI. Using it as a feature would give the model the answer; no generalisation would occur."},
        {"Column": "Xylene", "Reason": "61.3% global missing; 100% absent in 16 of 26 cities. City-specific imputation would fabricate entire city profiles rather than fill occasional gaps."},
        {"Column": "City", "Reason": "Used only for city-specific median imputation. Not passed as a model feature to avoid overfitting on city identity."},
        {"Column": "Date", "Reason": "Used for chronological split only. Temporal features (Month, Season) not yet added to feature set."},
    ])
    st.dataframe(excl.set_index("Column"), use_container_width=True)

    st.subheader("Preprocessing Pipeline")
    st.markdown("""
1. **Load** `city_day.csv` read-only; parse `Date` → `datetime64`
2. **Drop** 4,681 rows with missing `AQI_Bucket` (target imputation prohibited)
3. **Sort** chronologically by `[Date, City]`
4. **Split** at `2019-05-27` — all dates before = train, on/after = test
5. **Imputation** — city-specific median from training rows only; global training median as fallback for cold-start cities
6. **log1p transform** — applied to all 11 features (no fitting required; prevents leakage)
7. **Label encoding** — Good=0 … Severe=5
8. **XGBoost training** with `sample_weight` (balanced class weights)
    """)

    st.subheader("Train / Test Split")
    split_df = pd.DataFrame([
        {"Split": "Train", "Date range": f"2015-01-01 → {meta['split_date']}",
         "Rows": meta["train_rows"], "Cities": 19},
        {"Split": "Test",  "Date range": f"{meta['split_date']} → 2020-07-01",
         "Rows": meta["test_rows"],  "Cities": 26},
    ])
    st.dataframe(split_df.set_index("Split"), use_container_width=True)
    st.info("**Chronological split** — no random shuffling. Ensures the model is never tested on observations it could have seen during training.")

    st.subheader("Class Imbalance Handling")
    st.markdown("""
- Training set imbalance ratio: **14.5×** (Moderate: 6,054 rows vs Good: 418 rows)
- Strategy: **`sample_weight='balanced'`** — each class contributes equally to the loss function
- SMOTE / synthetic oversampling was intentionally deferred to preserve temporal integrity
- Primary evaluation metric: **Macro-F1** (treats all classes equally regardless of size)
    """)

    st.subheader("Full Model Comparison — Evaluation Metrics")
    rows = []
    for name, ev in meta["evaluation"].items():
        rows.append({
            "Model": name,
            "Features": "LR(7)" if name == "LogisticRegression" else "All(11)",
            "Macro-F1 (train)": f"{ev['macro_f1_train']:.4f}",
            "Macro-F1 (test)":  f"{ev['macro_f1_test']:.4f}",
            "Wtd-F1 (test)":    f"{ev['weighted_f1_test']:.4f}",
            "Bal-Acc (test)":   f"{ev['balanced_acc_test']:.4f}",
            "Accuracy (test)":  f"{ev['accuracy_test']:.4f}",
            "Train time (s)":   f"{ev['train_time_s']:.2f}",
        })
    ev_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(ev_df, use_container_width=True)
    st.success("**XGBoost** selected: highest Macro-F1 (0.7598), Balanced Accuracy (0.7573), and overall accuracy (0.7893) — evaluated on the held-out chronological test set.")

    st.subheader("Confusion Matrix — XGBoost (Test Set)")
    BUCKET_ORDER = meta["bucket_order"]
    cm = meta["confusion_matrix"]["XGBoost"]
    cm_df = pd.DataFrame(cm, index=[f"True {b}" for b in BUCKET_ORDER],
                         columns=[f"Pred {b}" for b in BUCKET_ORDER])
    st.dataframe(cm_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
def page_feature_importance(meta):
    st.title("📈 Feature Importance")
    st.warning(
        "Feature importance indicates how much each pollutant contributed to the "
        "model's predictions. **It does NOT establish causation.** "
        "Correlation with AQI may reflect shared environmental sources or co-emission patterns."
    )

    ALL_FEATURES  = meta["all_features"]
    fi            = meta["feature_importance"]
    perm          = meta["permutation_importance"]

    tab1, tab2 = st.tabs(["Built-in Importance (Gain)", "Permutation Importance"])

    with tab1:
        st.subheader("XGBoost Built-in Feature Importance (Gain)")
        fi_df = (
            pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
            .sort_values("Importance", ascending=False)
            .set_index("Feature")
        )
        st.bar_chart(fi_df, use_container_width=True)
        st.dataframe(fi_df.style.format({"Importance": "{:.4f}"}), use_container_width=True)

    with tab2:
        st.subheader("Permutation Importance (mean macro-F1 drop, test set)")
        perm_df = pd.DataFrame([
            {"Feature": k, "Mean drop": v["mean"], "Std": v["std"]}
            for k, v in sorted(perm.items(), key=lambda x: -x[1]["mean"])
        ]).set_index("Feature")
        st.bar_chart(perm_df["Mean drop"], use_container_width=True)
        st.dataframe(perm_df.style.format({"Mean drop": "{:.4f}", "Std": "{:.4f}"}),
                     use_container_width=True)
        st.caption("Permutation importance: macro-F1 drop when each feature is randomly shuffled. "
                   "Larger drop = more important. Computed on 2,000 test-set samples, 10 repeats.")

    st.subheader("Interpretation")
    st.markdown("""
| Rank | Feature | Built-in | Permutation | Interpretation |
|------|---------|----------|-------------|----------------|
| 1 | PM2.5 | 0.288 | 0.248 | **Dominant predictor** — fine particulate matter strongly correlates with all AQI bands |
| 2 | PM10 | 0.206 | 0.154 | Second-strongest — coarse particulates add independent information beyond PM2.5 |
| 3 | CO | 0.137 | 0.173 | **Rises to 2nd in permutation** — particularly important for Ahmedabad's Severe events |
| 4 | SO2 | 0.057 | 0.009 | Industrial combustion indicator; built-in overestimates due to split-point sensitivity |
| 5–9 | NOx, O3, NH3, Toluene, NO | 0.05–0.04 | 0.05–0.01 | Moderate contributors |
| 10–11 | NO2, Benzene | 0.038–0.037 | 0.014/−0.002 | Weakest predictors; Benzene has near-zero permutation importance |
    """)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA QUALITY & LIMITATIONS
# ─────────────────────────────────────────────────────────────────────────────
def page_data_quality():
    st.title("⚠️ Data Quality & Limitations")

    st.subheader("Missing Data Summary")
    miss_df = pd.DataFrame([
        {"Column": "Xylene",       "Missing %": "61.3%", "Severity": "Critical", "Action": "Excluded from model"},
        {"Column": "PM10",         "Missing %": "37.7%", "Severity": "Moderate", "Action": "City-specific median imputation"},
        {"Column": "NH3",          "Missing %": "35.0%", "Severity": "Moderate", "Action": "City-specific median imputation"},
        {"Column": "Toluene",      "Missing %": "27.2%", "Severity": "Moderate", "Action": "City-specific median imputation"},
        {"Column": "Benzene",      "Missing %": "19.0%", "Severity": "Moderate", "Action": "City-specific median imputation"},
        {"Column": "AQI_Bucket",   "Missing %": "15.9%", "Severity": "Moderate", "Action": "Rows dropped — target imputation prohibited"},
        {"Column": "PM2.5",        "Missing %": "15.6%", "Severity": "Low",      "Action": "City-specific median imputation"},
        {"Column": "AQI",          "Missing %": "15.9%", "Severity": "Low",      "Action": "Excluded from features (leakage risk)"},
        {"Column": "NO / NO2 / O3","Missing %": "~12–14%","Severity":"Low",      "Action": "City-specific median imputation"},
        {"Column": "CO",           "Missing %": "7.0%",  "Severity": "Low",      "Action": "City-specific median imputation"},
    ])
    st.dataframe(miss_df.set_index("Column"), use_container_width=True)

    st.subheader("City-Specific Data Issues")
    city_issues = pd.DataFrame([
        {"City": "Ahmedabad", "Issue": "AQI values up to 2,049 (well above 400 Severe threshold). "
                                       "Mean AQI = 452 — nearly 2× Delhi's 259. Driven by extreme CO and SO2, not PM alone. NH3 100% missing."},
        {"City": "Mumbai",    "Issue": "61–80% missing across most pollutants including PM2.5, PM10, NO, NO2, SO2, O3."},
        {"City": "Jorapokhar","Issue": "100% missing for NOx, Benzene, Toluene, Xylene. PM2.5 68% missing."},
        {"City": "Chennai",   "Issue": "PM10 85% missing."},
        {"City": "Lucknow",   "Issue": "PM10 100% missing."},
        {"City": "Patna",     "Issue": "PM10 and NH3 each ~90% missing."},
    ])
    st.dataframe(city_issues.set_index("City"), use_container_width=True)

    st.subheader("Ahmedabad Sensitivity Analysis")
    st.markdown("""
Ahmedabad was retained in the main dataset per analytical decision. A sensitivity test showed:

| Evaluation set | Macro-F1 |
|---|---|
| Full test set (with Ahmedabad) | **0.7598** |
| Test set excluding Ahmedabad | 0.7656 (+0.006) |
| Ahmedabad rows only | 0.5580 |

Ahmedabad's unique CO/SO2-driven AQI profile (with PM10 entirely missing)
makes it the hardest city to classify. Model correctly identifies Severe events
(86% recall) but struggles with intermediate buckets.
    """)

    st.subheader("Class Imbalance & Temporal Shift")
    st.markdown("""
- **Imbalance ratio: 14.5×** (Moderate: 6,054 vs Good: 418 in training)
- **Temporal drift:** Test set (2019–2020) has proportionally more Good/Satisfactory
  days than training (2015–2019) — air quality genuinely improved in later years,
  amplified by the 2020 COVID-19 lockdown
- This means the model was trained on historically dirtier air than it is tested on
- PM2.5 mean is 38% lower in test vs train; CO mean is 41% lower

**Cold-start cities** (7 cities that appear only in test set):
Aizawl, Bhopal, Chandigarh, Coimbatore, Ernakulam, Kochi, Shillong —
monitoring started after the split date. These cities receive global-training-median
imputation. Despite this, they achieve 0.68–0.83 macro-F1.
    """)

    st.subheader("Other Known Data Issues")
    st.markdown("""
- **PM10 = 1,000** — likely a sensor ceiling/cap value (right-censored); affects a small number of rows
- **Zero values in CO, Benzene, Toluene** — likely below detection limit (8–16% zeros), not true absence
- **Missingness is MAR (Missing At Random)** — clusters in 2015–2017 early years as the monitoring network expanded; not randomly distributed
- **Xylene** — excluded from all analysis due to 61.3% global missing; completely absent in 16/26 cities
    """)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
def page_methodology():
    st.title("📖 Methodology")

    st.subheader("Project Workflow")
    st.markdown("""
```
Raw Dataset (city_day.csv, 29,531 rows, 26 cities, 2015–2020)
    │
    ▼
Phase 1 — Dataset Inspection
    Shape · dtypes · date ranges · duplicate check · basic profiling
    │
    ▼
Phase 2 — Exploratory Data Analysis
    Missing values · distributions · AQI_Bucket consistency check
    Outlier detection (IQR + Z-score) · seasonal patterns · city rankings
    Spearman correlations · descriptive statistics per pollutant
    │
    ▼
Phase 3 — Preprocessing & Split Design
    Drop rows with missing AQI_Bucket (4,681 rows)
    Chronological train/test split: 2019-05-27
    City-specific median imputation (fit on train only)
    Multicollinearity analysis (Pearson + VIF)
    Feature matrix: 11 pollutants (Xylene and AQI excluded)
    log1p transformation · Label encoding
    │
    ▼
Phase 4 — Model Training & Evaluation
    DummyClassifier · Logistic Regression · Random Forest
    Gradient Boosting · XGBoost · LightGBM
    Evaluation: Macro-F1, Balanced Accuracy, Weighted-F1, Accuracy
    Confusion matrices · Per-class recall · Feature importance
    Ahmedabad sensitivity · Cold-start analysis · Temporal analysis
    │
    ▼
Phase 5 — Streamlit Application & Submission
    Single-file ArnabKundu_IndianAirQualityAnalytics.py · Loads artifacts, not original CSVs for inference
    Read-only data exploration · Model transparency pages
    Academic report · README · requirements.txt
```
    """)

    st.subheader("Key Methodological Decisions")
    decisions = pd.DataFrame([
        {"Decision": "Exclude AQI from features",
         "Rationale": "AQI_Bucket is deterministically derived from AQI — using it would be target leakage with ~100% artificial accuracy"},
        {"Decision": "Exclude Xylene",
         "Rationale": "61.3% global missing; 100% absent in 16/26 cities — imputation would fabricate entire city profiles"},
        {"Decision": "Chronological split, not random",
         "Rationale": "Environmental data is temporally autocorrelated; random splitting leaks future observations into training"},
        {"Decision": "City-specific median imputation from training set only",
         "Rationale": "Cities have radically different pollutant baselines; global mean would inject wrong-city profiles. Training-only prevents data leakage"},
        {"Decision": "log1p transformation",
         "Rationale": "All 11 pollutants are right-skewed (skew 1.4–27.6); log1p reduces Benzene skew from 27.6 to 0.9"},
        {"Decision": "class_weight='balanced'",
         "Rationale": "14.5× imbalance between Moderate and Good classes; balanced weights prevent majority-class dominance"},
        {"Decision": "Macro-F1 as primary metric",
         "Rationale": "Accuracy is misleading for imbalanced classification; macro-F1 treats all 6 classes equally"},
        {"Decision": "Retain Ahmedabad",
         "Rationale": "Analytically significant outlier; documented in sensitivity analysis rather than silently removed"},
    ])
    st.dataframe(decisions.set_index("Decision"), use_container_width=True)

    st.subheader("Academic & Ethical Notes")
    st.markdown("""
- The **original datasets were never modified** — all preprocessing is performed in-memory
- **Feature importance ≠ causation** — model association is documented as such
- This project is **distinctly different** from standard class examples (Supermarket Sales, House Price Prediction) —
  it combines time-series environmental data, multi-class ordinal classification, temporal generalization,
  and an Ahmedabad outlier sensitivity analysis that would not arise in typical tabular ML projects
- The model's predictions are **health-advisory guidance, not medical or regulatory advice**
- AQI_Bucket categories follow the **Central Pollution Control Board (CPCB), India** classification scheme
    """)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Indian Air Quality Analytics",
        page_icon="🌫️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load artifacts
    with st.spinner("Loading model and artifacts…"):
        try:
            model    = load_model()
            meta     = load_metadata()
            imp_stats = load_imputation_stats()
        except FileNotFoundError as e:
            st.error(
                f"**Artifact not found:** `{e}`\n\n"
                "Ensure the `artifacts/` directory contains:\n"
                "- `best_model.pkl`\n- `imputation_stats.json`\n- `model_metadata.json`\n\n"
                "Run `phase4_modeling.py` to regenerate artifacts."
            )
            st.stop()

    # Sidebar navigation
    st.sidebar.title("🌫️ Air Quality Analytics")
    st.sidebar.markdown(f"**Model:** {meta['best_model_name']} | **Macro-F1:** 0.7598")
    st.sidebar.divider()

    pages = {
        "🏠 Home":                    "home",
        "🔮 AQI Prediction":          "predict",
        "📊 Data Explorer":           "explorer",
        "🤖 Model Information":       "model_info",
        "📈 Feature Importance":      "feat_imp",
        "⚠️ Data Quality & Limits":  "data_quality",
        "📖 Methodology":             "methodology",
    }
    page_choice = st.sidebar.radio("Navigation", list(pages.keys()))
    page_key = pages[page_choice]

    st.sidebar.divider()
    st.sidebar.caption(
        "**Indian Air Quality Analytics**\n\n"
        "Author: Arnab\n"
        "Dataset: city_day.csv (2015–2020)\n"
        "Original files: read-only ✓"
    )

    # Route to page
    if page_key == "home":
        page_home()
    elif page_key == "predict":
        page_prediction(model, meta, imp_stats)
    elif page_key == "explorer":
        page_data_explorer()
    elif page_key == "model_info":
        page_model_info(meta)
    elif page_key == "feat_imp":
        page_feature_importance(meta)
    elif page_key == "data_quality":
        page_data_quality()
    elif page_key == "methodology":
        page_methodology()

if __name__ == "__main__":
    main()
