import os
import re
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="QA Analytics & ML Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

RUBRIC_COLS: List[str] = [
    "Greeting Opening/Ending",
    "Active Listening",
    "Product/Service Knowledge",
    "Solution Explanation",
    "No Assistance",
    "Politeness and Empathy",
    "Language Proficiency",
    "First Contact Resolution",
    "Escalation",
    "QA Satisfaction",
]

EXCEL_ERRORS = {"#DIV/0!", "#VALUE!", "#REF!", "#NAME?", "#N/A", "#NUM!", "NaN", "nan", ""}


# -----------------------------------------------------------------------------
# DATA LOADING & CLEANING PIPELINE (CACHED)
# -----------------------------------------------------------------------------
def _parse_rubric_score(val) -> float:
    """Extracts leading numeric float (1.0, 0.5, 0.0) from strings like '1 (Good)'."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in EXCEL_ERRORS:
        return np.nan
    match = re.match(r"^([0-9]*\.?[0-9]+)", s)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    return np.nan


def _clean_improvement_tags(series: pd.Series) -> pd.Series:
    """Normalizes casing, whitespace, and trailing punctuation on improvement tags."""
    cleaned = []
    for val in series.dropna():
        # Split on comma or slash if multiple tags appear in one cell
        parts = re.split(r"[,;/|]+", str(val))
        for p in parts:
            tag = re.sub(r"^[\s\.\-\?\!\:\;]+|[\s\.\-\?\!\:\;]+$", "", p).strip()
            if tag and tag not in EXCEL_ERRORS and len(tag) > 1:
                # Keep short acronyms uppercase (e.g., CES), Title Case others
                if tag.upper() in {"CES", "FCR", "QA", "SOP", "PNR", "GDS"}:
                    cleaned.append(tag.upper())
                else:
                    cleaned.append(tag.title())
    return pd.Series(cleaned)


@st.cache_data(show_spinner="Parsing QA2026 sheet & normalizing rubric scores...")
def load_and_clean_data(file_source) -> pd.DataFrame:
    """Reads the QA2026 sheet, resolves the two-row header quirk, and cleans all fields."""
    raw_df = pd.read_excel(file_source, sheet_name="QA2026", header=0, engine="openpyxl")

    # Row 0 in DataFrame (Excel Row 2) contains the actual sub-headers
    sub_headers = raw_df.iloc[0].tolist()
    top_headers = raw_df.columns.tolist()

    # Exact positional mapping based on schema specification
    # Cols 0..3: Date, Recording ID, Agent, QC by
    # Cols 4..13: 10 Rubric Criteria
    # Col 14: Total Score
    # Col 15: Remarks
    # Col 16: Areas of Improvement
    # Col 17: Call Type
    mapped_columns = {}
    for idx, top_col in enumerate(top_headers):
        top_clean = str(top_col).strip()
        sub_clean = str(sub_headers[idx]).strip() if idx < len(sub_headers) and pd.notna(sub_headers[idx]) else ""

        if idx == 0 or top_clean.lower() == "date":
            mapped_columns[top_col] = "Date"
        elif idx == 1 or "recording" in top_clean.lower():
            mapped_columns[top_col] = "Recording ID"
        elif idx == 2 or top_clean.lower() == "agent":
            mapped_columns[top_col] = "Agent"
        elif idx == 3 or "qc" in top_clean.lower():
            mapped_columns[top_col] = "QC by"
        elif 4 <= idx <= 13:
            # Prefer our canonical rubric name if sub_header matches or fallback positionally
            canonical = RUBRIC_COLS[idx - 4]
            mapped_columns[top_col] = canonical
        elif idx == 14 or "total" in top_clean.lower():
            mapped_columns[top_col] = "Total Score"
        elif idx == 15 or "remark" in sub_clean.lower():
            mapped_columns[top_col] = "Remarks"
        elif idx == 16 or "improvement" in sub_clean.lower():
            mapped_columns[top_col] = "Areas of Improvement"
        elif idx == 17 or "call type" in sub_clean.lower():
            mapped_columns[top_col] = "Call Type"
        else:
            mapped_columns[top_col] = f"_ignore_{idx}"

    # Drop the sub-header row (Excel Row 2) and rename columns
    df = raw_df.iloc[1:].copy().reset_index(drop=True)
    df = df.rename(columns=mapped_columns)

    # Keep only valid mapped columns (drops Unnamed: 18..25)
    keep_cols = [c for c in df.columns if not str(c).startswith("_ignore_")]
    df = df[keep_cols].copy()

    # Replace Excel formula errors across the dataset
    df = df.replace(list(EXCEL_ERRORS), np.nan)

    # Clean Date & drop rows where Date or Agent is null
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Agent"] = df["Agent"].astype(str).str.strip()
    df.loc[df["Agent"].isin(["nan", "None", ""]), "Agent"] = np.nan
    df = df.dropna(subset=["Date", "Agent"]).reset_index(drop=True)

    # Clean categorical text fields
    df["Recording ID"] = df["Recording ID"].astype(str).str.strip()
    df["QC by"] = df["QC by"].fillna("Unknown").astype(str).str.strip()
    df["Call Type"] = df["Call Type"].fillna("Unspecified").astype(str).str.strip()
    df.loc[df["Call Type"] == "", "Call Type"] = "Unspecified"
    df["Remarks"] = df["Remarks"].fillna("").astype(str).str.strip()

    # Parse 10 Rubric Criteria strings -> float
    for col in RUBRIC_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_rubric_score)
        else:
            df[col] = np.nan

    # Parse Total Score (recalculate from rubric sum if Excel had #DIV/0! or NaN)
    df["Total Score"] = pd.to_numeric(df["Total Score"], errors="coerce")
    rubric_sum = df[RUBRIC_COLS].sum(axis=1, min_count=1)
    df["Total Score"] = df["Total Score"].fillna(rubric_sum).clip(0.0, 10.0)
    df = df.dropna(subset=["Total Score"]).reset_index(drop=True)

    # Feature engineering for time trends & ML
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Month_Num"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
    df["Day_of_Week"] = df["Date"].dt.day_name()
    df["Is_Low_Score"] = (df["Total Score"] < 8.0).astype(int)

    return df


# -----------------------------------------------------------------------------
# MACHINE LEARNING MODELS (CACHED)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Fitting Fairness & Calibration Ridge Model...")
def fit_examiner_fairness_model(df_hash: int, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Fits a Ridge Regression fixed-effects model:
        Total Score ~ QC_by + Agent + Call_Type
    Controls for agent skill and call difficulty to estimate examiner strictness bias
    and compute fairness-adjusted agent scores.
    """
    model_df = df[["Total Score", "QC by", "Agent", "Call Type"]].dropna().copy()
    if len(model_df) < 30 or model_df["QC by"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame(), 0.0

    X = model_df[["QC by", "Agent", "Call Type"]]
    y = model_df["Total Score"].values

    encoder = OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False)
    X_encoded = encoder.fit_transform(X)
    feature_names = encoder.get_feature_names_out(["QC by", "Agent", "Call Type"])

    ridge = Ridge(alpha=5.0, random_state=42)
    ridge.fit(X_encoded, y)

    # Extract examiner coefficients and center around weighted mean = 0
    qc_effects = {}
    for fname, coef in zip(feature_names, ridge.coef_):
        if fname.startswith("QC by_"):
            qc_name = fname.replace("QC by_", "")
            qc_effects[qc_name] = coef

    qc_counts = model_df["QC by"].value_counts()
    total_obs = sum(qc_counts.get(k, 0) for k in qc_effects)
    weighted_mean_effect = (
        sum(qc_effects[k] * qc_counts.get(k, 0) for k in qc_effects) / total_obs
        if total_obs > 0
        else 0.0
    )

    # Strictness offset: negative means stricter than average; positive means more lenient
    centered_effects = {k: v - weighted_mean_effect for k, v in qc_effects.items()}

    examiner_summary = []
    for qc_name, offset in centered_effects.items():
        raw_mean = model_df.loc[model_df["QC by"] == qc_name, "Total Score"].mean()
        count = int(qc_counts.get(qc_name, 0))
        examiner_summary.append(
            {
                "QC Examiner": qc_name,
                "Evaluated Calls": count,
                "Raw Avg Score": round(raw_mean, 3),
                "Strictness Offset (Pts)": round(offset, 3),
                "Calibration Note": (
                    "Stricter than baseline" if offset < -0.05 else ("More lenient than baseline" if offset > 0.05 else "Well calibrated")
                ),
            }
        )
    examiner_df = pd.DataFrame(examiner_summary).sort_values("Strictness Offset (Pts)")

    # Adjust each call score by removing the examiner's bias offset
    model_df["Examiner_Offset"] = model_df["QC by"].map(centered_effects).fillna(0.0)
    model_df["Adjusted_Score"] = (model_df["Total Score"] - model_df["Examiner_Offset"]).clip(0.0, 10.0)

    leaderboard = (
        model_df.groupby("Agent", as_index=False)
        .agg(
            Calls=("Total Score", "count"),
            Raw_Avg=("Total Score", "mean"),
            Fairness_Adjusted_Avg=("Adjusted_Score", "mean"),
        )
        .sort_values("Fairness_Adjusted_Avg", ascending=False)
    )
    leaderboard["Net_Adjustment"] = leaderboard["Fairness_Adjusted_Avg"] - leaderboard["Raw_Avg"]
    leaderboard["Raw_Rank"] = leaderboard["Raw_Avg"].rank(ascending=False, method="min").astype(int)
    leaderboard["Adjusted_Rank"] = leaderboard["Fairness_Adjusted_Avg"].rank(ascending=False, method="min").astype(int)
    leaderboard["Rank_Change"] = leaderboard["Raw_Rank"] - leaderboard["Adjusted_Rank"]

    r2 = float(ridge.score(X_encoded, y))
    return examiner_df, leaderboard, r2


@st.cache_resource(show_spinner="Training Low-Score Risk Random Forest Classifier...")
def train_risk_classifier(df_hash: int, df: pd.DataFrame) -> Tuple[Optional[Pipeline], dict, pd.DataFrame]:
    """Trains a RandomForestClassifier to predict Total Score < 8.0."""
    features = ["Agent", "Call Type", "Day_of_Week", "Month_Num"]
    ml_df = df[features + ["Is_Low_Score"]].dropna().copy()

    if len(ml_df) < 50 or ml_df["Is_Low_Score"].nunique() < 2:
        return None, {}, pd.DataFrame()

    X = ml_df[features]
    y = ml_df["Is_Low_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Agent", "Call Type", "Day_of_Week"]),
            ("num", "passthrough", ["Month_Num"]),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", clf)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "base_risk_rate": float(y.mean()),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    # Aggregate One-Hot feature importances back to original 4 feature groups
    ohe_names = pipeline.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(
        ["Agent", "Call Type", "Day_of_Week"]
    )
    all_names = list(ohe_names) + ["Month_Num"]
    importances = pipeline.named_steps["model"].feature_importances_

    grouped_imp = {"Agent": 0.0, "Call Type": 0.0, "Day of Week": 0.0, "Month": 0.0}
    for name, imp in zip(all_names, importances):
        if name.startswith("Agent_"):
            grouped_imp["Agent"] += imp
        elif name.startswith("Call Type_"):
            grouped_imp["Call Type"] += imp
        elif name.startswith("Day_of_Week_"):
            grouped_imp["Day of Week"] += imp
        else:
            grouped_imp["Month"] += imp

    imp_df = (
        pd.DataFrame({"Feature": list(grouped_imp.keys()), "Importance": list(grouped_imp.values())})
        .sort_values("Importance", ascending=True)
    )

    return pipeline, metrics, imp_df


# -----------------------------------------------------------------------------
# SIDEBAR: FILE UPLOADER & GLOBAL FILTERS
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ QA Control Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload Weekly QA Excel (.xlsx)",
    type=["xlsx"],
    help="Defaults to local 'Agent QA Report.xlsx' if available.",
)

default_path = "Agent QA Report.xlsx"
data_source = None
if uploaded_file is not None:
    data_source = uploaded_file
elif os.path.exists(default_path):
    data_source = default_path

if data_source is None:
    st.title("📊 Call Center QA Analytics & ML Command Center")
    st.info(
        "👋 Welcome! Please upload **`Agent QA Report.xlsx`** using the sidebar file uploader "
        "(or place `Agent QA Report.xlsx` in the same folder as `app.py`)."
    )
    st.stop()

try:
    df_full = load_and_clean_data(data_source)
except Exception as e:
    st.error(f"Failed to read sheet `QA2026` from the Excel file. Details: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")

min_date = df_full["Date"].min().date()
max_date = df_full["Date"].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_dt, end_dt = pd.to_datetime(min_date), pd.to_datetime(max_date)

all_agents = sorted(df_full["Agent"].unique().tolist())
selected_agents = st.sidebar.multiselect("Agent(s)", options=all_agents, default=[])

all_qcs = sorted(df_full["QC by"].unique().tolist())
selected_qcs = st.sidebar.multiselect("QC Examiner(s)", options=all_qcs, default=[])

all_call_types = sorted(df_full["Call Type"].unique().tolist())
selected_call_types = st.sidebar.multiselect("Call Type(s)", options=all_call_types, default=[])

# Apply global filters
mask = (df_full["Date"] >= start_dt) & (df_full["Date"] <= end_dt)
if selected_agents:
    mask &= df_full["Agent"].isin(selected_agents)
if selected_qcs:
    mask &= df_full["QC by"].isin(selected_qcs)
if selected_call_types:
    mask &= df_full["Call Type"].isin(selected_call_types)

df = df_full.loc[mask].copy()

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(df):,}** of **{len(df_full):,}** evaluated calls.")

if df.empty:
    st.warning("No QA records match your current sidebar filters. Please broaden your selection.")
    st.stop()


# -----------------------------------------------------------------------------
# MAIN HEADER & 4 INTERACTIVE TABS
# -----------------------------------------------------------------------------
st.title("📊 Call Center QA Analytics & ML Command Center")
st.caption(
    f"Active Period: **{start_dt.strftime('%b %d, %Y')}** to **{end_dt.strftime('%b %d, %Y')}** "
    f"| Evaluated Calls: **{len(df):,}** | Active Agents: **{df['Agent'].nunique()}**"
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 1. Executive Overview & Trends",
        "🎯 2. Agent 360° Coaching & Skill Radar",
        "⚖️ 3. QA Examiner Calibration & Fairness",
        "🤖 4. Low-Score Risk Predictor (ML)",
    ]
)

# =============================================================================
# TAB 1: EXECUTIVE OVERVIEW & TRENDS
# =============================================================================
with tab1:
    total_calls = len(df)
    avg_score = df["Total Score"].mean()
    pass_rate = (df["Total Score"] >= 8.0).mean() * 100
    fcr_rate = df["First Contact Resolution"].mean() * 100 if df["First Contact Resolution"].notna().any() else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Evaluated Calls", f"{total_calls:,}")
    c2.metric("Average QA Score", f"{avg_score:.2f} / 10")
    c3.metric("Pass Rate (≥ 8.0)", f"{pass_rate:.1f}%")
    c4.metric("First Contact Resolution %", f"{fcr_rate:.1f}%")

    st.markdown("---")

    col_left, col_right = st.columns([1.4, 1.0])

    with col_left:
        granularity = st.radio(
            "Trend Granularity",
            options=["Weekly", "Monthly"],
            horizontal=True,
            key="trend_gran",
        )
        time_col = "Week" if granularity == "Weekly" else "Month"
        trend_df = (
            df.groupby(time_col, as_index=False)
            .agg(Avg_Score=("Total Score", "mean"), Call_Volume=("Recording ID", "count"))
            .sort_values(time_col)
        )

        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Bar(
                x=trend_df[time_col],
                y=trend_df["Call_Volume"],
                name="Evaluated Calls",
                marker_color="rgba(99, 110, 250, 0.25)",
                yaxis="y2",
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=trend_df[time_col],
                y=trend_df["Avg_Score"],
                name="Avg QA Score",
                mode="lines+markers",
                line=dict(color="#00CC96", width=3),
                fill="tozeroy",
                fillcolor="rgba(0, 204, 150, 0.12)",
            )
        )
        fig_trend.update_layout(
            title=f"{granularity} QA Score & Evaluation Volume Trend",
            yaxis=dict(title="Avg QA Score (0-10)", range=[0, 10.5]),
            yaxis2=dict(title="Call Volume", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        min_calls_ct = st.slider("Min Calls per Call Type", min_value=5, max_value=100, value=15, step=5)
        ct_stats = (
            df.groupby("Call Type", as_index=False)
            .agg(Avg_Score=("Total Score", "mean"), Calls=("Recording ID", "count"))
        )
        ct_filtered = ct_stats[ct_stats["Calls"] >= min_calls_ct].sort_values("Avg_Score", ascending=True).head(10)

        if ct_filtered.empty:
            st.info("Lower the minimum call threshold to view Top 10 hardest Call Types.")
        else:
            fig_ct = px.bar(
                ct_filtered,
                x="Avg_Score",
                y="Call Type",
                orientation="h",
                color="Avg_Score",
                color_continuous_scale="RdYlGn",
                range_color=[6.0, 9.5],
                text=ct_filtered.apply(lambda r: f"{r['Avg_Score']:.2f} ({int(r['Calls'])})", axis=1),
                title="Top 10 Lowest-Scoring Call Types (Hardest Categories)",
            )
            fig_ct.update_traces(textposition="outside")
            fig_ct.update_layout(
                xaxis=dict(title="Average QA Score", range=[0, 11]),
                yaxis=dict(title=""),
                coloraxis_showscale=False,
                height=420,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_ct, use_container_width=True)


# =============================================================================
# TAB 2: AGENT 360° COACHING & SKILL RADAR
# =============================================================================
with tab2:
    agent_list = sorted(df["Agent"].unique().tolist())
    selected_coach_agent = st.selectbox("Select Agent for 360° Coaching Review", options=agent_list)

    agent_df = df[df["Agent"] == selected_coach_agent]
    agent_avg = agent_df["Total Score"].mean()
    team_avg = df["Total Score"].mean()
    delta_team = agent_avg - team_avg

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Evaluated Calls", f"{len(agent_df):,}")
    m2.metric("Agent Avg Score", f"{agent_avg:.2f}", delta=f"{delta_team:+.2f} vs Team")
    m3.metric("Agent Pass Rate (≥ 8.0)", f"{(agent_df['Total Score'] >= 8.0).mean() * 100:.1f}%")
    m4.metric("Low-Score Calls (< 8.0)", f"{int((agent_df['Total Score'] < 8.0).sum()):,}")

    st.markdown("---")
    r_col1, r_col2 = st.columns([1.15, 1.0])

    with r_col1:
        agent_rubric = [agent_df[c].mean() * 100 for c in RUBRIC_COLS]
        team_rubric = [df[c].mean() * 100 for c in RUBRIC_COLS]

        # Close the polygon loop for Plotly Scatterpolar
        theta_closed = RUBRIC_COLS + [RUBRIC_COLS[0]]
        agent_closed = agent_rubric + [agent_rubric[0]]
        team_closed = team_rubric + [team_rubric[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=team_closed,
                theta=theta_closed,
                fill="toself",
                name="Team Average (%)",
                line=dict(color="#888888", dash="dash"),
                fillcolor="rgba(136, 136, 136, 0.15)",
            )
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=agent_closed,
                theta=theta_closed,
                fill="toself",
                name=f"{selected_coach_agent} (%)",
                line=dict(color="#636EFA", width=3),
                fillcolor="rgba(99, 110, 250, 0.25)",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
            title=f"10-Point Rubric Mastery: {selected_coach_agent} vs. Team",
            legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
            height=470,
            margin=dict(l=50, r=50, t=50, b=60),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with r_col2:
        tags_series = _clean_improvement_tags(agent_df["Areas of Improvement"])
        if tags_series.empty:
            st.info(f"No 'Areas of Improvement' tags recorded for **{selected_coach_agent}** in this period.")
        else:
            top_tags = tags_series.value_counts().head(10).reset_index()
            top_tags.columns = ["Area of Improvement", "Count"]
            top_tags = top_tags.sort_values("Count", ascending=True)

            fig_tags = px.bar(
                top_tags,
                x="Count",
                y="Area of Improvement",
                orientation="h",
                text="Count",
                color="Count",
                color_continuous_scale="OrRd",
                title=f"Top Recurring Areas of Improvement ({selected_coach_agent})",
            )
            fig_tags.update_traces(textposition="outside")
            fig_tags.update_layout(
                coloraxis_showscale=False,
                xaxis=dict(title="Frequency", range=[0, top_tags["Count"].max() * 1.2]),
                yaxis=dict(title=""),
                height=470,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_tags, use_container_width=True)

    st.subheader(f"🔍 Lowest-Scoring Calls & Examiner Remarks — {selected_coach_agent}")
    search_remark = st.text_input(
        "Search Remarks / Call Type / Recording ID",
        placeholder="Type keyword (e.g., refund, greeting, hold, CES)...",
    )

    low_calls = agent_df.sort_values(["Total Score", "Date"], ascending=[True, False]).copy()
    if search_remark.strip():
        q = search_remark.strip().lower()
        low_calls = low_calls[
            low_calls["Remarks"].str.lower().str.contains(q, na=False)
            | low_calls["Call Type"].str.lower().str.contains(q, na=False)
            | low_calls["Recording ID"].str.lower().str.contains(q, na=False)
            | low_calls["Areas of Improvement"].fillna("").str.lower().str.contains(q, na=False)
        ]

    display_cols = ["Date", "Recording ID", "QC by", "Call Type", "Total Score", "Areas of Improvement", "Remarks"]
    st.dataframe(
        low_calls[display_cols].head(50),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Total Score": st.column_config.NumberColumn("Total Score", format="%.2f"),
        },
    )


# =============================================================================
# TAB 3: QA EXAMINER CALIBRATION & FAIRNESS (STATISTICAL ML)
# =============================================================================
with tab3:
    st.markdown(
        "Examiners naturally vary in grading strictness. Below we compare raw score distributions across "
        "examiners and fit a **Ridge Fixed-Effects Regression** controlling for **Agent** and **Call Type** difficulty."
    )

    qc_col1, qc_col2 = st.columns([1.0, 1.3])

    with qc_col1:
        fig_box = px.box(
            df,
            x="QC by",
            y="Total Score",
            color="QC by",
            points="outliers",
            title="Total Score Distribution by QA Examiner",
        )
        fig_box.update_layout(showlegend=False, height=400, xaxis_title="QA Examiner", yaxis_title="Total Score (0-10)")
        st.plotly_chart(fig_box, use_container_width=True)

    with qc_col2:
        rubric_by_qc = df.groupby("QC by")[RUBRIC_COLS].mean().T.reset_index()
        rubric_by_qc = rubric_by_qc.rename(columns={"index": "Criterion"})
        rubric_melted = rubric_by_qc.melt(id_vars="Criterion", var_name="QC Examiner", value_name="Avg Rating")
        rubric_melted["Avg Rating (%)"] = rubric_melted["Avg Rating"] * 100

        fig_grouped = px.bar(
            rubric_melted,
            x="Criterion",
            y="Avg Rating (%)",
            color="QC Examiner",
            barmode="group",
            title="Average Rubric Score (%) by Examiner Across All 10 Criteria",
        )
        fig_grouped.update_layout(
            xaxis_tickangle=-35,
            yaxis=dict(range=[0, 105]),
            height=400,
            margin=dict(l=20, r=20, t=50, b=80),
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

    st.markdown("---")
    st.subheader("⚖️ Ridge Regression Examiner Strictness & Fairness-Adjusted Leaderboard")

    df_hash = int(pd.util.hash_pandas_object(df[["Recording ID", "Total Score", "QC by", "Agent"]]).sum())
    examiner_df, leaderboard_df, r2_score = fit_examiner_fairness_model(df_hash, df)

    if examiner_df.empty:
        st.info("Not enough multi-examiner data under current filters to fit the calibration model.")
    else:
        st.caption(f"Model Explains **{r2_score * 100:.1f}%** of score variance across Agent, Call Type, and Examiner effects.")
        st.dataframe(examiner_df, use_container_width=True, hide_index=True)

        st.markdown("#### Fairness-Adjusted Agent Leaderboard")
        st.dataframe(
            leaderboard_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Raw_Avg": st.column_config.NumberColumn("Raw Avg Score", format="%.3f"),
                "Fairness_Adjusted_Avg": st.column_config.NumberColumn("Fairness-Adjusted Score", format="%.3f"),
                "Net_Adjustment": st.column_config.NumberColumn("Bias Adjustment", format="%+.3f"),
                "Rank_Change": st.column_config.NumberColumn("Rank Shift (+ = Improved)", format="%+d"),
            },
        )


# =============================================================================
# TAB 4: LOW-SCORE RISK PREDICTOR (MACHINE LEARNING SIMULATOR)
# =============================================================================
with tab4:
    st.markdown(
        "Predicts the probability that an upcoming call will result in a **Low-Score / Coaching Needed (`Total Score < 8.0`)** "
        "outcome based on **Agent**, **Call Type**, **Day of Week**, and **Month**."
    )

    ml_hash = int(pd.util.hash_pandas_object(df_full[["Recording ID", "Total Score"]]).sum())
    clf_pipeline, val_metrics, imp_df = train_risk_classifier(ml_hash, df_full)

    if clf_pipeline is None:
        st.warning("Insufficient class diversity in dataset to train the Low-Score Risk Classifier.")
    else:
        sim_col, val_col = st.columns([1.2, 1.0])

        with sim_col:
            st.subheader("🔮 Interactive Call Risk Simulator")
            s1, s2 = st.columns(2)
            sim_agent = s1.selectbox("Select Agent", options=sorted(df_full["Agent"].unique()), key="sim_agent")
            sim_call_type = s2.selectbox("Select Call Type", options=sorted(df_full["Call Type"].unique()), key="sim_ct")

            s3, s4 = st.columns(2)
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            sim_day = s3.selectbox("Day of Week", options=days_order, key="sim_day")
            sim_month = s4.slider("Month (1 = Jan, 12 = Dec)", min_value=1, max_value=12, value=11, key="sim_month")

            input_row = pd.DataFrame(
                [
                    {
                        "Agent": sim_agent,
                        "Call Type": sim_call_type,
                        "Day_of_Week": sim_day,
                        "Month_Num": sim_month,
                    }
                ]
            )
            risk_prob = float(clf_pipeline.predict_proba(input_row)[0][1]) * 100

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_prob,
                    number={"suffix": "%", "valueformat": ".1f"},
                    delta={"reference": val_metrics["base_risk_rate"] * 100, "suffix": "% vs Baseline"},
                    title={"text": "Predicted Low-Score (< 8.0) Risk Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#EF553B" if risk_prob >= 50 else ("#FFA15A" if risk_prob >= 30 else "#00CC96")},
                        "steps": [
                            {"range": [0, 30], "color": "rgba(0, 204, 150, 0.15)"},
                            {"range": [30, 60], "color": "rgba(255, 161, 90, 0.15)"},
                            {"range": [60, 100], "color": "rgba(239, 85, 59, 0.15)"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": 50},
                    },
                )
            )
            fig_gauge.update_layout(height=310, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Data-driven coaching recommendation based on agent's weakest criteria on this Call Type
            sub_hist = df_full[(df_full["Agent"] == sim_agent) & (df_full["Call Type"] == sim_call_type)]
            if len(sub_hist) < 3:
                sub_hist = df_full[df_full["Agent"] == sim_agent]

            weakest_3 = sub_hist[RUBRIC_COLS].mean().sort_values().head(3)
            focus_bullets = " | ".join([f"**{k}** ({v * 100:.0f}%)" for k, v in weakest_3.items()])
            st.info(f"💡 **Recommended Pre-Call Coaching Focus for {sim_agent}:** {focus_bullets}")

        with val_col:
            st.subheader("📊 Model Validation & Drivers")
            v1, v2 = st.columns(2)
            v1.metric("Validation Accuracy", f"{val_metrics['accuracy'] * 100:.1f}%")
            v2.metric("ROC-AUC Score", f"{val_metrics['roc_auc']:.3f}")

            fig_imp = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                text=imp_df["Importance"].apply(lambda x: f"{x * 100:.1f}%"),
                color="Importance",
                color_continuous_scale="Blues",
                title="Feature Importance (What Drives Low QA Scores?)",
            )
            fig_imp.update_traces(textposition="outside")
            fig_imp.update_layout(
                coloraxis_showscale=False,
                xaxis=dict(title="Relative Importance", range=[0, imp_df["Importance"].max() * 1.25]),
                yaxis=dict(title=""),
                height=340,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption(
                f"Trained on {val_metrics['train_rows']:,} calls and validated on {val_metrics['test_rows']:,} unseen calls "
                f"(Baseline Low-Score Rate: {val_metrics['base_risk_rate'] * 100:.1f}%)."
            )