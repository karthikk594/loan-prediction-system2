from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.loan_prediction.modeling import (
    clean_dataframe,
    explain_prediction,
    feature_importance_frame,
    load_artifacts,
    save_artifacts,
    train_decision_tree,
)


st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

TRAINING_DATASET_PATH = Path("data/loan_approval_dataset.csv")
MODEL_OUTPUT_PATH = Path("models/loan_decision_tree.joblib")
TARGET_COLUMN = "loan_status"
APPROVED_LABEL = "Approved"
TREE_MAX_DEPTH = 4
TREE_MIN_SAMPLES_SPLIT = 4


def render_header() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(251, 191, 36, 0.22), transparent 24%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.18), transparent 26%),
                linear-gradient(135deg, #fff7ed 0%, #eff6ff 46%, #ecfeff 100%);
            color: #111827;
        }
        .stApp, .stApp p, .stApp label, .stApp span, .stApp div, .stApp li, .stApp h2, .stApp h3 {
            color: #111827;
        }
        .stMarkdown, .stText, .stCaption {
            color: #111827;
        }
        .hero {
            padding: 1.5rem 1.8rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #1f2937, #0f766e);
            color: #ffffff;
            box-shadow: 0 26px 60px rgba(15, 23, 42, 0.2);
            margin-bottom: 1rem;
        }
        .hero-title {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            margin: 0;
            font-size: 2.6rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            font-family: "Trebuchet MS", "Avenir Next", "Segoe UI", sans-serif;
            color: #ffffff;
        }
        .hero-logo {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 3.2rem;
            height: 3.2rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.14);
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.18);
            font-size: 1.8rem;
        }
        .hero-title span:last-child {
            color: #ffffff;
            text-shadow: 0 2px 18px rgba(255, 255, 255, 0.14);
        }
        .hero h1, .hero h1 span {
            color: #ffffff !important;
        }
        .panel {
            border-radius: 20px;
            padding: 1rem;
            background: rgba(255, 251, 235, 0.88);
            border: 1px solid rgba(251, 191, 36, 0.28);
            box-shadow: 0 18px 40px rgba(148, 163, 184, 0.18);
        }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            color: #111827;
        }
        [data-testid="stDataFrame"] div {
            color: #111827;
        }
        .stButton > button {
            background: linear-gradient(135deg, #0f766e, #0284c7);
            color: #ffffff;
            border: none;
        }
        .stSelectbox label, .stNumberInput label {
            color: #111827;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero">
            <h1 class="hero-title">
                <span class="hero-logo">🏦</span>
                <span>Loan Approval Predictor</span>
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_training_dataset() -> pd.DataFrame:
    if not TRAINING_DATASET_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found at {TRAINING_DATASET_PATH}")
    dataset = pd.read_csv(TRAINING_DATASET_PATH)
    return clean_dataframe(dataset)


def ensure_trained_model(dataset: pd.DataFrame):
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_OUTPUT_PATH.exists():
        return load_artifacts(str(MODEL_OUTPUT_PATH))

    artifacts = train_decision_tree(
        data=dataset,
        target_column=TARGET_COLUMN,
        positive_label=APPROVED_LABEL,
        max_depth=TREE_MAX_DEPTH,
        min_samples_split=TREE_MIN_SAMPLES_SPLIT,
    )
    save_artifacts(artifacts, str(MODEL_OUTPUT_PATH))
    return artifacts


def build_applicant_input_frame(
    data: pd.DataFrame, feature_columns: list[str], numeric_features: list[str]
) -> pd.DataFrame:
    values: dict[str, object] = {}
    columns = st.columns(2)

    for index, column_name in enumerate(feature_columns):
        column = columns[index % 2]
        with column:
            if column_name in numeric_features:
                series = pd.to_numeric(data[column_name], errors="coerce")
                fallback = float(series.median()) if series.notna().any() else 0.0
                minimum = float(series.min()) if series.notna().any() else 0.0
                maximum = float(series.max()) if series.notna().any() else max(fallback + 1.0, 1.0)
                step = 1.0 if maximum - minimum > 10 else 0.1
                values[column_name] = st.number_input(
                    column_name,
                    value=fallback,
                    min_value=minimum,
                    max_value=maximum if maximum > minimum else minimum + 1.0,
                    step=step,
                )
            else:
                options = data[column_name].dropna().astype(str).unique().tolist()
                options = sorted(options) if options else ["Unknown"]
                values[column_name] = st.selectbox(column_name, options=options)

    return pd.DataFrame([values])


def main() -> None:
    render_header()

    try:
        dataset = load_training_dataset()
        artifacts = ensure_trained_model(dataset)
    except Exception as error:
        st.error(f"Unable to load or train the model: {error}")
        return

    feature_columns = artifacts.X_train.columns.tolist()

    top_left, top_right = st.columns([1.2, 0.8], gap="large")

    with top_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Applicant Details")
        applicant_frame = build_applicant_input_frame(
            dataset[feature_columns + [TARGET_COLUMN]],
            feature_columns,
            artifacts.numeric_features,
        )
        predict_button = st.button("Predict Loan Result", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with top_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Model Accuracy")
        st.metric("Accuracy", f"{artifacts.accuracy * 100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    if not predict_button:
        st.info("Fill in the applicant details and click Predict Loan Result.")
        return

    explanation = explain_prediction(artifacts, applicant_frame)
    prediction = explanation["prediction"]
    approved = prediction == artifacts.positive_label

    decision_title = "Loan Approved" if approved else "Loan Rejected"
    decision_message = st.success if approved else st.error
    decision_message(f"{decision_title}: predicted class is `{prediction}`.")

    probability_frame = pd.DataFrame(
        {
            "Class": list(explanation["probabilities"].keys()),
            "Probability": list(explanation["probabilities"].values()),
        }
    )

    result_col, chart_col = st.columns([1.0, 1.0], gap="large")

    with result_col:
        st.subheader("Decision Basis")
        if explanation["path_rules"]:
            for rule in explanation["path_rules"]:
                st.write(f"- {rule}")
        else:
            st.write("The tree reached a decision without additional visible splits.")

        top_features = feature_importance_frame(artifacts).head(5)
        if not top_features.empty:
            top_features["feature"] = (
                top_features["feature"]
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
            )
            st.subheader("Top Influencing Features")
            st.dataframe(top_features, use_container_width=True)

            basis_title = "Approval basis" if approved else "Rejection basis"
            st.subheader(basis_title)
            for _, row in top_features.iterrows():
                st.write(f"- {row['feature']} influenced the result with importance {row['importance']:.3f}.")

    with chart_col:
        st.subheader("Prediction Confidence")
        st.bar_chart(probability_frame.set_index("Class"))

        st.subheader("Feature Importance")
        top_feature_chart = feature_importance_frame(artifacts).head(10).copy()
        if not top_feature_chart.empty:
            top_feature_chart["feature"] = (
                top_feature_chart["feature"]
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
            )
            st.bar_chart(top_feature_chart.set_index("feature"))

    st.subheader("Training Data Class Distribution")
    distribution = (
        dataset[TARGET_COLUMN]
        .astype(str)
        .value_counts()
        .rename_axis("class")
        .reset_index(name="count")
    )
    st.bar_chart(distribution.set_index("class"))


if __name__ == "__main__":
    main()
