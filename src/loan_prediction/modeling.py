from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


@dataclass
class TrainingArtifacts:
    model: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    accuracy: float
    confusion: np.ndarray
    report: dict
    target_name: str
    positive_label: str
    negative_label: str
    numeric_features: list[str]
    categorical_features: list[str]


def clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    for column in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[column] = cleaned[column].astype(str).str.strip()
    return cleaned


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def train_decision_tree(
    data: pd.DataFrame,
    target_column: str,
    positive_label: str,
    max_depth: int,
    min_samples_split: int,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingArtifacts:
    working = clean_dataframe(data)
    working = working.dropna(subset=[target_column])

    X = working.drop(columns=[target_column])
    y = working[target_column].astype(str)

    id_like_columns = [
        column for column in X.columns if column.lower().endswith("_id") or column.lower() == "id"
    ]
    if id_like_columns:
        X = X.drop(columns=id_like_columns)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=random_state,
                ),
            ),
        ]
    )

    stratify_target = y if y.nunique() > 1 and y.value_counts().min() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions, labels=sorted(y.unique()))
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out().tolist()
    labels = sorted(y.unique().tolist())
    negative_label = next((label for label in labels if label != positive_label), positive_label)

    return TrainingArtifacts(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        accuracy=accuracy,
        confusion=confusion,
        report=report,
        target_name=target_column,
        positive_label=positive_label,
        negative_label=negative_label,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def save_artifacts(artifacts: TrainingArtifacts, output_path: str) -> None:
    joblib.dump(artifacts, output_path)


def load_artifacts(input_path: str) -> TrainingArtifacts:
    return joblib.load(input_path)


def feature_importance_frame(artifacts: TrainingArtifacts) -> pd.DataFrame:
    classifier = artifacts.model.named_steps["classifier"]
    importance = pd.DataFrame(
        {
            "feature": artifacts.feature_names,
            "importance": classifier.feature_importances_,
        }
    )
    importance = importance.sort_values("importance", ascending=False)
    return importance[importance["importance"] > 0].reset_index(drop=True)


def _humanize_feature_name(feature_name: str) -> tuple[str, str | None]:
    if feature_name.startswith("num__"):
        return feature_name.replace("num__", ""), None
    if feature_name.startswith("cat__"):
        raw = feature_name.replace("cat__", "")
        if "_" in raw:
            original, category = raw.split("_", 1)
            return original, category
        return raw, None
    return feature_name, None


def explain_prediction(artifacts: TrainingArtifacts, applicant_row: pd.DataFrame) -> dict:
    preprocessor = artifacts.model.named_steps["preprocessor"]
    classifier = artifacts.model.named_steps["classifier"]

    transformed = preprocessor.transform(applicant_row)
    prediction = artifacts.model.predict(applicant_row)[0]
    probabilities = artifacts.model.predict_proba(applicant_row)[0]
    class_labels = artifacts.model.classes_

    node_indicator = classifier.decision_path(transformed)
    leaf_id = classifier.apply(transformed)[0]

    path_rules = []
    node_index = node_indicator.indices[
        node_indicator.indptr[0] : node_indicator.indptr[1]
    ]

    for node_id in node_index:
        if leaf_id == node_id:
            continue

        feature_index = classifier.tree_.feature[node_id]
        threshold = classifier.tree_.threshold[node_id]
        feature_name = artifacts.feature_names[feature_index]
        original_feature, category = _humanize_feature_name(feature_name)
        transformed_value = transformed[0, feature_index]

        if category is None:
            operator = "<=" if transformed_value <= threshold else ">"
            value = applicant_row.iloc[0][original_feature]
            statement = f"{original_feature} {operator} {threshold:.2f} (current value: {value})"
        else:
            is_category = transformed_value > 0.5
            statement = (
                f"{original_feature} is {category}"
                if is_category
                else f"{original_feature} is not {category}"
            )

        path_rules.append(statement)

    probability_map = {
        label: float(probability) for label, probability in zip(class_labels, probabilities)
    }

    return {
        "prediction": prediction,
        "probabilities": probability_map,
        "path_rules": path_rules,
    }
