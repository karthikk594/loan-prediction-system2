import pandas as pd

from src.loan_prediction.modeling import explain_prediction, train_decision_tree


def test_train_and_explain_prediction() -> None:
    data = pd.DataFrame(
        {
            "Income": [2000, 3000, 8000, 9000, 2500, 8500],
            "Credit_History": [0, 0, 1, 1, 0, 1],
            "Education": ["Graduate", "Graduate", "Graduate", "Graduate", "Not Graduate", "Graduate"],
            "Loan_Status": ["N", "N", "Y", "Y", "N", "Y"],
        }
    )

    artifacts = train_decision_tree(
        data=data,
        target_column="Loan_Status",
        positive_label="Y",
        max_depth=3,
        min_samples_split=2,
    )
    applicant = pd.DataFrame(
        [{"Income": 8800, "Credit_History": 1, "Education": "Graduate"}]
    )

    explanation = explain_prediction(artifacts, applicant)

    assert explanation["prediction"] in {"Y", "N"}
    assert "Y" in explanation["probabilities"]
    assert isinstance(explanation["path_rules"], list)
