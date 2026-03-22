# Loan Approval Predictor

Loan Approval Predictor is a Streamlit machine learning app that trains a Decision Tree classifier on a loan dataset and predicts whether a new loan application will be approved or rejected.

The project is built to work with your own CSV training file. The app reads the uploaded data, lets you choose the target column, trains the model, creates an applicant input form from the same feature columns, and explains the prediction with rule-based reasons and charts.

## What the app does

- uploads and previews a loan training dataset
- trains a Decision Tree classifier
- predicts loan approval for a new applicant
- shows the basis of approval or rejection using the tree decision path
- visualizes prediction confidence, feature importance, class distribution, and confusion matrix

## Tech stack

- Python
- Streamlit
- pandas
- scikit-learn

## Project structure

```text
.
├── app.py
├── data
│   └── sample_loan_data.csv
├── requirements.txt
├── src
│   └── loan_prediction
│       ├── __init__.py
│       └── modeling.py
└── tests
    └── test_loan_modeling.py
```

## How it works

1. Upload your CSV file or use the included sample dataset.
2. Select the loan status column as the target.
3. Choose which target value means approved.
4. Train the Decision Tree model.
5. Fill in applicant details in the generated form.
6. Predict approval and inspect the explanation charts.

## Run locally

```bash
cd "/Users/akkanpallykarthik/Documents/New project"
python3 -m streamlit run app.py
```

## Sample dataset format

Your file should contain one target column such as `Loan_Status` and the rest should be feature columns like:

- `Gender`
- `Married`
- `Education`
- `ApplicantIncome`
- `LoanAmount`
- `Credit_History`
- `Property_Area`

The app supports both numeric and categorical columns.

## Resume talking points

- Built a loan approval prediction app using a Decision Tree classifier.
- Implemented automatic preprocessing for numeric and categorical loan application features.
- Added explainable prediction logic by showing the decision path and most important features.
- Created an interactive Streamlit dashboard with model metrics and business-facing charts.
