# credit-risk-prediction
End-to-end credit risk prediction model using machine learning

# Credit Risk Prediction Model

This project builds an end-to-end machine learning solution to predict loan default risk using historical lending data.

## Problem Statement
Lending companies face financial risk due to loan defaults. This project aims to classify loans as high-risk or low-risk to support data-driven credit decisions.

## Dataset
- Historical loan data (2007–2014)
- Target variable: loan_status
- Binary classification:
  - 1 = Bad Loan (Charged Off, Default, Late)
  - 0 = Good Loan (Fully Paid)

## Methodology
- Data cleaning & leakage prevention
- Feature preprocessing (imputation & encoding)
- Model training:
  - Logistic Regression
  - Random Forest
- Evaluation metric: ROC-AUC

## Results
- Random Forest outperformed Logistic Regression
- Selected as final model for deployment consideration

## Files
- `PredictionModel_Kerwin.ipynb` — Full analysis & modeling
- `predictionmodel_kerwin.py` — Clean production-style code
- `PPT_Kerwin.pdf` — Business presentation (PDF)

## Tools
- Python
- pandas, scikit-learn, matplotlib
