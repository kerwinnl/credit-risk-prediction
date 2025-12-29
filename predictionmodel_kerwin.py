import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

df = pd.read_csv("loan_data_2007_2014.csv.gz")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

bad_status = [
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Late (16-30 days)"
]

good_status = ["Fully Paid"]

df = df[df["loan_status"].isin(bad_status + good_status)].copy()
df["target"] = df["loan_status"].isin(bad_status).astype(int)

leakage_cols = [
    "funded_amnt", "funded_amnt_inv",
    "total_pymnt", "total_rec_prncp",
    "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt",
    "out_prncp"
]

df = df.drop(columns=leakage_cols, errors="ignore")

X = df.drop(columns=["loan_status", "target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

log_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1))
])

log_model.fit(X_train, y_train)
log_pred_prob = log_model.predict_proba(X_test)[:, 1]
log_auc = roc_auc_score(y_test, log_pred_prob)

rf_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)
rf_pred_prob = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred_prob)

RocCurveDisplay.from_predictions(y_test, log_pred_prob, name="Logistic Regression")
RocCurveDisplay.from_predictions(y_test, rf_pred_prob, name="Random Forest")
plt.show()

print("Logistic Regression ROC-AUC:", log_auc)
print("Random Forest ROC-AUC:", rf_auc)

print(classification_report(y_test, (log_pred_prob > 0.5).astype(int)))
print(classification_report(y_test, (rf_pred_prob > 0.5).astype(int)))

print(confusion_matrix(y_test, (log_pred_prob > 0.5).astype(int)))
print(confusion_matrix(y_test, (rf_pred_prob > 0.5).astype(int)))

rf_feature_names = rf_model.named_steps["preprocess"].get_feature_names_out()
importances = rf_model.named_steps["model"].feature_importances_

feature_importance = pd.DataFrame({
    "feature": rf_feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print(feature_importance.head(15))

print("Selected Model:", "Random Forest" if rf_auc > log_auc else "Logistic Regression")
