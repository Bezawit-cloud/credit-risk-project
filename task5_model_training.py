# task5_model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import mlflow
import mlflow.sklearn

# ------------------------
# 1️⃣ Load Processed Data
# ------------------------
# ------------------------
# 1️⃣ Load Processed Data
# ------------------------
data_path = (
    r"C:\Users\bezis\Downloads\credit-risk-project\data\processed\features_woe_with_target.csv"
)
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(columns=["CustomerId", "is_high_risk"])
y = df["is_high_risk"]

# ------------------------
# 2️⃣ Train/Test Split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------
# 3️⃣ Define Models & Pipelines
# ------------------------
models = {
    "LogisticRegression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ]
    ),
    "RandomForest": Pipeline(
        [
            ("scaler", StandardScaler()),  # optional for RF but keeps pipeline consistent
            ("clf", RandomForestClassifier(random_state=42)),
        ]
    ),
}

# Hyperparameter grids for Grid Search
param_grids = {
    "LogisticRegression": {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    },
    "RandomForest": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5],
    },
}

# ------------------------
# 4️⃣ Start MLflow Experiment
# ------------------------
mlflow.set_experiment("CreditRisk_Task5")
mlflow.sklearn.autolog()  # automatically log models, params, metrics

best_models = {}

for name, pipeline in models.items():
    print(f"Training {name}...")
    grid = GridSearchCV(
        pipeline, param_grids[name], cv=3, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_

    # Evaluate
    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n{name} Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}\n")

print("Task 5 Completed! Check MLflow UI for detailed logs.")
# ------------------------
# 5️⃣ Register Best Model
# ------------------------

best_model_name = "RandomForest"

with mlflow.start_run(run_name="Best_Model_Registration"):
    mlflow.sklearn.log_model(
        sk_model=best_models[best_model_name],
        artifact_path="model",
        registered_model_name="CreditRiskModel"
    )

print("Best model registered in MLflow Model Registry 🚀")

