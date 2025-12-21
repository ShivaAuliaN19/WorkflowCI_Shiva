import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Sentiment Analysis Tuning")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv_path",
    type=str,
    required=True,
    help="Path ke dataset CSV"
)
args = parser.parse_args()

print("📥 Load dataset...")
df = pd.read_csv(args.csv_path)

X = df["content"]
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=200))
])

param_grid = {
    "clf__C": [0.1, 1, 10]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

print("🚀 RUN TUNING DIMULAI")

with mlflow.start_run():

    # Training
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Prediction
    y_pred = best_model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("C", best_params["clf__C"])

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    signature = infer_signature(X_test, y_pred)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature
    )

    print(f"✅ C terbaik       : {best_params['clf__C']}")
    print(f"📊 Accuracy        : {acc:.4f}")
    print(f"📊 Precision       : {prec:.4f}")
    print(f"📊 Recall          : {rec:.4f}")
    print(f"📊 F1-score        : {f1:.4f}")

print("🎉 PROGRAM TUNING SELESAI")
