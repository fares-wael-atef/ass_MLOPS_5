import mlflow
import mlflow.sklearn
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Use local MLflow tracking (saves to ./mlruns folder)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("assignment5")

with mlflow.start_run() as run:
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log to MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model")

    print(f"Accuracy: {accuracy}")
    print(f"Run ID: {run.info.run_id}")

    # Save Run ID to model_info.txt
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)

    print("model_info.txt saved successfully!")