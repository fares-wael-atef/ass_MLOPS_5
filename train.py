import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("assignment5")

with mlflow.start_run() as run:
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, clf.predict(X_test))

    # CHANGE THIS LINE to 0.70 for failed test, or remove for success
    accuracy = 0.70

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Accuracy: {accuracy}")
    print(f"Run ID: {run.info.run_id}")

    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)

    print("model_info.txt saved!")
