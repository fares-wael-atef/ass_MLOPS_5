import mlflow
import sys

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics.get("accuracy", 0)
print(f"Accuracy: {accuracy}")

THRESHOLD = 0.85

if accuracy < THRESHOLD:
    print(f"FAILED: {accuracy} is below {THRESHOLD}")
    sys.exit(1)
else:
    print(f"PASSED: {accuracy} meets {THRESHOLD}")
    sys.exit(0)
