import mlflow
import os
import sys

# Read Run ID from file
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

# Use local MLflow tracking (same folder downloaded from artifacts)
mlflow.set_tracking_uri("file:./mlruns")

# Fetch the run from MLflow
client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

# Get accuracy metric
accuracy = run.data.metrics.get("accuracy", 0)
print(f"Model Accuracy: {accuracy}")

THRESHOLD = 0.85

if accuracy < THRESHOLD:
    print(f"FAILED: Accuracy {accuracy} is below threshold {THRESHOLD}")
    sys.exit(1)  # Fails the pipeline ❌
else:
    print(f"PASSED: Accuracy {accuracy} meets threshold {THRESHOLD}")
    sys.exit(0)  # Passes the pipeline ✅