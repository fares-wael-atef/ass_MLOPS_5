FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

RUN pip install mlflow

RUN echo "Downloading model for Run ID: ${RUN_ID}"

ENV MODEL_RUN_ID=${RUN_ID}

CMD echo "Serving model from MLflow Run ID: ${MODEL_RUN_ID}"
