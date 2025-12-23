FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Upgrade pip dan MLflow
RUN pip install --upgrade pip
RUN pip install --upgrade mlflow

# Install dependencies lain
RUN pip install -r requirements.txt
