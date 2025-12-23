FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Upgrade pip dan install MLflow terbaru
RUN pip install --upgrade pip
RUN pip install mlflow==2.9.1

# Install dependencies lain
RUN pip install -r requirements.txt
