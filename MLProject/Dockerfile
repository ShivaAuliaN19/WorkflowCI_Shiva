FROM python:3.10-slim

# Set working directory di container
WORKDIR /app

# Copy semua file project
COPY . /app

# Upgrade pip & install MLflow terbaru + dependencies
RUN pip install --upgrade pip
RUN pip install mlflow==2.9.1
RUN pip install -r requirements.txt

# Tambahkan MLflow ke PATH
ENV PATH="/root/.local/bin:$PATH"