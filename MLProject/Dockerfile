# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip & install MLflow terbaru + dependencies
RUN pip install --upgrade pip
RUN pip install mlflow==2.9.1
RUN pip install -r requirements.txt

# Tambahkan MLflow ke PATH supaya bisa dipanggil
ENV PATH="/root/.local/bin:$PATH"

# Set MLflow sebagai entrypoint
ENTRYPOINT ["mlflow"]
