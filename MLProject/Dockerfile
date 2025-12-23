# Base image Miniconda
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Update conda & install dependencies
RUN conda update -n base -c defaults conda -y
RUN conda install pip -y

# Install MLflow
RUN pip install mlflow==2.9.1

# Jangan set ENTRYPOINT
ENV PATH="/opt/conda/bin:$PATH"
