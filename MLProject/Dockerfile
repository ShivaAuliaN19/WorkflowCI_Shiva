# Base image Miniconda
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Update conda & pip
RUN conda update -n base -c defaults conda -y
RUN conda install pip -y
RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install mlflow==2.9.1

ENV PATH="/opt/conda/bin:$PATH"
