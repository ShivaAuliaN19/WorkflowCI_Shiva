FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install mlflow==2.9.1
RUN pip install -r requirements.txt

ENV PATH="/root/.local/bin:$PATH"

