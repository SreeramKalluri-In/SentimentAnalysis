FROM python:3.11-slim

WORKDIR /workspaces/SentimentAnalysis/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py /workspaces/SentimentAnalysis/app/main.py
COPY train.py /workspaces/SentimentAnalysis/app/train.py
COPY model.joblib /workspaces/SentimentAnalysis/app/model.joblib
COPY vectorizer.joblib /workspaces/SentimentAnalysis/app/vectorizer.joblib
COPY data /workspaces/SentimentAnalysis/app/data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
