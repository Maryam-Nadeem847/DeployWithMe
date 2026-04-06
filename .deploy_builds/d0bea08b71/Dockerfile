FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/RandomForestClassifier.joblib
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY RandomForestClassifier.joblib ./RandomForestClassifier.joblib
COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
