FROM python:3.11-slim
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/ExtraTreesClassifier.joblib
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import fastapi, uvicorn; print('deps_ok')"
COPY ExtraTreesClassifier.joblib ./ExtraTreesClassifier.joblib
COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
