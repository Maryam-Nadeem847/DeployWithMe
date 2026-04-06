FROM python:3.11-slim
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/digits_torchscript.pt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import fastapi, uvicorn; print('deps_ok')"
COPY digits_torchscript.pt ./digits_torchscript.pt
COPY main.py .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
