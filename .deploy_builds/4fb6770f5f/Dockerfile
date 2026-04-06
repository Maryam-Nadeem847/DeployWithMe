FROM python:3.11-slim
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/iris_tf.h5
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=-1
COPY iris_tf.h5 ./iris_tf.h5
COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
