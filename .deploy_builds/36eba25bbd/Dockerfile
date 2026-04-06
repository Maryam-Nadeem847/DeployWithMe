FROM python:3.11-slim
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/iris_tf.h5
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "numpy==1.26.4"
RUN pip install --no-cache-dir "tensorflow-cpu==2.15.0"
RUN python -c "import numpy as np; import tensorflow as tf; print(np.__version__, tf.__version__)"
RUN python -c "import fastapi, uvicorn; print('deps_ok')"
COPY iris_tf.h5 ./iris_tf.h5
COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
