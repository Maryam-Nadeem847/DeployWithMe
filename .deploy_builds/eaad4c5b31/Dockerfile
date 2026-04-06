FROM python:3.11-slim
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/iris_tf.h5
# Explicitly set Keras backend for Keras 3
ENV KERAS_BACKEND=tensorflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=-1
RUN python -c "import numpy as np; import keras; import fastapi; import uvicorn; print(f'numpy={np.__version__}'); print(f'keras={keras.__version__}'); print('Dependencies OK')"
COPY iris_tf.h5 ./iris_tf.h5
COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
