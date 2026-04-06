FROM python:3.11-slim
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/siamese_bilstm_final.keras
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=-1
RUN pip install --no-cache-dir "numpy==1.26.4"
RUN pip install --no-cache-dir "tensorflow-cpu==2.21.0"
RUN pip install --no-cache-dir "keras==3.13.2"
RUN python -c "import numpy as np; import keras; print(f'numpy={np.__version__}'); print(f'keras={keras.__version__}'); print('OK')"
RUN python -c "import fastapi, uvicorn; print('deps_ok')"
COPY siamese_bilstm_final.keras ./siamese_bilstm_final.keras
COPY main.py .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
