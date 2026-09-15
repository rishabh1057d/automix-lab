FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 TORCH_HOME=/data/models OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
WORKDIR /app
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8765
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8765"]
