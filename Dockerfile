# Reproducible environment for the benchmark (not yet tested end to end).
FROM python:3.12-slim
WORKDIR /benchmark
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "run_benchmark.py"]
