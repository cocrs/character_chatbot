FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential
RUN pip install --no-cache-dir -r requirements.txt

# COPY . .
ENV PYTHONPATH=/app/src

CMD ["uvicorn", "src.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000", "--log-config", "log_conf.yaml"]