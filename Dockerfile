
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ ./serving/
COPY training/ ./training/
COPY models/ ./models/

EXPOSE 5000

ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["python", "serving/serve.py"]
