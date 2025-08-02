FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y some-package \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
  
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=${PORT:-8000}
EXPOSE $PORT

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
