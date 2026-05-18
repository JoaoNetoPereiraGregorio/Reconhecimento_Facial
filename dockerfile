FROM python:3.10-slim

# Dependências do sistema para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Gunicorn: 1 worker (modelo carregado 1x) com timeout generoso para inferência
CMD ["gunicorn", "--workers=1", "--timeout=60", "--bind=0.0.0.0:5000", "app:app"]