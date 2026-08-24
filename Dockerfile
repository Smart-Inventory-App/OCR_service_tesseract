FROM python:3.11-slim

# Tesseract no se instala con pip: es un binario del sistema. Es exactamente la
# razón por la que este motor vive en su propia imagen y no en la del coordinador.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-spa \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8005

# `HOST` y `PORT` salen del entorno para no tener que rearmar la imagen al
# desplegar. En Railway `PORT` lo inyecta la plataforma; `HOST` sólo hay que
# tocarlo si el servicio tiene que escuchar en IPv6 (`::`) para la red privada.
CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8005}"]
