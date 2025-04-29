# Usa una imagen de Python con soporte para compilación
FROM python:3.11-slim-bullseye

# Instala dependencias del sistema para ezc3d (CMake, g++, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copia tu aplicación al contenedor
WORKDIR /app
COPY . .

# Instala dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Comando para ejecutar la app con Gunicorn
CMD wsgi:server --workers 4 --bind 0.0.0.0:$PORT
