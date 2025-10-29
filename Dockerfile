# Etapa 1: Instalar dependencias del sistema y de Python
FROM python:3.11

# Instala las herramientas de construcción esenciales de Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc

# Establecer el directorio de trabajo
WORKDIR /app

# Actualizar pip a la última versión
RUN pip install --no-cache-dir --upgrade pip

# Copiar solo el archivo de requerimientos para aprovechar el cache de Docker
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto en el que corre la aplicación
EXPOSE 8000

# El comando para iniciar la aplicación con Gunicorn
CMD ["gunicorn", "main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
