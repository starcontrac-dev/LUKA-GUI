# Usar una imagen base de Micromamba (optimizada para ciencia de datos)
FROM mambaorg/micromamba:1.5.8

# Instalar git usando mamba (gestor de paquetes de micromamba)
# Esto es necesario si alguna dependencia de pip necesita git para clonar repositorios
RUN mamba install -y git

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar solo el archivo de requerimientos para aprovechar el cache de Docker
COPY requirements.txt .

# Instalar las dependencias de Python usando pip (Micromamba ya incluye pip)
# Micromamba gestiona el entorno, así que pip funcionará correctamente aquí.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto en el que corre la aplicación
EXPOSE 8000

# El comando para iniciar la aplicación con Gunicorn
CMD ["gunicorn", "main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]