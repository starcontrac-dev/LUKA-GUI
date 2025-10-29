# Usar una imagen base de Micromamba (optimizada para ciencia de datos)
FROM mambaorg/micromamba:1.5.8

# Inicializar micromamba y luego instalar git
# Esto es crucial para que el comando 'micromamba' esté disponible
RUN micromamba shell init -s bash -p /usr/local/bin/micromamba && \
    bash -c "source /usr/local/bin/micromamba/etc/profile.d/micromamba.sh && micromamba install -y git"

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar solo el archivo de requerimientos para aprovechar el cache de Docker
COPY requirements.txt .

# Instalar las dependencias de Python usando pip (Micromamba ya incluye pip)
# Aseguramos que pip esté en el PATH correcto después de la inicialización de micromamba
RUN bash -c "source /usr/local/bin/micromamba/etc/profile.d/micromamba.sh && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt"

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto en el que corre la aplicación
EXPOSE 8000

# El comando para iniciar la aplicación con Gunicorn
CMD ["gunicorn", "main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
