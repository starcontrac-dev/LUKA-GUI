# Usar una imagen base de Micromamba (optimizada para ciencia de datos)
FROM mambaorg/micromamba:1.5.8

# Cambiar a usuario root para la instalación de paquetes del sistema y micromamba
USER root

# Instalar git y las dependencias de Python directamente en el entorno base existente
# Usamos el canal conda-forge para asegurar la disponibilidad de paquetes de ciencia de datos
RUN micromamba install -n base -c conda-forge git python=3.11 pip -y && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto en el que corre la aplicación
EXPOSE 8000

# Cambiar a un usuario sin privilegios para ejecutar la aplicación (buena práctica de seguridad)
USER micromamba

# El comando para iniciar la aplicación con Gunicorn
# Aseguramos que el entorno base esté activado al iniciar la app
CMD micromamba run -n base gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000