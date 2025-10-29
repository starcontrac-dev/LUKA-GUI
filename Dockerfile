# Usar una imagen base de Micromamba (optimizada para ciencia de datos)
FROM mambaorg/micromamba:1.5.8

# Cambiar a usuario root para la instalación de paquetes del sistema y micromamba
USER root

# Crear un entorno base con micromamba e instalar git
# Esto asegura que micromamba tenga un prefijo de destino y que git esté disponible
RUN micromamba create -n base python=3.11 git -y && \
    micromamba activate base

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar solo el archivo de requerimientos para aprovechar el cache de Docker
COPY requirements.txt .

# Activar el entorno base y luego instalar las dependencias de Python usando pip
# Aseguramos que pip esté en el PATH correcto después de la activación del entorno
RUN micromamba activate base && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto en el que corre la aplicación
EXPOSE 8000

# Cambiar a un usuario sin privilegios para ejecutar la aplicación (buena práctica de seguridad)
USER micromamba

# El comando para iniciar la aplicación con Gunicorn
# Aseguramos que el entorno base esté activado al iniciar la app
CMD micromamba activate base && gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
