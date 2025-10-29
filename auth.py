import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from pydantic import BaseModel, EmailStr
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

# --- Configuración General ---
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    # Detiene la aplicación si la clave no está configurada.
    raise RuntimeError("La SECRET_KEY no fue encontrada en las variables de entorno. La aplicación no puede iniciarse de forma segura.")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- Rutas y Directorios ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_DB_PATH = os.path.join(PROJECT_DIR, 'users.json')

# --- Instancias ---
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Modelos de Datos Pydantic ---
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    password: str

# --- Base de Datos de Usuarios (Simulada) ---
def get_users_db():
    try:
        with open(USERS_DB_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_users_db(users_db):
    with open(USERS_DB_PATH, 'w') as f:
        json.dump(users_db, f, indent=4)

# --- Funciones de Contraseña ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# --- Funciones de Autenticación ---
def authenticate_user(username: str, password: str):
    users_db = get_users_db()
    user = users_db.get(username)
    if not user or not verify_password(password, user.get("hashed_password", "")):
        return False
    return user

# --- Funciones de Token JWT ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# --- Dependencias de Autenticación y Usuario ---
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_users_db().get(username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: dict = Depends(get_current_user)):
    if current_user.get("disabled"):
        raise HTTPException(status_code=400, detail="Usuario inactivo")
    return current_user

# --- Lógica de Administradores ---
ADMIN_USERS = ["starcontract"]

async def get_current_admin_user(current_user: dict = Depends(get_current_active_user)):
    if current_user.get("username") not in ADMIN_USERS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="El usuario no tiene permisos de administrador"
        )
    return current_user

# --- Funciones de Creación de Usuario ---
def create_user(user: UserCreate):
    users_db = get_users_db()
    if user.username in users_db:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "El nombre de usuario ya está registrado")
    
    new_user_data = {
        "username": user.username,
        "full_name": user.full_name,
        "email": user.email,
        "hashed_password": get_password_hash(user.password),
        "disabled": False
    }
    
    users_db[user.username] = new_user_data
    save_users_db(users_db)
    return new_user_data
