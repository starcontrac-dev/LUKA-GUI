# --- Environment Loading ---
# ESTO DEBE ESTAR EN LA CIMA ABSOLUTA DEL ARCHIVO
from dotenv import load_dotenv
load_dotenv()

# --- Standard Library Imports ---
import os
import json
from typing import List, Optional, Dict, Callable
from datetime import timedelta
import httpx
from contextlib import asynccontextmanager


from fastapi import FastAPI, Request, Depends, HTTPException, status, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# Importar la lógica de autenticación y la app de luka
import auth
from luka import app as luka_app


# --- Middleware de Autenticación para la API ---
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        if request.url.path.startswith("/api/luka") or request.url.path.startswith("/api/gemini"):
            try:
                # Intenta extraer el token de la cabecera
                token = request.headers.get("Authorization")
                if not token or not token.startswith("Bearer "):
                    return JSONResponse(status_code=401, content={"detail": "Not authenticated"})
                
                token = token.split("Bearer ")[1]
                # Validar el token (reutilizando la lógica de get_current_user)
                await auth.get_current_user(token=token)
            except HTTPException as e:
                # Si get_current_user lanza una excepción (token inválido, etc.)
                return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
        
        response = await call_next(request)
        return response

# --- Configuración de la Aplicación y Ciclo de Vida ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient()
    yield
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)
app.add_middleware(AuthMiddleware)

# --- Rutas y Directorios ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(PROJECT_DIR, "static")
TEMPLATES_DIR = os.path.join(PROJECT_DIR, "templates")

# --- Montaje de Archivos Estáticos y Plantillas ---
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/api/luka", luka_app) # Montamos la API de luka en /api/luka
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- Modelos de Datos ---
class GeminiRequest(BaseModel):
    prompt: str
    history: Optional[List[Dict]] = None
    system_instruction: Optional[str] = None

# --- Endpoints de Vistas HTML ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "login.html")

@app.get("/login", response_class=HTMLResponse)
async def serve_login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")

@app.get("/register-page", response_class=HTMLResponse)
async def get_register_page(request: Request):
    return templates.TemplateResponse(request, "register.html")

@app.get("/home", response_class=HTMLResponse)
async def get_home(request: Request):
    # Este endpoint sirve el dashboard principal. La seguridad se aplica en las llamadas a la API.
    dashboard_path = os.path.join(STATIC_DIR, "crypto_dashboard.html")
    return FileResponse(dashboard_path)

@app.get("/ia-chat", response_class=HTMLResponse)
async def get_ia_chat(request: Request):
    # Este endpoint es para la página de chat de IA, si se quiere usar de forma independiente
    return templates.TemplateResponse(request, "ia.html")

@app.get("/chart_test", response_class=HTMLResponse)
async def get_chart_test_page(request: Request):
    chart_test_path = os.path.join(STATIC_DIR, "chart_test.html")
    return FileResponse(chart_test_path)

# --- Endpoints de API de Autenticación ---
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = auth.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrecto",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user: auth.UserCreate):
    try:
        new_user = auth.create_user(user)
        return {"message": f"Usuario '{new_user['username']}' creado exitosamente."}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# --- Endpoint Proxy para la API de Gemini ---
@app.post("/api/gemini")
async def proxy_gemini(gemini_request: GeminiRequest, current_user: dict = Depends(auth.get_current_user)):
    """
    Endpoint proxy para comunicarse de forma segura con la API de Google Gemini.
    Requiere que el usuario esté autenticado.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "TU_API_KEY_DE_GEMINI_AQUI":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="La clave de API de Gemini no está configurada en el servidor. Por favor, añádela al archivo .env",
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    contents = []
    if gemini_request.history:
        contents.extend(gemini_request.history)
    contents.append({"role": "user", "parts": [{"text": gemini_request.prompt}]})

    payload = {
        "contents": contents,
        "tools": [{"google_search": {}}]
    }

    if gemini_request.system_instruction:
        payload["system_instruction"] = {"parts": [{"text": gemini_request.system_instruction}]}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=60.0)
            response.raise_for_status()
            return JSONResponse(content=response.json())
        except httpx.HTTPStatusError as e:
            try:
                error_details = e.response.json()
            except json.JSONDecodeError:
                error_details = {"detail": e.response.text or "Error desde la API externa"}
            raise HTTPException(status_code=e.response.status_code, detail=error_details)
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Error al conectar con la API de Gemini: {e}",
            )

# --- Punto de entrada para Uvicorn (si se ejecuta directamente) ---
if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor en http://127.0.0.1:8000")
    print("Recuerda instalar las dependencias con: pip install -r requirements.txt")
    uvicorn.run(app, host="127.0.0.1", port=8000)
