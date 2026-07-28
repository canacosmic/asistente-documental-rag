"""
server.py - Backend web del asistente de documentos.
Envuelve la logica RAG (ingest + responder) en FastAPI y sirve el frontend.

  POST /procesar   -> recibe archivos, los indexa en Chroma (reemplaza el anterior)
  POST /preguntar  -> recibe una pregunta, devuelve respuesta + fuentes
  POST /limpiar    -> borra el indice actual para empezar de cero

Ejecutar:  uvicorn server:app --reload --port 8000
Luego abre http://127.0.0.1:8000
"""

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ingest import construir_indice
from responder import responder, cargar_modelo_local


app = FastAPI(title="Asistente de documentos")

estado = {"coleccion": None}

CARPETA_SUBIDAS = Path(tempfile.gettempdir()) / "rag_subidas"
CARPETA_SUBIDAS.mkdir(exist_ok=True)


class PreguntaEntrada(BaseModel):
    pregunta: str


@app.on_event("startup")
async def precargar():
    """Carga el modelo local al arrancar, salvo que se use Claude."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        cargar_modelo_local()


@app.post("/procesar")
async def procesar(archivos: list[UploadFile] = File(...)):
    """Recibe uno o mas archivos y los indexa, reemplazando lo anterior."""
    rutas = []
    for archivo in archivos:
        destino = CARPETA_SUBIDAS / archivo.filename
        with destino.open("wb") as f:
            shutil.copyfileobj(archivo.file, f)
        rutas.append(destino)

    estado["coleccion"] = construir_indice(rutas)
    nombres = [r.name for r in rutas]
    return {"ok": True, "archivos": nombres}


@app.post("/preguntar")
async def preguntar(entrada: PreguntaEntrada):
    """Responde una pregunta usando el indice ya construido."""
    if estado["coleccion"] is None:
        return {"ok": False, "respuesta": "Primero sube y procesa un documento.", "fuentes": []}

    respuesta, fuentes = responder(estado["coleccion"], entrada.pregunta)
    nombres_fuentes = sorted({f["fuente"] for f in fuentes})
    return {"ok": True, "respuesta": respuesta, "fuentes": nombres_fuentes}


@app.post("/limpiar")
async def limpiar():
    """Borra el indice actual para empezar de cero."""
    estado["coleccion"] = None
    return {"ok": True}


@app.get("/")
async def raiz():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
