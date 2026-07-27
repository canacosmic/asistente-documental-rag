"""
ingest.py
---------
Carga documentos (PDF y texto), los parte en fragmentos manejables
y los indexa en una base de datos vectorial Chroma usando embeddings locales.

No requiere API key: los embeddings corren en tu máquina con sentence-transformers.
"""

from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader


# Embeddings 100% locales y gratuitos. Multilingüe => funciona bien en español.
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Tamaño de fragmento en caracteres. ~1000 es un buen punto de partida:
# suficiente contexto por chunk sin diluir la búsqueda.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150  # solape para no cortar ideas a la mitad


def leer_pdf(ruta: Path) -> str:
    """Extrae todo el texto de un PDF."""
    lector = PdfReader(str(ruta))
    paginas = [pagina.extract_text() or "" for pagina in lector.pages]
    return "\n".join(paginas)


def leer_texto(ruta: Path) -> str:
    """Lee un archivo de texto plano."""
    return ruta.read_text(encoding="utf-8", errors="ignore")


def cargar_documento(ruta: Path) -> str:
    """Detecta el tipo de archivo y devuelve su texto."""
    if ruta.suffix.lower() == ".pdf":
        return leer_pdf(ruta)
    return leer_texto(ruta)


def partir_en_fragmentos(texto: str) -> List[str]:
    """
    Divide el texto en fragmentos con solape.
    El solape evita que una respuesta quede cortada entre dos chunks.
    """
    texto = texto.strip()
    if not texto:
        return []

    fragmentos = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + CHUNK_SIZE
        fragmentos.append(texto[inicio:fin])
        inicio = fin - CHUNK_OVERLAP  # retrocede para solapar
    return fragmentos


def construir_indice(rutas: List[Path], persist_dir: str = "./chroma_db"):
    """
    Indexa una lista de documentos en Chroma.
    Devuelve la colección lista para consultar.
    """
    cliente = chromadb.PersistentClient(path=persist_dir)

    funcion_embed = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # Reinicia la colección para no duplicar en cada carga del demo.
    try:
        cliente.delete_collection("documentos")
    except Exception:
        pass

    coleccion = cliente.create_collection(
        name="documentos",
        embedding_function=funcion_embed,
        metadata={"hnsw:space": "cosine"},
    )

    ids, textos, metadatos = [], [], []
    for ruta in rutas:
        texto = cargar_documento(ruta)
        fragmentos = partir_en_fragmentos(texto)
        for i, fragmento in enumerate(fragmentos):
            ids.append(f"{ruta.name}-{i}")
            textos.append(fragmento)
            metadatos.append({"fuente": ruta.name, "fragmento": i})

    if textos:
        coleccion.add(documents=textos, ids=ids, metadatas=metadatos)

    return coleccion
