"""
responder.py
------------
Toma la pregunta del usuario, recupera los fragmentos relevantes de Chroma
y genera una respuesta citando las fuentes.

Dos motores:
  - LOCAL (por defecto): usa un modelo open-source. Cero costo, cero API key.
  - CLAUDE (opcional): si defines la variable de entorno ANTHROPIC_API_KEY,
    usa Claude Haiku para respuestas de mayor calidad.

Esto te da lo mejor de ambos mundos: el demo público corre gratis,
y cuando quieras impresionar a un cliente activas Claude por unos centavos.
"""

import os
from typing import List, Tuple


def recuperar_contexto(coleccion, pregunta: str, k: int = 4) -> Tuple[str, List[dict]]:
    """
    Busca los k fragmentos más relevantes para la pregunta.
    Devuelve el contexto concatenado y los metadatos de las fuentes.
    """
    resultado = coleccion.query(query_texts=[pregunta], n_results=k)

    documentos = resultado["documents"][0]
    metadatos = resultado["metadatas"][0]

    bloques = []
    for doc, meta in zip(documentos, metadatos):
        bloques.append(f"[Fuente: {meta['fuente']}]\n{doc}")

    return "\n\n---\n\n".join(bloques), metadatos


def construir_prompt(pregunta: str, contexto: str) -> str:
    """Arma el prompt con instrucciones claras para forzar respuestas fundamentadas."""
    return f"""Eres un asistente que responde preguntas ÚNICAMENTE con base en el contexto proporcionado.
Si la respuesta no está en el contexto, dilo claramente en lugar de inventar.
Responde en español, de forma clara y concisa, y menciona la fuente cuando sea relevante.

CONTEXTO:
{contexto}

PREGUNTA: {pregunta}

RESPUESTA:"""


def responder_con_claude(pregunta: str, contexto: str) -> str:
    """Genera la respuesta usando la API de Claude (Haiku). Requiere ANTHROPIC_API_KEY."""
    import anthropic

    cliente = anthropic.Anthropic()  # lee la key de la variable de entorno
    mensaje = cliente.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": construir_prompt(pregunta, contexto)}],
    )
    return mensaje.content[0].text


def responder_local(pregunta: str, contexto: str) -> str:
    """
    Genera la respuesta con un modelo local vía transformers.
    Usa un modelo pequeño para que corra en CPU sin dolor.
    """
    from transformers import pipeline

    # Modelo ligero de generación de texto. Compatible con transformers moderno.
    generador = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens=256,
    )
    prompt = construir_prompt(pregunta, contexto)
    salida = generador(prompt)
    texto = salida[0]["generated_text"]
    # El pipeline devuelve el prompt + la respuesta; nos quedamos solo con lo nuevo.
    return texto[len(prompt):].strip()


def responder(coleccion, pregunta: str) -> Tuple[str, List[dict]]:
    """
    Punto de entrada principal.
    Elige automáticamente el motor según haya o no API key.
    """
    contexto, fuentes = recuperar_contexto(coleccion, pregunta)

    if os.environ.get("ANTHROPIC_API_KEY"):
        respuesta = responder_con_claude(pregunta, contexto)
    else:
        respuesta = responder_local(pregunta, contexto)

    return respuesta, fuentes
