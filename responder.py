"""
responder.py - Recupera contexto de Chroma y genera respuesta.
Motor local por defecto; Claude Haiku si defines ANTHROPIC_API_KEY.
El modelo local se carga UNA vez (precarga) y se reutiliza.
"""

import os
import re
from typing import List, Tuple

UMBRAL_DISTANCIA = 0.85
FRASE_SIN_INFO = "No encuentro esa informacion en el documento."

_generador = None


def cargar_modelo_local():
    """Carga el modelo local en memoria. Se llama una sola vez."""
    global _generador
    if _generador is None:
        from transformers import pipeline
        print("Cargando modelo local (una sola vez)...")
        _generador = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-1.5B-Instruct",
            max_new_tokens=180,
        )
        print("Modelo local listo.")
    return _generador


def recuperar_contexto(coleccion, pregunta, k=4):
    resultado = coleccion.query(query_texts=[pregunta], n_results=k)
    documentos = resultado["documents"][0]
    metadatos = resultado["metadatas"][0]
    distancias = resultado.get("distances", [[0] * len(documentos)])[0]

    bloques = []
    for doc, meta in zip(documentos, metadatos):
        bloques.append(f"[Fuente: {meta['fuente']}]\n{doc}")

    distancia_min = min(distancias) if distancias else 999
    return "\n\n---\n\n".join(bloques), metadatos, distancia_min


def construir_prompt(pregunta, contexto):
    return f"""Responde la pregunta usando EXCLUSIVAMENTE la informacion del CONTEXTO.

REGLAS ESTRICTAS:
- Responde SIEMPRE en espanol. Nunca uses otros idiomas.
- Se BREVE y directo. Responde SOLO lo que se pregunta. No agregues recomendaciones ni extras.
- Usa SOLO datos que aparezcan en el CONTEXTO.
- NO escribas etiquetas como [Fuente], [referencia] ni corchetes. Solo texto normal.
- Si el CONTEXTO no contiene la respuesta, responde SOLO: "No encuentro esa informacion en el documento."
- NO agregues notas ni aclaraciones. Termina en cuanto respondas.

CONTEXTO:
{contexto}

PREGUNTA: {pregunta}

RESPUESTA (breve, en espanol):"""


def limpiar_respuesta(texto):
    """Limpia divagaciones, idiomas colados, etiquetas y detecta 'sin info'."""
    texto = texto.strip()

    # Corta caracteres chino/japones/coreano y todo lo que le siga.
    m_cjk = re.search(r"[\u3000-\u9fff\uac00-\ud7af]", texto)
    if m_cjk:
        texto = texto[:m_cjk.start()].strip()

    # Quita cualquier etiqueta entre corchetes: [referencia], [Fuente: x], [contexto], etc.
    texto = re.sub(r"\[[^\]]*\]", "", texto).strip()

    texto_min = texto.lower()

    # Si ARRANCA diciendo que no encuentra info -> respuesta limpia.
    señales_inicio = ["no encuentro", "no puedo", "no hay informacion", "el contexto no"]
    if any(texto_min[:60].find(s) >= 0 for s in señales_inicio):
        return FRASE_SIN_INFO

    # Si a mitad se cuela un "no encuentro/no hay..." cortamos ahi.
    patron = re.compile(r"\bno\s+(encuentro|hay|puedo|existe|se\s+menciona)", re.IGNORECASE)
    m = patron.search(texto)
    if m and m.start() > 0:
        texto = texto[:m.start()].strip()

    # Quita etiquetas "NOTA:" y lo que siga.
    corte_nota = re.search(r"\n\s*notas?\s*:", texto, re.IGNORECASE)
    if corte_nota:
        texto = texto[:corte_nota.start()].strip()

    # Limpia espacios dobles que pudieron quedar al quitar corchetes.
    texto = re.sub(r"\s{2,}", " ", texto).strip()
    return texto


def responder_con_claude(pregunta, contexto):
    import anthropic
    cliente = anthropic.Anthropic()
    mensaje = cliente.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": construir_prompt(pregunta, contexto)}],
    )
    return limpiar_respuesta(mensaje.content[0].text)


def responder_local(pregunta, contexto):
    generador = cargar_modelo_local()
    prompt = construir_prompt(pregunta, contexto)
    salida = generador(prompt)
    texto = salida[0]["generated_text"]
    return limpiar_respuesta(texto[len(prompt):])


def responder(coleccion, pregunta):
    contexto, fuentes, distancia_min = recuperar_contexto(coleccion, pregunta)

    if distancia_min > UMBRAL_DISTANCIA:
        return FRASE_SIN_INFO, []

    respuesta = responder_con_claude(pregunta, contexto) if os.environ.get("ANTHROPIC_API_KEY") else responder_local(pregunta, contexto)

    if respuesta == FRASE_SIN_INFO or not respuesta:
        return FRASE_SIN_INFO, []
    return respuesta, fuentes
