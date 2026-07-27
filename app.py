"""
app.py
------
Interfaz web del asistente de preguntas sobre documentos.

Flujo:
  1. El usuario sube uno o varios PDFs (o archivos de texto).
  2. Se indexan en Chroma con embeddings locales.
  3. El usuario hace preguntas y recibe respuestas fundamentadas, con las fuentes.

Corre gratis con modelo local. Si defines ANTHROPIC_API_KEY, usa Claude Haiku.

Ejecutar:  python app.py
"""

import tempfile
from pathlib import Path

import gradio as gr

from ingest import construir_indice
from responder import responder


# Estado de la app: la colección viva entre interacciones.
estado = {"coleccion": None}


def indexar_archivos(archivos):
    """Callback: recibe los archivos subidos y construye el índice."""
    if not archivos:
        return "Sube al menos un documento para empezar."

    rutas = [Path(a.name) for a in archivos]
    estado["coleccion"] = construir_indice(rutas)
    nombres = ", ".join(r.name for r in rutas)
    return f"Listo. Indexé: {nombres}. Ya puedes hacer preguntas."


def preguntar(pregunta, historial):
    """Callback del chat: responde usando el índice."""
    if estado["coleccion"] is None:
        historial.append({"role": "user", "content": pregunta})
        historial.append({"role": "assistant", "content": "Primero sube y procesa un documento arriba."})
        return historial, ""

    if not pregunta.strip():
        return historial, ""

    respuesta, fuentes = responder(estado["coleccion"], pregunta)

    nombres_fuentes = sorted({f["fuente"] for f in fuentes})
    pie = "\n\n_Fuentes consultadas: " + ", ".join(nombres_fuentes) + "_"

    historial.append({"role": "user", "content": pregunta})
    historial.append({"role": "assistant", "content": respuesta + pie})
    return historial, ""


# --- Interfaz ---------------------------------------------------------------

# Tema con identidad propia: azul pizarra profundo + neutros cálidos.
# Evita el gris default de Gradio para que se vea intencional ante un cliente.
tema = gr.themes.Soft(
    primary_hue=gr.themes.colors.slate,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

CSS = """
.gradio-container { max-width: 900px !important; margin: auto !important; }
#titulo { text-align: center; margin-bottom: 0.2rem; }
#subtitulo { text-align: center; color: #64748b; margin-top: 0; }
footer { visibility: hidden; }
"""

with gr.Blocks(title="Asistente Documental") as demo:
    gr.Markdown("# Asistente de preguntas sobre tus documentos", elem_id="titulo")
    gr.Markdown(
        "Sube un PDF y pregúntale lo que quieras. Responde en español, "
        "citando de dónde salió cada respuesta.",
        elem_id="subtitulo",
    )

    with gr.Row():
        archivos = gr.File(
            label="Tus documentos (PDF o texto)",
            file_count="multiple",
            file_types=[".pdf", ".txt", ".md"],
        )

    boton_procesar = gr.Button("Procesar documentos", variant="primary")
    estado_texto = gr.Markdown("")

    gr.Markdown("---")

    chat = gr.Chatbot(label="Conversación", height=380)
    with gr.Row():
        entrada = gr.Textbox(
            placeholder="Escribe tu pregunta y presiona Enter...",
            show_label=False,
            scale=8,
        )
        boton_enviar = gr.Button("Preguntar", variant="primary", scale=1)

    gr.Markdown(
        "_Demo de RAG (Retrieval-Augmented Generation) · Chroma + embeddings "
        "multilingües · corre en local, con opción de Claude para producción._"
    )

    # Conexiones
    boton_procesar.click(indexar_archivos, inputs=archivos, outputs=estado_texto)
    boton_enviar.click(preguntar, inputs=[entrada, chat], outputs=[chat, entrada])
    entrada.submit(preguntar, inputs=[entrada, chat], outputs=[chat, entrada])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=tema, css=CSS)
