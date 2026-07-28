# Kortex · Asistente RAG

Sube un documento y hazle preguntas en lenguaje natural. Kortex encuentra la
información relevante dentro del documento y responde en español **citando la
fuente**, en lugar de inventar.

Si la respuesta no está en el documento, lo dice claramente
("No encuentro esa información en el documento") en vez de alucinar — el
comportamiento que distingue a un sistema RAG confiable de uno que no lo es.

---

## El problema que resuelve

Empresas y profesionales acumulan documentos que nadie tiene tiempo de leer:
manuales, reglamentos internos, contratos, catálogos, normativa, protocolos.
Buscar un dato específico ahí dentro es lento y frustrante.

Kortex convierte cualquier documento en algo que se puede **consultar como si
le preguntaras a un experto** que se lo leyó entero — y que solo responde con
lo que el documento realmente dice.

**Casos de uso:**

- Un estudio contable consulta normativa tributaria al instante.
- Una inmobiliaria revisa cláusulas de decenas de contratos.
- Una clínica consulta sus protocolos internos.
- Soporte técnico responde con base en manuales de producto.

## Cómo funciona

1. **Ingesta** — El documento se parte en fragmentos con solape.
2. **Indexado** — Cada fragmento se convierte en un vector (embedding) y se
   guarda en una base de datos vectorial (**Chroma**).
3. **Recuperación** — Ante una pregunta, se buscan los fragmentos más
   parecidos por significado, no por palabras exactas.
4. **Filtro anti-alucinación** — Si ningún fragmento es suficientemente
   relevante (según un umbral de distancia), Kortex responde honestamente que
   no encuentra la información, sin llamar al modelo.
5. **Generación** — El modelo redacta la respuesta usando solo esos
   fragmentos, y se muestra la fuente citada.

```
Documento → Fragmentos → Embeddings → Chroma
                                         ↓
   Pregunta → Búsqueda semántica → ¿relevante? → Contexto → Modelo → Respuesta + fuente
                                        ↓ no
                          "No encuentro esa información"
```

## Arquitectura

Kortex separa el backend de la interfaz, como una aplicación web real:

- **Backend** — API en **FastAPI** que expone los endpoints de procesar,
  preguntar y limpiar. Contiene toda la lógica de RAG.
- **Frontend** — Interfaz web a medida (HTML/CSS/JS), sin frameworks pesados,
  con diseño de dos columnas y modo claro/oscuro automático.
- **Doble motor de generación** — Corre **100% local y gratis** por defecto
  (modelo Qwen2.5, sin API key). Si se define `ANTHROPIC_API_KEY`, usa
  **Claude Haiku** para calidad de producción. El demo público no genera
  costos; la versión premium está a un paso.

| Componente     | Tecnología                              |
|----------------|-----------------------------------------|
| Backend        | FastAPI + Uvicorn                       |
| Base vectorial | Chroma                                  |
| Embeddings     | sentence-transformers (multilingüe)     |
| Generación     | Qwen2.5 local · Claude Haiku opcional   |
| Lectura de PDF | pypdf                                   |
| Frontend       | HTML/CSS/JS a medida                    |

## Cómo ejecutarlo

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```

Abre `http://127.0.0.1:8000`, sube un documento y pregunta.

Para usar Claude en lugar del modelo local:

```bash
export ANTHROPIC_API_KEY="tu-key"
uvicorn server:app --reload --port 8000
```

## Estructura

```
server.py       Backend FastAPI: endpoints y arranque
ingest.py       Carga documentos, los trocea e indexa en Chroma
responder.py    Recupera contexto, filtra y genera la respuesta
static/         Frontend web (index.html)
```

## Decisiones de diseño

- **Filtro anti-alucinación por umbral de distancia:** si la pregunta no se
  parece a nada del documento, se corta antes de llamar al modelo. Esto evita
  que el sistema invente respuestas — clave para que un cliente confíe.
- **Embeddings multilingües:** entiende preguntas y documentos en español sin
  configuración extra.
- **Limpieza de salida:** se filtran divagaciones, etiquetas sobrantes y
  cualquier idioma que el modelo local pudiera colar, para respuestas limpias.
- **Precarga del modelo:** el modelo se carga al arrancar el servidor, no en la
  primera pregunta, para que todas las respuestas sean ágiles.

## Mejoras futuras (roadmap)

Kortex es un MVP funcional. Las siguientes mejoras escalarían su calidad de
recuperación y respuesta:

- **Query expansion** — reformular la pregunta antes de buscar, para recuperar
  fragmentos que con las palabras exactas se escaparían.
- **Reranking** — reordenar los fragmentos recuperados con un modelo dedicado
  para priorizar los más pertinentes.
- **Búsqueda híbrida** — combinar búsqueda semántica con búsqueda por palabra
  clave (BM25) para cubrir más casos.
- **Chunking inteligente** — trocear respetando la estructura del documento
  (secciones, párrafos) en lugar de por tamaño fijo.
- **Evaluación con métricas** — medir la calidad de las respuestas de forma
  sistemática para iterar con datos.

---

*Desarrollado por Rodrigo · Sistemas conversacionales y NLP aplicado.*
