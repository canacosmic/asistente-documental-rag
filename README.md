# Asistente de preguntas sobre documentos (RAG)

Sube un PDF y hazle preguntas en lenguaje natural. El sistema encuentra la
información relevante dentro del documento y responde en español, **citando
la fuente** de cada respuesta en lugar de inventar.

> Construido como demostración de un sistema RAG (Retrieval-Augmented
> Generation) aplicado a un caso de negocio real: consultar documentación
> extensa sin tener que leerla completa.

---

## El problema que resuelve

Empresas y profesionales acumulan documentos que nadie tiene tiempo de leer:
manuales, reglamentos internos, contratos, catálogos, normativa, protocolos
clínicos. Buscar un dato específico ahí dentro es lento y frustrante.

Este asistente convierte cualquier conjunto de documentos en algo que se
puede **consultar como si le preguntaras a un experto** que se los leyó todos.

**Casos de uso reales:**

- Un estudio contable consulta normativa tributaria al instante.
- Una inmobiliaria revisa cláusulas de decenas de contratos.
- Una clínica consulta sus protocolos internos.
- Soporte técnico responde con base en manuales de producto.

## Cómo funciona

1. **Ingesta** — El documento se parte en fragmentos con solape, para no
   cortar ideas a la mitad.
2. **Indexado** — Cada fragmento se convierte en un vector (embedding) y se
   guarda en una base de datos vectorial (**Chroma**).
3. **Recuperación** — Ante una pregunta, se buscan los fragmentos más
   parecidos semánticamente, no por palabras exactas sino por significado.
4. **Generación** — Un modelo de lenguaje redacta la respuesta usando solo
   esos fragmentos, y se muestran las fuentes consultadas.

```
Documento → Fragmentos → Embeddings → Chroma
                                         ↓
              Pregunta → Búsqueda semántica → Contexto → LLM → Respuesta + fuentes
```

## Stack

| Componente        | Tecnología                                   |
|-------------------|----------------------------------------------|
| Interfaz web      | Gradio                                       |
| Base vectorial    | Chroma                                       |
| Embeddings        | sentence-transformers (multilingüe, local)   |
| Generación        | Modelo local por defecto · Claude Haiku opcional |
| Lectura de PDF    | pypdf                                         |

**Diseño de doble motor:** el demo corre **100% gratis y local** (sin API key).
Para calidad de producción, basta definir `ANTHROPIC_API_KEY` y el sistema
usa Claude Haiku automáticamente. Así el demo público no genera costos y la
versión premium está a un paso.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python app.py
```

Se abre en `http://localhost:7860`. Sube un PDF, procésalo y pregunta.

Para usar Claude en lugar del modelo local:

```bash
export ANTHROPIC_API_KEY="tu-key"
python app.py
```

## Estructura

```
ingest.py       Carga documentos, los trocea e indexa en Chroma
responder.py    Recupera contexto y genera la respuesta (local o Claude)
app.py          Interfaz web en Gradio
```

## Notas de diseño

- Los embeddings son **multilingües**: el sistema entiende preguntas y
  documentos en español sin configuración extra.
- El *prompt* instruye al modelo a responder **solo** con base en el
  contexto y a admitir cuando la respuesta no está en el documento. Esto
  reduce las alucinaciones, clave para que un cliente confíe en el sistema.
- El troceado con solape (`CHUNK_OVERLAP`) evita perder información en los
  cortes entre fragmentos.

---

*Desarrollado por Rodrigo · Sistemas conversacionales y NLP aplicado.*
