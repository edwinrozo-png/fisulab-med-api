from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import json, unicodedata, os

# Inicializar API Key desde variables de entorno
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = FastAPI()

def to_ascii(texto: str) -> str:
    if texto is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", texto)
    return nfkd.encode("ascii", "ignore").decode("ascii")

SYSTEM_PROMPT = to_ascii("""
Eres un asistente de escritura en espanol.
Tu tarea SIEMPRE es devolver un JSON EXACTO con esta estructura:

{
  "correccion": "texto corregido completo",
  "sugerencia": "texto alternativo completo",
  "explicacion": "explicacion corta"
}

REGLAS:
- "correccion" debe contener TODO el texto reescrito correctamente, como parrafos completos.
- "sugerencia" debe ser una version alternativa mas clara del mismo contenido.
- "explicacion" debe resumir los cambios mas importantes.
- Sin markdown ni comillas triples.
""")

def corregir_texto(texto: str, modelo: str = "gpt-4o-mini") -> dict:
    if not texto.strip():
        return {
            "correccion": "",
            "sugerencia": "",
            "explicacion": "No se recibio texto para corregir."
        }

    texto_ascii = to_ascii(texto)

    response = client.responses.create(
        model=modelo,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": texto_ascii}
        ]
    )

    try:
        raw_output = response.output_text
    except:
        raw_output = str(response)

    raw_clean = (
        raw_output.strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        data = json.loads(raw_clean)
    except:
        data = {
            "correccion": texto_ascii,
            "sugerencia": texto_ascii,
            "explicacion": "No se pudo parsear JSON"
        }

    return data


# --------------------------------------------------------------
# 🔵 MODELO DE ENTRADA DEL FORM
# --------------------------------------------------------------
class Paciente(BaseModel):
    edad: int
    sintomas: str
    antecedentes: str | None = None

@app.post("/recomendar")
def recomendar(p: Paciente):
    
    # 1️⃣ RECOMENDACIÓN (tu lógica original)
    sintomas_low = p.sintomas.lower()

    if "fiebre" in sintomas_low:
        recomendacion = "Valorar en 24 horas por posible proceso infeccioso."
    else:
        recomendacion = "Seguimiento ambulatorio."

    # 2️⃣ CORRECCIÓN DE TEXTO (OpenAI Mini)
    correccion = corregir_texto(p.sintomas)

    # 3️⃣ DEVOLVER TODO A GOOGLE SHEETS
    return {
        "recomendacion": recomendacion,
        "correccion": correccion["correccion"],
        "sugerencia": correccion["sugerencia"],
        "explicacion": correccion["explicacion"]
    }
