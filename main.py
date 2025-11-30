from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import json, unicodedata, os

# Inicializar API Key desde variables de entorno
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = FastAPI()

# --------------------------------------------------------------
# 🔣 UTILIDAD: NORMALIZAR TEXTO A ASCII
# --------------------------------------------------------------
def to_ascii(texto: str) -> str:
    if texto is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", texto)
    return nfkd.encode("ascii", "ignore").decode("ascii")


# --------------------------------------------------------------
# 🧠 PROMPT DE CORRECCIÓN DE TEXTO
# --------------------------------------------------------------
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
    if not texto or not texto.strip():
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
    except Exception:
        raw_output = str(response)

    raw_clean = (
        raw_output.strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        data = json.loads(raw_clean)
    except Exception:
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


# --------------------------------------------------------------
# 🧩 SEGMENTACIÓN POR EDAD PARA PACIENTES CON LPH
# --------------------------------------------------------------
def segmentar_paciente_lph(edad: int) -> str:
    if edad <= 1:
        return "lactante"
    elif 2 <= edad <= 5:
        return "primera_infancia"
    elif 6 <= edad <= 12:
        return "escolar"
    elif 13 <= edad <= 17:
        return "adolescente"
    else:
        return "adulto"


# --------------------------------------------------------------
# 🩺 LÓGICA BASE DE RECOMENDACIÓN LPH
# --------------------------------------------------------------
def generar_recomendacion_lph(p: Paciente) -> str:
    sintomas = to_ascii(p.sintomas).lower()
    antecedentes = to_ascii(p.antecedentes or "").lower()
    segmento = segmentar_paciente_lph(p.edad)

    postoperatorio = any(x in sintomas for x in ["posoperatorio", "postoperatorio", "cirugia", "operacion"])
    sangrado_importante = any(x in sintomas for x in ["sangrado abundante", "sangra mucho", "sangrado profuso"])
    fiebre_alta = any(x in sintomas for x in ["fiebre alta", "mas de 38", "mas de 38.5"])
    dificultad_respirar = any(x in sintomas for x in ["dificultad para respirar", "ahogo", "falta de aire"])
    dolor_intenso = any(x in sintomas for x in ["dolor muy fuerte", "dolor intenso", "dolor que no cede"])
    signos_infeccion_herida = any(
        x in sintomas for x in ["enrojecimiento", "secrecion", "pus", "mal olor en la herida"]
    )

    problemas_alimentacion = any(
        x in sintomas for x in [
            "no gana peso", "baja de peso", "dificultad para alimentarse",
            "tarda mucho en comer", "se atora", "sale leche por la nariz"
        ]
    )
    problemas_habla = any(
        x in sintomas for x in [
            "no se entiende", "habla nasal", "voz nasal", "rinolalia", "dificultad para hablar"
        ]
    )
    signos_otitis = any(
        x in sintomas for x in [
            "dolor de oido", "supuracion de oido", "otitis", "escucha poco", "baja audicion"
        ]
    )
    impacto_emocional = any(
        x in sintomas for x in ["triste", "rechazo", "burlas", "bullying", "no quiere salir", "verguenza"]
    )

    # 🔴 Signos de alarma
    if postoperatorio and (sangrado_importante or fiebre_alta or dificultad_respirar or dolor_intenso or signos_infeccion_herida):
        return "Urgencias por posible complicación posoperatoria relacionada con labio y/o paladar fisurado."

    # 🟠 Lactantes con problemas de alimentación
    if segmento == "lactante" and problemas_alimentacion:
        return "Valoración prioritaria de alimentación y ganancia de peso con nutrición y terapia orofacial."

    # 🟡 Problemas de habla o audición
    if segmento in ["primera_infancia", "escolar", "adolescente"] and (problemas_habla or signos_otitis):
        return "Valoración por fonoaudiología y otorrinolaringología para estudio de habla y audición en contexto de LPH."

    # 🟡 Impacto emocional
    if segmento in ["adolescente", "adulto"] and impacto_emocional:
        return "Valoración por psicología o trabajo social para apoyo emocional y abordaje psicosocial."

    # 🟢 Caso base
    return "Seguimiento ambulatorio en la ruta de atención para labio y/o paladar fisurado."


# --------------------------------------------------------------
# 🤖 PROMPT TÉCNICO PARA REFINAR RECOMENDACIÓN
# --------------------------------------------------------------
SYSTEM_PROMPT_RECOM_TECNICA = to_ascii("""
Eres un profesional de la salud que redacta recomendaciones tecnicas para un equipo interdisciplinario
que atiende a personas con labio y/o paladar fisurado.

Tu tarea es redactar UNA SOLA recomendacion tecnica breve (2-3 frases maximo),
sin emitir diagnosticos ni formular medicamentos.

Reglas:
- Usa lenguaje clinico claro.
- Evita diagnosticos y nombres de enfermedades.
- Puedes sugerir valoraciones (nutricion, fonoaudiologia, psicologia, otorrinolaringologia, cirugia).
- Usa tercera persona profesional.
- No uses markdown, listas ni comillas.
""")

def refinar_recomendacion_tecnica(p: Paciente, recomendacion_base: str, modelo: str = "gpt-4o-mini") -> str:
    segmento = segmentar_paciente_lph(p.edad)

    user_prompt = to_ascii(f"""
Paciente con labio y/o paladar fisurado.
Edad: {p.edad} anios
Segmento: {segmento}
Sintomas: {p.sintomas}
Antecedentes: {p.antecedentes or 'No reporta'}

Recomendacion base:
{recomendacion_base}

Genera una recomendacion tecnica breve y coherente con el contexto de LPH.
""")

    response = client.responses.create(
        model=modelo,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_RECOM_TECNICA},
            {"role": "user", "content": user_prompt}
        ]
    )

    try:
        texto = response.output_text.strip()
    except Exception:
        texto = recomendacion_base

    return texto


# --------------------------------------------------------------
# 🔵 ENDPOINT PRINCIPAL (Siempre usa IA técnica)
# --------------------------------------------------------------
@app.post("/recomendar")
def recomendar(p: Paciente):
    # 1️⃣ Reglas base
    recomendacion_base = generar_recomendacion_lph(p)

    # 2️⃣ Refinamiento técnico SIEMPRE activado
    recomendacion = refinar_recomendacion_tecnica(p, recomendacion_base)

    # 3️⃣ Corrección del texto de síntomas
    correccion = corregir_texto(p.sintomas)

    # 4️⃣ Respuesta final
    return {
        "recomendacion": recomendacion,
        "correccion": correccion["correccion"],
        "sugerencia": correccion["sugerencia"],
        "explicacion": correccion["explicacion"]
    }

