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
# 🔵 MODELO DE ENTRADA DEL FORM (GOOGLE SHEETS / FRONT)
# --------------------------------------------------------------
class Paciente(BaseModel):
    edad: int
    sintomas: str
    antecedentes: str | None = None


# --------------------------------------------------------------
# 🧩 SEGMENTACIÓN POR EDAD PARA PACIENTES CON LPH
# --------------------------------------------------------------
def segmentar_paciente_lph(edad: int) -> str:
    """
    Segmentos simples pensados para la ruta de labio y/o paladar fisurado.
    Ajustable según lo que defina FISULAB.
    """
    if edad <= 1:
        return "lactante"             # 0-1 año (alimentación, post-cirugía inicial)
    elif 2 <= edad <= 5:
        return "primera_infancia"     # 2-5 años (lenguaje temprano, otitis, odontología)
    elif 6 <= edad <= 12:
        return "escolar"              # 6-12 (habla, rendimiento escolar, ortodoncia)
    elif 13 <= edad <= 17:
        return "adolescente"          # 13-17 (autoimagen, ortodoncia, apoyo emocional)
    else:
        return "adulto"               # 18+ (revisiones funcionales/estéticas, apoyo psicosocial)


# --------------------------------------------------------------
# 🩺 LÓGICA DE RECOMENDACIÓN BASADA EN REGLAS (ENFOQUE LPH)
# ⚠️ EJEMPLO TÉCNICO, DEBE SER VALIDADO POR EL EQUIPO CLÍNICO
# --------------------------------------------------------------
def generar_recomendacion_lph(p: Paciente) -> str:
    sintomas = to_ascii(p.sintomas).lower()
    antecedentes = to_ascii(p.antecedentes or "").lower()
    segmento = segmentar_paciente_lph(p.edad)

    # Flags simples derivados del texto
    postoperatorio = any(x in sintomas for x in ["posoperatorio", "postoperatorio", "cirugia", "operacion"])
    sangrado_importante = any(x in sintomas for x in ["sangrado abundante", "sangra mucho", "sangrado profuso"])
    fiebre_alta = any(x in sintomas for x in ["fiebre alta", "mas de 38", "mas de 38.5"])
    dificultad_respirar = any(x in sintomas for x in ["dificultad para respirar", "ahogo", "falta de aire"])
    dolor_intenso = any(x in sintomas for x in ["dolor muy fuerte", "dolor intenso", "dolor que no cede"])
    signos_infeccion_herida = any(
        x in sintomas for x in ["enrojecimiento", "secrecion", "pus", "mal olor en la herida"]
    )

    problemas_alimentacion = any(
        x in sintomas
        for x in [
            "no gana peso", "baja de peso", "dificultad para alimentarse",
            "tarda mucho en comer", "se atora", "sale leche por la nariz"
        ]
    )
    problemas_habla = any(
        x in sintomas
        for x in ["no se entiende", "habla nasal", "voz nasal", "rinolalia", "dificultad para hablar"]
    )
    signos_otitis = any(
        x in sintomas
        for x in ["dolor de oido", "supuracion de oido", "otitis", "escucha poco", "baja audicion"]
    )
    impacto_emocional = any(
        x in sintomas
        for x in ["triste", "rechazo", "burlas", "bullying", "no quiere salir", "verguenza"]
    )

    # 🔴 1) POSIBLE URGENCIA / COMPLICACIONES POSTOPERATORIAS
    if postoperatorio and (sangrado_importante or fiebre_alta or dificultad_respirar or dolor_intenso or signos_infeccion_herida):
        return (
            "Se sugiere que el paciente acuda de inmediato a un servicio de urgencias y contacte al equipo tratante, "
            "por posibles complicaciones posoperatorias relacionadas con labio y/o paladar fisurado."
        )

    # 🟠 2) LACTANTES CON PROBLEMAS DE ALIMENTACIÓN / PESO
    if segmento == "lactante" and problemas_alimentacion:
        return (
            "Se recomienda priorizar valoracion presencial en corto plazo para revisar alimentacion, ganancia de peso "
            "y tecnica de alimentacion en contexto de labio y/o paladar fisurado, considerando apoyo de nutricion "
            "y terapia orofacial segun criterio del equipo clinico."
        )

    # 🟡 3) PROBLEMAS DE HABLA / AUDICIÓN EN INFANCIA / ESCOLAR / ADOLESCENTE
    if segmento in ["primera_infancia", "escolar", "adolescente"] and (problemas_habla or signos_otitis):
        return (
            "Se recomienda programar valoracion interdisciplinaria (fonoaudiologia y otorrinolaringologia) "
            "para evaluar habla y audicion en un paciente con labio y/o paladar fisurado."
        )

    # 🟡 4) IMPACTO EMOCIONAL EN ADOLESCENTES Y ADULTOS
    if segmento in ["adolescente", "adulto"] and impacto_emocional:
        return (
            "Se sugiere valoracion por psicologia o trabajo social para abordar impacto emocional, autoimagen "
            "y posibles situaciones de rechazo o bullying asociadas al labio y/o paladar fisurado."
        )

    # 🟢 5) CASO BASE: SEGUIMIENTO PROGRAMADO
    return (
        "Se recomienda control ambulatorio en el marco de la ruta integral de cuidado para labio y/o paladar fisurado, "
        "ajustando la prioridad segun criterio del equipo interdisciplinario."
    )


# --------------------------------------------------------------
# 🤖 PROMPT TÉCNICO PARA REFINAR RECOMENDACIÓN (OPENAI)
# --------------------------------------------------------------
SYSTEM_PROMPT_RECOM_TECNICA = to_ascii("""
Eres un profesional de la salud que redacta planes de manejo para un equipo interdisciplinario
que atiende a personas con labio y/o paladar fisurado.

Tu tarea es redactar UNA SOLA recomendacion tecnica breve (2 o 3 frases maximo),
dirigida al equipo de salud, sin emitir diagnosticos ni indicar medicamentos.

REGLAS:
- Usa lenguaje tecnico-clinico pero claro.
- No des diagnosticos ni nombres de enfermedades.
- No indiques farmacos ni dosis.
- Puedes referirte a valoraciones (nutricion, fonoaudiologia, psicologia, otorrinolaringologia, cirugia, etc.).
- Usa tercera persona y un tono profesional.
- No uses markdown ni viñetas.
""")


def refinar_recomendacion_tecnica(p: Paciente, recomendacion_base: str, modelo: str = "gpt-4o-mini") -> str:
    segmento = segmentar_paciente_lph(p.edad)

    user_prompt = to_ascii(f"""
Contexto del paciente:
- Diagnostico de base: labio y/o paladar fisurado.
- Edad (anios): {p.edad}
- Segmento: {segmento}
- Sintomas o motivo de consulta (texto libre): {p.sintomas}
- Antecedentes relevantes: {p.antecedentes or 'No reporta'}

Recomendacion base sugerida por el sistema:
{recomendacion_base}

A partir de esta informacion, redacta UNA recomendacion tecnica breve (2-3 frases),
dirigida al equipo de salud, coherente con la recomendacion base y con el contexto de labio/paladar fisurado.
""")

    response = client.responses.create(
        model=modelo,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_RECOM_TECNICA},
            {"role": "user",   "content": user_prompt}
        ]
    )

    try:
        texto = response.output_text.strip()
    except Exception:
        texto = recomendacion_base  # fallback si algo sale mal

    return texto


# --------------------------------------------------------------
# 🔵 ENDPOINT PRINCIPAL
# --------------------------------------------------------------
@app.post("/recomendar")
def recomendar(p: Paciente, usar_ia_en_recomendacion: bool = True):
    """
    Endpoint que devuelve:
    - recomendacion: basada en LPH + edad (reglas + opcional refinamiento IA tecnico)
    - correccion / sugerencia / explicacion: correccion de texto de sintomas
    """

    # 1️⃣ Reglas: generan recomendacion_base (deterministica, auditable)
    recomendacion_base = generar_recomendacion_lph(p)

    # 2️⃣ IA: refina el texto para que suene mas tecnico (si esta activado)
    if usar_ia_en_recomendacion:
        recomendacion = refinar_recomendacion_tecnica(p, recomendacion_base)
    else:
        recomendacion = recomendacion_base

    # 3️⃣ Correccion del texto de sintomas (tu proceso original)
    correccion = corregir_texto(p.sintomas)

    # 4️⃣ Respuesta para Google Sheets / front
    return {
        "recomendacion": recomendacion,
        "correccion": correccion["correccion"],
        "sugerencia": correccion["sugerencia"],
        "explicacion": correccion["explicacion"]
    }

