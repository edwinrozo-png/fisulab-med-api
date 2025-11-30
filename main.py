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
# 🧠 PROMPT DE CORRECCIÓN DE TEXTO (EL MISMO QUE YA USABAS)
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
        for x in ["no gana peso", "baja de peso", "dificultad para alimentarse",
                  "tarda mucho en comer", "se atora", "sale leche por la nariz"]
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
            "Sugerir que el paciente acuda de inmediato a un servicio de urgencias y contacte al equipo tratante, "
            "por posibles complicaciones posoperatorias relacionadas con labio y/o paladar fisurado."
        )

    # 🟠 2) LACTANTES CON PROBLEMAS DE ALIMENTACIÓN / PESO
    if segmento == "lactante" and problemas_alimentacion:
        return (
            "Priorizar valoracion presencial en corto plazo para revisar alimentacion, ganancia de peso "
            "y tecnica de alimentacion en contexto de labio y/o paladar fisurado. "
            "Se puede considerar apoyo de nutricion y terapia orofacial segun criterio del equipo clinico."
        )

    # 🟡 3) PROBLEMAS DE HABLA / AUDICIÓN EN INFANCIA / ESCOLAR / ADOLESCENTE
    if segmento in ["primera_infancia", "escolar", "adolescente"] and (problemas_habla or signos_otitis):
        return (
            "Recomendar programar valoracion interdisciplinaria (fonoaudiologia y otorrinolaringologia) "
            "para evaluar habla y audicion en paciente con labio y/o paladar fisurado."
        )

    # 🟡 4) IMPACTO EMOCIONAL EN ADOLESCENTES Y ADULTOS
    if segmento in ["adolescente", "adulto"] and impacto_emocional:
        return (
            "Sugerir valoracion por psicologia o trabajo social para abordar impacto emocional, "
            "autoimagen y posibles situaciones de rechazo o bullying asociadas al labio y/o paladar fisurado."
        )

    # 🟢 5) CASO BASE: SEGUIMIENTO PROGRAMADO
    return (
        "Sugerir seguimiento ambulatorio dentro de la ruta habitual de cuidado para labio y/o paladar fisurado, "
        "ajustando la prioridad segun la disponibilidad del equipo interdisciplinario."
    )


# --------------------------------------------------------------
# 🤖 OPCIONAL: RECOMENDACIÓN ENRIQUECIDA CON OPENAI (LPH)
# --------------------------------------------------------------
SYSTEM_PROMPT_LPH = to_ascii("""
Eres un profesional de la salud que apoya a un equipo interdisciplinario
que atiende a personas con labio y/o paladar fisurado.

Tu tarea es sugerir un tipo de recomendacion NO DIAGNOSTICA en lenguaje claro
para la familia y para el equipo tratante.

SIEMPRE debes responder con un solo parrafo corto, sin diagnosticar ni indicar medicamentos.
No uses markdown. No des ordenes tajantes, usa expresiones como "se puede sugerir",
"es recomendable considerar", "el equipo tratante podria valorar".

La recomendacion debe ser SIEMPRE considerada como apoyo y debera ser revisada
por el equipo clinico de la institucion.
""")

def generar_recomendacion_lph_con_ia(p: Paciente, modelo: str = "gpt-4o-mini") -> str:
    segmento = segmentar_paciente_lph(p.edad)
    user_prompt = to_ascii(f"""
Paciente con labio y/o paladar fisurado.
Edad (anios): {p.edad}
Segmento: {segmento}
Sintomas o motivo de consulta: {p.sintomas}
Antecedentes relevantes: {p.antecedentes or 'No reporta'}

Genera una recomendacion breve y general, sin diagnosticos,
solo sobre tipo de atencion sugerida (ej. seguimiento, valoracion prioritaria,
consulta interdisciplinaria, apoyo emocional, etc.).
""")

    response = client.responses.create(
        model=modelo,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_LPH},
            {"role": "user",   "content": user_prompt}
        ]
    )

    try:
        texto = response.output_text.strip()
    except Exception:
        texto = "No se pudo generar una recomendacion automatica. El caso debe ser revisado por el equipo clinico."

    return texto


# --------------------------------------------------------------
# 🔵 ENDPOINT PRINCIPAL
# --------------------------------------------------------------
@app.post("/recomendar")
def recomendar(p: Paciente, usar_ia_en_recomendacion: bool = False):
    """
    Endpoint que devuelve:
    - recomendacion: basada en LPH + edad (reglas o IA)
    - correccion / sugerencia / explicacion: correccion de texto de sintomas
    """

    # 1️⃣ RECOMENDACIÓN PRINCIPAL (LPH + EDAD)
    if usar_ia_en_recomendacion:
        recomendacion = generar_recomendacion_lph_con_ia(p)
    else:
        recomendacion = generar_recomendacion_lph(p)

    # 2️⃣ CORRECCIÓN DE TEXTO (OpenAI, igual que antes)
    correccion = corregir_texto(p.sintomas)

    # 3️⃣ RESPUESTA PARA GOOGLE SHEETS
    return {
        "recomendacion": recomendacion,
        "correccion": correccion["correccion"],
        "sugerencia": correccion["sugerencia"],
        "explicacion": correccion["explicacion"]
    }
