from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Ajusta estos campos luego a tu formulario real
class Paciente(BaseModel):
    edad: int
    sintomas: str
    antecedentes: str | None = None

@app.post("/recomendar")
def recomendar(p: Paciente):
    # 🚨 Aquí luego pones tu modelo real (cargar .pkl, etc.)
    # Por ahora: lógica de ejemplo
    sintomas = p.sintomas.lower()

    if "fiebre" in sintomas or "temperatura" in sintomas:
        recomendacion = "Valorar en las próximas 24 horas por posible proceso infeccioso."
    elif "dolor" in sintomas and "pecho" in sintomas:
        recomendacion = "Priorizar valoración médica inmediata por posible compromiso cardiovascular."
    else:
        recomendacion = "Seguimiento ambulatorio. No se identifican signos claros de alarma en la descripción."

    return {"recomendacion": recomendacion}
