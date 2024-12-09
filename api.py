from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Datos simulados y modelo preentrenado
nota_mapping = {'AD': 3, 'A': 2, 'B': 1, 'C': 0}
result_mapping_inv = {1: 'Promovido', 2: 'Recuperación', 3: 'Permanente'}

# Modelo cargado previamente
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit([[2, 1, 2, 1], [1, 1, 1, 2], [0, 2, 3, 1], [1, 1, 0, 1]], [1, 2, 3, 2])  # Simulado

# API
app = FastAPI()

class Notas(BaseModel):
    nota1: str
    nota2: str
    nota3: str
    nota4: str

def consejos(estado):
    if estado == 'Promovido':
        return "¡Felicidades! Sigue manteniendo tu esfuerzo."
    elif estado == 'Recuperación':
        return "Es importante enfocarse más en las áreas donde hay dificultades. Considera trabajar con un tutor."
    elif estado == 'Permanente':
        return "Deberías mejorar en varias áreas para poder pasar al siguiente grado. Busca ayuda adicional."
@app.get("/")
def read_root():
    return {"message": "API para predecir estado de promoción de alumnos. Usa /predecir/ para realizar predicciones."}

@app.post("/predecir/")
def predecir_rendimiento(notas: Notas):
    # Convertir notas a formato numérico
    try:
        notas_numericas = [nota_mapping[notas.nota1], nota_mapping[notas.nota2], 
                           nota_mapping[notas.nota3], nota_mapping[notas.nota4]]
    except KeyError:
        return {"error": "Notas deben ser 'AD', 'A', 'B', o 'C'."}
    
    # Predecir estado
    prediccion = model.predict([notas_numericas])
    estado_predicho = result_mapping_inv[prediccion[0]]
    consejo = consejos(estado_predicho)

    return {"estado": estado_predicho, "consejo": consejo}
