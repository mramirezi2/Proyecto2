# ======================================================
# Modelo de clasificación - Brecha digital
# Pregunta de negocio:
# ¿El acceso a recursos tecnológicos en el hogar permite
# clasificar a los estudiantes del Cauca según su probabilidad
# de bajo desempeño en las pruebas Saber 11?
# ======================================================

import pandas as pd
import numpy as np
import mlflow

# Importar el archivo de limpieza que ya existe en el proyecto
import Proyecto2

# Tomar el DataFrame limpio resultante del archivo Proyecto2.py
df_limpio = Proyecto2.df

# Crear una copia para trabajar la pregunta de brecha digital
df_brecha = df_limpio.copy()

# ======================================================
# 3. Selección de variables para el modelo de brecha digital
# ======================================================

variables_x = ["fami_tienecomputador","fami_tieneinternet","fami_estratovivienda","fami_personashogar"]
variable_y_base = "punt_global"

# Crear dataframe solo con las variables necesarias
df_modelo_pau = df_brecha[variables_x + [variable_y_base]].copy()

# ======================================================
# 4. Limpieza y codificación de variables explicativas
# ======================================================

# Normalizar texto
for col in ["fami_tienecomputador", "fami_tieneinternet", "fami_personashogar"]:
    df_modelo_pau[col] = df_modelo_pau[col].astype(str).str.strip().str.lower()

# Convertir computador e internet a variables dicotómicas
mapa_si_no = {"si": 1, "sí": 1, "no": 0}

df_modelo_pau["fami_tienecomputador"] = df_modelo_pau["fami_tienecomputador"].map(mapa_si_no)
df_modelo_pau["fami_tieneinternet"] = df_modelo_pau["fami_tieneinternet"].map(mapa_si_no)

# Asegurar que estrato sea numérico
df_modelo_pau["fami_estratovivienda"] = pd.to_numeric(df_modelo_pau["fami_estratovivienda"],errors="coerce")

# Convertir personas del hogar de texto a número
mapa_personas = {
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "diez o más": 10,
    "mas de diez": 10,
    "más de diez": 10}

df_modelo_pau["fami_personashogar"] = df_modelo_pau["fami_personashogar"].map(mapa_personas)

# Asegurar que punt_global sea numérico
df_modelo_pau["punt_global"] = pd.to_numeric(df_modelo_pau["punt_global"],errors="coerce")

# Eliminar filas con faltantes después de la codificación
df_modelo_pau = df_modelo_pau.dropna().copy()

# ======================================================
# 5. Creación de la variable objetivo: bajo_desempeno
# ======================================================

# Bajo desempeño = estudiantes en el 25% inferior del puntaje global
umbral_bajo = df_modelo_pau["punt_global"].quantile(0.25)

df_modelo_pau["bajo_desempeno"] = np.where(df_modelo_pau["punt_global"] <= umbral_bajo,1,0)

# ======================================================
# 6. Separar X e y
# ======================================================

X = df_modelo_pau[["fami_tienecomputador","fami_tieneinternet","fami_estratovivienda","fami_personashogar"]].copy()
y = df_modelo_pau["bajo_desempeno"].copy()

# Verificación
print("Umbral de bajo desempeño:", umbral_bajo)
print("\nDistribución de la variable objetivo:")
print(y.value_counts())
print("\nPrimeras filas de X:")
print(X.head())

# ======================================================
# 7. División entrenamiento / validación / prueba
# ======================================================
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Normalization
from tensorflow.keras.callbacks import EarlyStopping


# Fijar semillas para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# Configuración de MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("brecha_digital_saber11")

# Primera división: entrenamiento + validación / prueba
X_train_val, X_test, y_train_val, y_test = train_test_split(X,y,
    test_size=0.20,
    random_state=42,
    stratify=y)

# Segunda división: entrenamiento / validación
X_train, X_val, y_train, y_val = train_test_split(X_train_val,y_train_val,
    test_size=0.20,
    random_state=42,
    stratify=y_train_val)

# Convertir a arrays para TensorFlow
X_train_np = X_train.astype("float32").values
X_val_np = X_val.astype("float32").values
X_test_np = X_test.astype("float32").values

y_train_np = y_train.astype("float32").values
y_val_np = y_val.astype("float32").values
y_test_np = y_test.astype("float32").values

print("X_train:", X_train_np.shape)
print("X_val:", X_val_np.shape)
print("X_test:", X_test_np.shape)

# ======================================================
# 8. Pesos de clase para manejar desbalance
# ======================================================

clases = np.unique(y_train_np)

pesos = compute_class_weight(
    class_weight="balanced",
    classes=clases,
    y=y_train_np)

class_weight = {int(clase): peso for clase, peso in zip(clases, pesos)}
print("Pesos de clase:", class_weight)

# ======================================================
# 9. Función para construir modelos
# ======================================================

def construir_modelo(nombre_modelo, capas, dropout=0.0, learning_rate=0.001):
    """
    Construye una red neuronal con capa de normalización.
    
    Parámetros:
    - nombre_modelo: nombre de la arquitectura.
    - capas: lista con número de neuronas por capa oculta.
    - dropout: proporción de dropout.
    - learning_rate: tasa de aprendizaje.
    """

    normalizador = Normalization(axis=-1, name=f"normalizacion_{nombre_modelo}")
    normalizador.adapt(X_train_np)

    entrada = Input(shape=(X_train_np.shape[1],), name="variables_brecha_digital")
    x = normalizador(entrada)

    for i, neuronas in enumerate(capas):
        x = Dense(neuronas, activation="relu", name=f"capa_{i+1}_{neuronas}")(x)

        if dropout > 0:
            x = Dropout(dropout, name=f"dropout_{i+1}")(x)

    salida = Dense(1, activation="sigmoid", name="prob_bajo_desempeno")(x)

    modelo = Model(inputs=entrada, outputs=salida, name=nombre_modelo)

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"])
    return modelo

# ======================================================
# 10. Arquitecturas a comparar
# ======================================================

arquitecturas = {
    "Modelo_1_simple": {
        "capas": [8],
        "dropout": 0.1,
        "learning_rate": 0.001
    },
    "Modelo_2_intermedio": {
        "capas": [16, 8],
        "dropout": 0.1,
        "learning_rate": 0.001
    },
    "Modelo_3_profundo": {
        "capas": [32, 16, 8],
        "dropout": 0.1,
        "learning_rate": 0.001
    }}

# ======================================================
# 11. Entrenamiento y evaluación de modelos
# ======================================================

resultados = []
modelos_entrenados = {}
historiales_entrenamiento = {}
run_ids = {}

EPOCHS = 50
BATCH_SIZE = 32

for nombre, config in arquitecturas.items():

    print("\n" + "="*60)
    print(f"Entrenando: {nombre}")
    print("="*60)

    # Construir modelo
    modelo = construir_modelo(
        nombre_modelo=nombre,
        capas=config["capas"],
        dropout=config["dropout"],
        learning_rate=config["learning_rate"])

    # Entrenar modelo
    historial = modelo.fit(
        X_train_np,
        y_train_np,
        validation_data=(X_val_np, y_val_np),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        verbose=0)

    # Guardar historial para graficar curvas de aprendizaje
    historiales_entrenamiento[nombre] = historial

    # Predicciones en validación
    y_val_prob = modelo.predict(X_val_np, verbose=0)
    y_val_pred = (y_val_prob >= 0.5).astype(int).ravel()

    # Predicciones en prueba
    y_test_prob = modelo.predict(X_test_np, verbose=0)
    y_test_pred = (y_test_prob >= 0.5).astype(int).ravel()

    # Métricas
    val_acc = accuracy_score(y_val_np, y_val_pred)
    val_f1 = f1_score(y_val_np, y_val_pred, zero_division=0)

    test_acc = accuracy_score(y_test_np, y_test_pred)
    test_f1 = f1_score(y_test_np, y_test_pred, zero_division=0)

        # ======================================================
    # Registro del experimento en MLflow
    # ======================================================

    with mlflow.start_run(run_name=nombre) as run:

        mlflow.log_param("modelo", nombre)
        mlflow.log_param("arquitectura", str(config["capas"]))
        mlflow.log_param("dropout", config["dropout"])
        mlflow.log_param("learning_rate", config["learning_rate"])
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("variables_x", ", ".join(variables_x))
        mlflow.log_param("umbral_bajo_desempeno", float(umbral_bajo))
        mlflow.log_param("capa_normalizacion", "Normalization")

        mlflow.log_metric("val_acc", val_acc)
        mlflow.log_metric("val_f1", val_f1)
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_f1", test_f1)

        mlflow.log_metric("loss_final", historial.history["loss"][-1])
        mlflow.log_metric("val_loss_final", historial.history["val_loss"][-1])

        run_ids[nombre] = run.info.run_id

    # Guardar resultados
    resultados.append({
        "Modelo": nombre,
        "Arquitectura": str(config["capas"]),
        "Dropout": config["dropout"],
        "Val Acc": round(val_acc, 4),
        "Val F1": round(val_f1, 4),
        "Test Acc": round(test_acc, 4),
        "Test F1": round(test_f1, 4),
        "Épocas": EPOCHS})

    # Guardar modelo entrenado
    modelos_entrenados[nombre] = modelo

# ======================================================
# 12. Tabla comparativa de resultados
# ======================================================

df_resultados = pd.DataFrame(resultados)

df_resultados = df_resultados.sort_values(
    by=["Val F1", "Val Acc"],
    ascending=False
).reset_index(drop=True)

print("\nResultados comparativos:")
print(df_resultados)

# ======================================================
# 13. Selección del mejor modelo
# ======================================================

mejor_modelo_nombre = df_resultados.loc[0, "Modelo"]
mejor_modelo = modelos_entrenados[mejor_modelo_nombre]

print("\nMejor modelo seleccionado:")
print(mejor_modelo_nombre)

print("\nResumen del mejor modelo:")
mejor_modelo.summary()

# ======================================================
# 14. Evaluación final del mejor modelo
# ======================================================

from sklearn.metrics import confusion_matrix, classification_report

# Predicciones del mejor modelo en validación y prueba
y_val_prob = mejor_modelo.predict(X_val_np, verbose=0).ravel()
y_test_prob = mejor_modelo.predict(X_test_np, verbose=0).ravel()

# Clasificación usando umbral 0.5
y_val_pred = (y_val_prob >= 0.5).astype(int)
y_test_pred = (y_test_prob >= 0.5).astype(int)

print("\nMatriz de confusión - Mejor modelo en prueba:")
print(confusion_matrix(y_test_np, y_test_pred))

print("\nReporte de clasificación - Mejor modelo en prueba:")
print(classification_report(y_test_np, y_test_pred, zero_division=0))

# ======================================================
# 15. Guardar resultados comparativos
# ======================================================

import os

os.makedirs("resultados", exist_ok=True)

ruta_resultados = "resultados/resultados_brecha_digital.csv"

df_resultados.to_csv(ruta_resultados, index=False)

print("\nResultados guardados en:", ruta_resultados)

# ======================================================
# 16. Guardar el mejor modelo
# ======================================================

os.makedirs("modelos", exist_ok=True)

ruta_modelo = "modelos/modelo_brecha_digital.keras"

mejor_modelo.save(ruta_modelo)

print("\nMejor modelo guardado en:", ruta_modelo)

# ======================================================
# 17. Guardar variables usadas por el modelo
# ======================================================

variables_modelo = pd.DataFrame({
    "orden": range(1, len(variables_x) + 1),
    "variable": variables_x
})

ruta_variables = "resultados/variables_modelo_brecha_digital.csv"

variables_modelo.to_csv(ruta_variables, index=False)

print("\nVariables del modelo guardadas en:", ruta_variables)

# ======================================================
# 18. Gráficas de evaluación del mejor modelo
# ======================================================

import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Crear carpeta para guardar gráficas
os.makedirs("resultados/graficas", exist_ok=True)

# Recuperar historial del mejor modelo
historial_mejor = historiales_entrenamiento[mejor_modelo_nombre]

# ======================================================
# 18.1 Curva de pérdida
# ======================================================

plt.figure(figsize=(8, 5))
plt.plot(historial_mejor.history["loss"], label="Entrenamiento")
plt.plot(historial_mejor.history["val_loss"], label="Validación")
plt.title(f"{mejor_modelo_nombre} - Curva de pérdida")
plt.xlabel("Época")
plt.ylabel("Binary Crossentropy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("resultados/graficas/loss_brecha_digital.png", dpi=300)
plt.show()


# ======================================================
# 18.2 Curva de accuracy
# ======================================================

plt.figure(figsize=(8, 5))
plt.plot(historial_mejor.history["accuracy"], label="Entrenamiento")
plt.plot(historial_mejor.history["val_accuracy"], label="Validación")
plt.title(f"{mejor_modelo_nombre} - Curva de accuracy")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("resultados/graficas/accuracy_brecha_digital.png", dpi=300)
plt.show()


# ======================================================
# 18.3 Matriz de confusión en prueba
# ======================================================

y_test_prob = mejor_modelo.predict(X_test_np, verbose=0).ravel()
y_test_pred = (y_test_prob >= 0.5).astype(int)

matriz = confusion_matrix(y_test_np, y_test_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=["No bajo desempeño", "Bajo desempeño"]
)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap="Blues", values_format="d", ax=ax)
plt.title(f"{mejor_modelo_nombre} - Matriz de confusión en prueba")
plt.tight_layout()
plt.savefig("resultados/graficas/matriz_confusion_brecha_digital.png", dpi=300)
plt.show()

# ======================================================
# 19. Guardar artefactos del mejor modelo en MLflow
# ======================================================

#with mlflow.start_run(run_id=run_ids[mejor_modelo_nombre]):

 #   mlflow.log_artifact(ruta_resultados)
  #  mlflow.log_artifact(ruta_variables)
   # mlflow.log_artifact(ruta_modelo)

   # mlflow.log_artifacts(
    #    "resultados/graficas",
     #   artifact_path="graficas"
    #)

# print("\nArtefactos del mejor modelo registrados en MLflow.")