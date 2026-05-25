# ======================================================
# Modelo de clasificación - Brecha digital
# Pregunta de negocio:
# ¿El acceso a recursos tecnológicos en el hogar permite
# clasificar a los estudiantes del Cauca según su probabilidad
# de bajo desempeño en las pruebas Saber 11?
# ======================================================

import pandas as pd
import numpy as np

# Importar el archivo de limpieza que ya existe en el proyecto
import Proyecto2

# Tomar el DataFrame limpio resultante del archivo Proyecto2.py
df_limpio = Proyecto2.df

# Crear una copia para trabajar la pregunta de brecha digital
df_brecha = df_limpio.copy()

# ======================================================
# 3. Selección de variables para el modelo de brecha digital
# ======================================================

import numpy as np
import pandas as pd

variables_x = ["fami_tienecomputador","fami_tieneinternet","fami_estratovivienda","fami_personashogar"]
variable_y_base = "punt_global"

# Crear dataframe solo con las variables necesarias
df_modelo = df_brecha[variables_x + [variable_y_base]].copy()

# ======================================================
# 4. Limpieza y codificación de variables explicativas
# ======================================================

# Normalizar texto
for col in ["fami_tienecomputador", "fami_tieneinternet", "fami_personashogar"]:
    df_modelo[col] = df_modelo[col].astype(str).str.strip().str.lower()

# Convertir computador e internet a variables dicotómicas
mapa_si_no = {"si": 1, "sí": 1, "no": 0}

df_modelo["fami_tienecomputador"] = df_modelo["fami_tienecomputador"].map(mapa_si_no)
df_modelo["fami_tieneinternet"] = df_modelo["fami_tieneinternet"].map(mapa_si_no)

# Asegurar que estrato sea numérico
df_modelo["fami_estratovivienda"] = pd.to_numeric(df_modelo["fami_estratovivienda"],errors="coerce")

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

df_modelo["fami_personashogar"] = df_modelo["fami_personashogar"].map(mapa_personas)

# Asegurar que punt_global sea numérico
df_modelo["punt_global"] = pd.to_numeric(df_modelo["punt_global"],errors="coerce")

# Eliminar filas con faltantes después de la codificación
df_modelo = df_modelo.dropna().copy()

# ======================================================
# 5. Creación de la variable objetivo: bajo_desempeno
# ======================================================

# Bajo desempeño = estudiantes en el 25% inferior del puntaje global
umbral_bajo = df_modelo["punt_global"].quantile(0.25)

df_modelo["bajo_desempeno"] = np.where(df_modelo["punt_global"] <= umbral_bajo,1,0)

# ======================================================
# 6. Separar X e y
# ======================================================

X = df_modelo[["fami_tienecomputador","fami_tieneinternet","fami_estratovivienda","fami_personashogar"]].copy()
y = df_modelo["bajo_desempeno"].copy()

# Verificación
print("Umbral de bajo desempeño:", umbral_bajo)
print("\nDistribución de la variable objetivo:")
print(y.value_counts())
print("\nPrimeras filas de X:")
print(X.head())