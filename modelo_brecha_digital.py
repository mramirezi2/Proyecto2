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

print("Dimensiones del DataFrame limpio:", df_limpio.shape)
print("Dimensiones del DataFrame de brecha digital:", df_brecha.shape)
print(df_brecha.head())