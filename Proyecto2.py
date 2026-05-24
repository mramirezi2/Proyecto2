###LIMPIEZA DE DATOS###

#Importar las librerías 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Cargar el dataset
df = pd.read_csv("datos_cauca.csv")
df_original = df.copy()
df.columns = df.columns.str.lower().str.strip()

total = len(df)
#print("Total datos iniciales sin modificación:", total)

#Solo permitir los que tienen consentimiento
df = df[df["estu_estadoinvestigacion"] == "PUBLICAR"]

# Eliminar faltantes y duplicados
df = df.drop_duplicates()
df = df.dropna(subset=[
    "punt_ingles",
    "punt_matematicas",
    "punt_sociales_ciudadanas",
    "punt_c_naturales",
    "punt_lectura_critica"
])

#Ajustar el tipo de dato de estrato
df["fami_estratovivienda"] = df["fami_estratovivienda"].str.extract(r"(\d)").astype(float)

# Calcular puntaje de icfes
df["punt_global"] = df["punt_global"].fillna(
    ((df[
        ["punt_matematicas",
         "punt_sociales_ciudadanas",
         "punt_c_naturales",
         "punt_lectura_critica"]
     ].sum(axis=1)*3 + df["punt_ingles"] )/ 13) * 5
)

#Crear columnas para cada área
cols_areas = [
    "punt_ingles",
    "punt_matematicas",
    "punt_sociales_ciudadanas",
    "punt_c_naturales",
    "punt_lectura_critica"
]

# Validar rangos y tipo de datos
for col in cols_areas:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df[col] >= 0) & (df[col] <= 100)]

df["punt_global"] = pd.to_numeric(df["punt_global"], errors="coerce")
df = df[(df["punt_global"] >= 0) & (df["punt_global"] <= 500)]

# Nueva variable de clasificación cualitativa por cuartiles
df["nivel_global"] = pd.qcut(
    df["punt_global"],
    4,
    labels=["Bajo", "Medio", "Alto", "Muy Alto"],
    duplicates="drop"
)

# Nueva variable promedio por áreas
df["promedio_areas"] = df[cols_areas].mean(axis=1)

# Reporte de impacto
#print("Tamaño final:",df.shape) 
#print("Encabezado DataFrame:",df.head())
#print("Registros iniciales:", df_original.shape[0])
#print("Tomas con datos faltantes:", df_original.isna().sum())
#print("Registros finales:", df.shape[0])
#print("Porcentaje eliminado:", 
      #round((1 - df.shape[0]/df_original.shape[0]) * 100, 2), "%")
#print("Duplicados:", df_original.duplicated().sum())

###FIN DE LIMPIEZA DE DATOS###

###MODELO DE REGRESIÓN LINEAL - MANUELA###

#Crear una lista de variables para el modelo de Manuela 
#print(df.columns)
variables_manuela = [
    "cole_area_ubicacion",
    "cole_bilingue",
    "cole_naturaleza",
    "fami_cuartoshogar",
    "fami_educacionmadre",
    "fami_educacionpadre",
    "fami_estratovivienda",
    "fami_personashogar",
    "fami_tieneautomovil",
    "fami_tienecomputador",
    "fami_tieneinternet",
    "fami_tienelavadora",
    "punt_ingles",
    "punt_matematicas",
    "punt_sociales_ciudadanas",
    "punt_c_naturales",
    "punt_lectura_critica",
    "punt_global"
]

#Crear un nuevo df con las variables para el modelo de Manuela 
df_manuela = df[variables_manuela].copy()
#print(type(df_manuela))
#print(df_manuela.head())

#Añadir variable de percentil nacional 
df_manuela["percentil_nacional"] = df_manuela["punt_global"].rank(pct=True) * 100
#print(df_manuela[["punt_global", "percentil_nacional"]].head())

#Cambiar el nombre de los headers 
df_manuela.rename(columns={
    "cole_area_ubicacion" : "zona",
    "cole_bilingue" : "bilingue",
    "cole_naturaleza" : "naturaleza",
    "fami_cuartoshogar" : "cuartos",
    "fami_educacionmadre" : "edu_ma",
    "fami_educacionpadre" : "edu_pa",
    "fami_estratovivienda" : "estrato",
    "fami_personashogar" : "personas",
    "fami_tieneautomovil" : "carro",
    "fami_tienecomputador" : "pc",
    "fami_tieneinternet" : "internet",
    "fami_tienelavadora" : "lavadora",
    "punt_ingles" : "ingles",
    "punt_matematicas" : "matematicas",
    "punt_sociales_ciudadanas" : "sociales",
    "punt_c_naturales" : "naturales",
    "punt_lectura_critica" : "lectura",
    "punt_global" : "global"}, inplace=True)

#Convertir las variables necesarias a binarias o números
    #Revisar los valores que entran a cada variable y cambiarlo si es necesario
#print(df_manuela["zona"].unique())#RURAL/URBANO
df_manuela["zona"] = df_manuela["zona"].map({"RURAL": 0,"URBANO": 1}) 

#print(df_manuela["bilingue"].unique())#N/S
df_manuela["bilingue"] = df_manuela["bilingue"].map({"N": 0, "S": 1})

#print(df_manuela["naturaleza"].unique())#OFICIAL/NO OFICIAL
df_manuela["naturaleza"] = df_manuela["naturaleza"].map({"OFICIAL": 0, "NO OFICIAL": 1})

#print(df_manuela["cuartos"].unique())#STR(1,2,3,4,5,6,7,8,9,10+,6+)
#print(df_manuela["cuartos"].value_counts())
#Como hay bajos valores para 6+, se agrupan con 6 para evitar outliers
df_manuela["cuartos"] = df_manuela["cuartos"].map({"Uno": 1,
                                                    "Dos": 2,
                                                    "Tres": 3,
                                                    "Cuatro": 4,
                                                    "Cinco": 5,
                                                    "Seis": 6,
                                                    "Seis o mas": 6,
                                                    "Siete": 6,
                                                    "Ocho": 6,
                                                    "Nueve": 6,
                                                    "Diez o más": 6
                                                })

#print(df_manuela["edu_ma"].unique())#12 VALORES DISTINTOS
df_manuela["edu_ma"] = df_manuela["edu_ma"].map({"No sabe": np.nan,
                                                 "No aplica": np.nan,
                                                 "Ninguno": 0,
                                                 "Primaria incompleta": 1,
                                                 "Primaria completa": 2,
                                                 "Secundaria (Bachillerato) incompleta": 3,
                                                 "Secundaria (Bachillerato) completa": 4,
                                                 "Técnica o tecnológica incompleta": 5,
                                                 "Técnica o tecnológica completa": 6,
                                                 "Educación profesional incompleta": 7,
                                                 "Educación profesional completa": 8,
                                                 "Postgrado": 9,})

#print(df_manuela["edu_pa"].unique())# 12 VALORES DISTINTOS
df_manuela["edu_pa"] = df_manuela["edu_pa"].map({"No sabe": np.nan,
                                                 "No aplica": np.nan,
                                                 "Ninguno": 0,
                                                 "Primaria incompleta": 1,
                                                 "Primaria completa": 2,
                                                 "Secundaria (Bachillerato) incompleta": 3,
                                                 "Secundaria (Bachillerato) completa": 4,
                                                 "Técnica o tecnológica incompleta": 5,
                                                 "Técnica o tecnológica completa": 6,
                                                 "Educación profesional incompleta": 7,
                                                 "Educación profesional completa": 8,
                                                 "Postgrado": 9,})

#print(df_manuela["estrato"].unique())#1,2,3,4,5,6
df_manuela["estrato"] = df_manuela["estrato"].fillna(0)#Los NaN no son aleatorios, por ende se conservan como 0 

#print(df_manuela["personas"].unique())#1,2,3,4,5,6,7,8,9,10+ (tiene muchos valores intermedios)
#print(df_manuela["personas"].value_counts())
df_manuela["personas"] = df_manuela["personas"].map({"Una": 1,
                                                    "Dos": 2,
                                                    "Tres": 3,
                                                    "Cuatro": 4,
                                                    "Cinco": 5,
                                                    "Seis": 6,
                                                    "Siete": 7,
                                                    "Ocho": 8,
                                                    "Nueve": 9,
                                                    "Diez": 10,
                                                    "Once": 11,
                                                    "1 a 2": 2,
                                                    "3 a 4": 4,
                                                    "5 a 6": 6,
                                                    "7 a 8": 8,
                                                    "9 o más": 9,
                                                    "Doce o más": 12})

#print(df_manuela["carro"].unique())#N/S
df_manuela["carro"] = df_manuela["carro"].map({"N": 0, "S": 1})


#print(df_manuela["pc"].unique())#N/S
df_manuela["pc"] = df_manuela["pc"].map({"N": 0, "S": 1})

#print(df_manuela["internet"].unique())#N/S
df_manuela["internet"] = df_manuela["internet"].map({"N": 0, "S": 1})

#print(df_manuela["lavadora"].unique())#N/S
df_manuela["lavadora"] = df_manuela["lavadora"].map({"N": 0, "S": 1})
