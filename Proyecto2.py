###LIMPIEZA DE DATOS###

#Importar las librerías 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns

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
    "punt_global" : "puntaje"}, inplace=True)

#Convertir las variables necesarias a binarias o números
    #Revisar los valores que entran a cada variable y cambiarlo si es necesario
#print(df_manuela["zona"].unique())#RURAL/URBANO
df_manuela["zona"] = df_manuela["zona"].map({"RURAL": 1,"URBANO": 0}) 

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
df_manuela["edu_ma"] = df_manuela["edu_ma"].map({"No sabe": 0,
                                                 "No aplica": 0,
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
df_manuela["edu_pa"] = df_manuela["edu_pa"].map({"No sabe": 0,
                                                 "No aplica": 0,
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
df_manuela["carro"] = df_manuela["carro"].map({"No": 0, "Si": 1})


#print(df_manuela["pc"].unique())#N/S
df_manuela["pc"] = df_manuela["pc"].map({"No": 0, "Si": 1})

#print(df_manuela["internet"].unique())#N/S
df_manuela["internet"] = df_manuela["internet"].map({"No": 0, "Si": 1})

#print(df_manuela["lavadora"].unique())#N/S
df_manuela["lavadora"] = df_manuela["lavadora"].map({"No": 0, "Si": 1})

#Crear nuevas variables de análisis 
df_manuela["hacinamiento"] = df_manuela["personas"] / df_manuela["cuartos"]
df_manuela["edu_prom_padres"]= df_manuela[["edu_ma", "edu_pa"]].mean(axis=1)
df_manuela["recursos_tecnologicos"] = df_manuela["internet"] + df_manuela["pc"]
df_manuela["recursos_hogar"] = df_manuela["carro"] + df_manuela["lavadora"]

#Revisar que el .map haya quedado bien 
#print(df_manuela.isna().sum())

#Preparar los datos para el modelo de regresión lineal 
#Importar librerías para el modelo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
import mlflow

#Crear df final para el modelo 
df_mmanuela = df_manuela.drop(columns=["puntaje"])
df_mmanuela = df_mmanuela.fillna(0)

#Separar variables independientes y dependientes
y_mmanu = df_mmanuela.pop("percentil_nacional")
x_mmanu = df_mmanuela 

#Dividir datos en entrenamiento, validación y prueba
X_train_full_manu, X_test_manu, y_train_full_manu, y_test_manu = train_test_split(
    df_mmanuela, y_mmanu, test_size=0.2, random_state=42)

X_train_manu, X_valid_manu, y_train_manu, y_valid_manu = train_test_split(
    X_train_full_manu, y_train_full_manu, test_size=0.2, random_state=42)

#print(X_train_manu.shape)

#Normalizar los datos de las variables independientes
scaler_manu = StandardScaler()
X_train_manu_scaled = scaler_manu.fit_transform(X_train_manu)
X_valid_manu_scaled = scaler_manu.transform(X_valid_manu)
X_test_manu_scaled = scaler_manu.transform(X_test_manu)

#Lista para guardar resultados de los modelos
res_mmodelos = []

#Configurar experimento en MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("modelo_manuela")
nombre_exp = "modelo_base_manuela"
with mlflow.start_run(run_name=nombre_exp):
    #Registrar parámetros del modelo
    mlflow.log_param("modelo", "secuencial")
    mlflow.log_param("capas", "64, 32, 1")
    mlflow.log_param("activacion", "relu")
    mlflow.log_param("optimizador", "adam")
    mlflow.log_param("loss", "mse")
    mlflow.log_param("epochs", 20)

    #Crear modelo
    modelo_manuela = keras.models.Sequential([
        keras.layers.Input(shape=(X_train_manu_scaled.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1)
    ])

    #Compilar modelo 
    modelo_manuela.compile(
        loss = "mse",
        optimizer = "adam",
        metrics = ["mae"]
    )

    print(modelo_manuela.summary())

    #Entrenar el modelo
    hist_mmanu = modelo_manuela.fit(X_train_manu_scaled, y_train_manu, 
                                    epochs=20,
                                    validation_data=(X_valid_manu_scaled, y_valid_manu)) 
    
    #Evaluar modelo
    y_pred_manu = modelo_manuela.predict(X_test_manu_scaled).ravel()

    mae_manu = mean_absolute_error(y_test_manu, y_pred_manu)
    mse_manu = mean_squared_error(y_test_manu, y_pred_manu)
    rmse_manu = np.sqrt(mse_manu)
    r2_manu = r2_score(y_test_manu, y_pred_manu)

    #Registrar métricas en MLflow
    mlflow.log_metric("MAE", mae_manu)
    mlflow.log_metric("MSE", mse_manu)
    mlflow.log_metric("RMSE", rmse_manu)
    mlflow.log_metric("R2", r2_manu)

    #Graficar historial de pérdida
    plt.figure(figsize=(8,5))
    plt.plot(hist_mmanu.history["loss"], label="Train loss")
    plt.plot(hist_mmanu.history["val_loss"], label="Validation loss")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.title("Historial de pérdida - Modelo base")
    plt.legend()
    plt.grid(True)

    #Guardar gráfico y registrar en MLflow
    plt.tight_layout()
    plt.savefig("loss_modelo_base_manuela.png")
    mlflow.log_artifact("loss_modelo_base_manuela.png")
    plt.show() 

    #Guardar modelo
    mlflow.keras.log_model(modelo_manuela, "modelo_manuela")

    #Imprimir métricas
    print(f"MAE: {mae_manu}, MSE: {mse_manu}, R2: {r2_manu}")

    #Guardar resultados del modelo
    res_mmodelos.append({
        "modelo": nombre_exp,
        "MAE": mae_manu,
        "MSE": mse_manu,
        "RMSE": rmse_manu,
        "R2": r2_manu
    })