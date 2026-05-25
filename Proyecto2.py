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


######################################## PREGUNTA VARIABLES DE COLEGIOS ##########################################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import mlflow

# VARIABLE OBJETIVO
print("\nDistribución de clases (variable objetivo):")
dist = df["nivel_global"].value_counts().sort_index()
for nivel, n in dist.items():
    print(f"  {nivel:<10} {n:>5} ({n/len(df)*100:.1f}%)")
 
# SELECCIÓN DE FEATURES
# Variables institucionales del colegio
features_categoricas = [
    "cole_area_ubicacion",  
    "cole_naturaleza",       
    "cole_jornada",          
    "cole_calendario",       
    "cole_caracter",         
    "cole_bilingue",         
    "cole_genero",           
]
 
print(f"\nFeatures seleccionadas: {len(features_categoricas)}")
 
# CODIFICACIÓN DE VARIABLES CATEGÓRICAS
X = pd.get_dummies(
    df[features_categoricas],
    drop_first=True,
    dtype=float
)
 
print(f"\nDimensiones después de One-Hot Encoding: {X.shape}")
print("Columnas generadas:")
for col in X.columns:
    print(f"  {col}")
 
# Codificación ordinal para la variable objetivo
label_map = {"Bajo": 0, "Medio": 1, "Alto": 2, "Muy Alto": 3}
y = df["nivel_global"].map(label_map).astype(int)
 
 
# DIVISIÓN
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.15,          
    stratify=y,
    random_state=42
)
 
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.176,         
    stratify=y_train_val,
    random_state=42
)
 
print(f"\nDivisión del dataset:")
print(f"  Train:      {len(X_train):>5} filas ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Validation: {len(X_val):>5} filas ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:       {len(X_test):>5} filas ({len(X_test)/len(X)*100:.1f}%)")
 
# Verificar estratificación
print("\nProporción de clases por split:")
for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    props = y_split.value_counts(normalize=True).sort_index()
    resumen = " | ".join([f"C{i}:{v:.2f}" for i, v in props.items()])
    print(f"  {split_name:<6}: {resumen}")
 
# NORMALIZACIÓN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
 
print(f"\nNormalización aplicada (StandardScaler):")
print(f"  Media train (primeras 3):  {X_train_scaled[:, :3].mean(axis=0).round(4)}")
print(f"  Std  train (primeras 3):   {X_train_scaled[:, :3].std(axis=0).round(4)}")
 
# PESOS DE CLASE (para manejar desbalance residual)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nPesos de clase (por si hay desbalance residual):")
label_inv = {v: k for k, v in label_map.items()}
for cls, w in class_weight_dict.items():
    print(f"  Clase {cls} ({label_inv[cls]:<10}): {w:.4f}")
 
# EXPORTACIÓN
print("RESUMEN PARA LA RED NEURONAL")
print(f"  Input shape:     {X_train_scaled.shape[1]} neuronas de entrada")
print(f"  Output shape:    4 clases (softmax)")
print(f"  Función de loss: categorical_crossentropy")
print(f"  Métrica base:    accuracy + F1-macro (por clases balanceadas)")
 
np.save("X_train.npy",  X_train_scaled)
np.save("X_val.npy",    X_val_scaled)
np.save("X_test.npy",   X_test_scaled)
np.save("y_train.npy",  y_train.values)
np.save("y_val.npy",    y_val.values)
np.save("y_test.npy",   y_test.values)
 
pd.Series(X.columns).to_csv("feature_names.csv", index=False)
 
print("\nPreprocesamiento completado")
 
### FIN PREPROCESAMIENTO ###

### MODELAMIENTO ###
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
 
# CARGAR DATOS
X_train = np.load("X_train.npy")
X_val   = np.load("X_val.npy")
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_val   = np.load("y_val.npy")
y_test  = np.load("y_test.npy")
 
feature_names = pd.read_csv("feature_names.csv").iloc[:, 0].tolist()
label_names   = ["Bajo", "Medio", "Alto", "Muy Alto"]
n_features    = X_train.shape[1]
n_classes     = 4
 
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
 
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
y_val_cat   = tf.keras.utils.to_categorical(y_val,   num_classes=n_classes)
y_test_cat  = tf.keras.utils.to_categorical(y_test,  num_classes=n_classes)
 
print(f"\nEjemplo y_train codificado (primeras 3 filas):\n{y_train_cat[:3]}")
 
cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(cw))
 
# ARQUITECTURAS
 
def modelo_A():
    tf.random.set_seed(42)
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax")
    ], name="ModeloA_Simple")
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=["accuracy"]
    )
    return model
 
def modelo_B():
    tf.random.set_seed(42)
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax")
    ], name="ModeloB_Dropout")
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=["accuracy"]
    )
    return model
 
def modelo_C():
    tf.random.set_seed(42)
    tf.keras.backend.clear_session()
    reg = tf.keras.regularizers.l2(0.001)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(n_classes, activation="softmax")
    ], name="ModeloC_L2_BN")
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=["accuracy"]
    )
    return model
 
modelos_config = {
    "ModeloA_Simple":  modelo_A,
    "ModeloB_Dropout": modelo_B,
    "ModeloC_L2_BN":   modelo_C,
}
 
# CALLBACK
EPOCHS     = 50
BATCH_SIZE = 128
 
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    verbose=1
)
 
# ENTRENAMIENTO

resultados = {}
mlflow.set_experiment("/Saber11_Cauca_Brecha_Territorial")
 
for nombre, build_fn in modelos_config.items():
    print(f"  Entrenando: {nombre}")
 
    model = build_fn()
    model.summary()
 
    with mlflow.start_run(run_name=nombre):
        mlflow.log_param("modelo", nombre)
        mlflow.log_param("epochs_max", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("total_params", model.count_params())

        history = model.fit(
            X_train, y_train_cat,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val_cat),
            class_weight=class_weight_dict,
            callbacks=[early_stop],
            verbose=1
        )

        epocas_reales = len(history.history["loss"])
    
        y_val_pred  = np.argmax(model.predict(X_val,  verbose=0), axis=1)
        y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
        val_acc  = accuracy_score(y_val,  y_val_pred)
        val_f1   = f1_score(y_val,  y_val_pred, average="macro")
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1  = f1_score(y_test, y_test_pred, average="macro")

        mlflow.log_metric("val_accuracy",  val_acc)
        mlflow.log_metric("val_f1_macro",  val_f1)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_macro", test_f1)
        mlflow.log_metric("epocas_reales", epocas_reales)
        mlflow.log_artifact(f"curva_{nombre}.png")
        mlflow.log_artifact(f"confusion_{nombre}.png")
    

        print(f"\n  Accuracy: {val_acc:.4f} | F1-macro: {val_f1:.4f}")
        print(f"  Accuracy: {test_acc:.4f} | F1-macro: {test_f1:.4f}")
        print(f"\n{classification_report(y_test, y_test_pred, target_names=label_names, digits=4)}")
 
    # Curva de aprendizaje
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{nombre} - Curvas de aprendizaje", fontsize=12, fontweight="bold")
    axes[0].plot(history.history["loss"],     label="Train", color="#2563eb")
    axes[0].plot(history.history["val_loss"], label="Val",   color="#dc2626")
    axes[0].set_title("Loss (categorical_crossentropy)")
    axes[0].set_xlabel("Época"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history.history["accuracy"],     label="Train", color="#2563eb")
    axes[1].plot(history.history["val_accuracy"], label="Val",   color="#dc2626")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Época"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"curva_{nombre}.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Curva guardada: curva_{nombre}.png")
 
    # Matriz de confusión
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title(f"{nombre} — Matriz de confusión (Test)", fontweight="bold")
    ax.set_ylabel("Real"); ax.set_xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(f"confusion_{nombre}.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Confusión guardada: confusion_{nombre}.png")
 
    model.save(f"modelo_{nombre}.keras")
    print(f"  Modelo guardado: modelo_{nombre}.keras")
 
    resultados[nombre] = {
        "val_accuracy":  val_acc,  "val_f1_macro":  val_f1,
        "test_accuracy": test_acc, "test_f1_macro": test_f1,
        "epocas":        epocas_reales,
    }
 
# COMPARACIÓN

print("COMPARACIÓN DE MODELOS")
print(f"{'Modelo':<22} {'Val Acc':>8} {'Val F1':>8} {'Test Acc':>9} {'Test F1':>8} {'Épocas':>7}")
 
mejor_nombre, mejor_f1 = None, -1
for nombre, r in resultados.items():
    marca = ""
    if r["test_f1_macro"] > mejor_f1:
        mejor_f1, mejor_nombre = r["test_f1_macro"], nombre
    print(f"{nombre:<22} {r['val_accuracy']:>8.4f} {r['val_f1_macro']:>8.4f} "
          f"{r['test_accuracy']:>9.4f} {r['test_f1_macro']:>8.4f} "
          f"{r['epocas']:>7}{marca}")
 
print(f"\n Mejor modelo: {mejor_nombre} (F1 - macro test: {mejor_f1:.4f})")
 
######################################## FIN PREGUNTA VARIABLES DE COLEGIOS ##########################################

##################################PREGUNTA SOCIODEMOGRÁFICA Y PERCENTIL NACIONAL #####################################
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
from keras.optimizers import Adam
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

#Configurar experimento en MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("modelo_manuela")
nombre_exp = "Mejor_modelo_manuela"
with mlflow.start_run(run_name=nombre_exp):
    #Registrar parámetros del modelo
    mlflow.log_param("modelo", "secuencial")
    mlflow.log_param("capas", "128, 64, 32, 1")
    mlflow.log_param("activacion", "relu")
    mlflow.log_param("optimizador", "adam")
    mlflow.log_param("loss", "mse")
    mlflow.log_param("epochs", 50)

    #Crear modelo
    modelo_manuela = keras.models.Sequential([
        keras.layers.Input(shape=(X_train_manu_scaled.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
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

    #print(modelo_manuela.summary())

    #Entrenar el modelo
    hist_mmanu = modelo_manuela.fit(X_train_manu_scaled, y_train_manu, 
                                    epochs=50,
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
    plt.title("Historial de pérdida - Modelo grande")
    plt.legend()
    plt.grid(True)

    #Guardar gráfico y registrar en MLflow
    plt.tight_layout()
    plt.savefig("loss_modelo_grande_manuela.png")
    mlflow.log_artifact("loss_modelo_grande_manuela.png")
    plt.show() 

    #Guardar modelo
    mlflow.keras.log_model(modelo_manuela, "modelo_manuela")

    #Imprimir métricas
    #print(f"MAE: {mae_manu}, MSE: {mse_manu}, R2: {r2_manu}")

#Guardar mejor modelo en Keras
#Crear carpeta de modelos
import os

os.makedirs("modelo_manuela", exist_ok=True)

#Guardar mejor modelo
modelo_manuela.save("modelo_manuela/modelo_final_manuela.keras")

print("Modelo guardado correctamente")

#Variables
variables_modelo_manu = pd.DataFrame({
    "orden": range(1, len(x_mmanu.columns) + 1),
    "variable": x_mmanu.columns
})
ruta_variables_manu = "modelo_manuela/variables_modelo_manuela.csv"
variables_modelo_manu.to_csv(ruta_variables_manu,index=False)

#Scaler
import joblib
joblib.dump(scaler_manu,"modelo_manuela/scaler_manuela.pkl")
print("Scaler guardado correctamente")

print("\nVariables del modelo guardadas en:",ruta_variables_manu)

##################################FIN PREGUNTA SOCIODEMOGRÁFICA Y PERCENTIL NACIONAL #####################################