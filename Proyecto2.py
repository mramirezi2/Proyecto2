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
print("Total datos iniciales sin modificación:", total)

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
print("Tamaño final:",df.shape) 
print("Encabezado DataFrame:",df.head())
print("Registros iniciales:", df_original.shape[0])
print("Tomas con datos faltantes:", df_original.isna().sum())
print("Registros finales:", df.shape[0])
print("Porcentaje eliminado:", 
      round((1 - df.shape[0]/df_original.shape[0]) * 100, 2), "%")
print("Duplicados:", df_original.duplicated().sum())

###FIN DE LIMPIEZA DE DATOS###



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