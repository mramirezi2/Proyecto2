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
print(f"  Función de loss: categorical_crossentropy (o sparse si y es int)")
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