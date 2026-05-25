# ======================================================
# Tab Pregunta 1 — Brechas Territoriales
# Clasificación de nivel de desempeño por condiciones del colegio
# ======================================================
#
# IMPORTS necesarios (agregar al inicio del archivo principal):
#   import numpy as np
#   import tensorflow as tf
#   from sklearn.preprocessing import StandardScaler
#   import joblib  # solo si guardas el scaler
# ======================================================
 
import numpy as np
from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
from pathlib import Path

try:
    import tensorflow as tf
    _DIR = Path(__file__).resolve().parent

    MODELO_B = tf.keras.models.load_model(
        str(_DIR.parent / "modelos" / "modelo_ModeloB_Dropout.keras")
    )
    MODELO_CARGADO = True
except Exception as e:
    MODELO_CARGADO = False
    print(f"Advertencia: no se pudo cargar el modelo — {e}")
 
# Nombres de columnas generadas por OHE (en el mismo orden que el preprocesamiento)
FEATURE_NAMES = [
    "cole_area_ubicacion_URBANO",
    "cole_bilingue_S",
    "cole_calendario_B",
    "cole_calendario_OTRO",
    "cole_caracter_NO APLICA",
    "cole_caracter_TÉCNICO",
    "cole_caracter_TÉCNICO/ACADÉMICO",
    "cole_genero_MASCULINO",
    "cole_genero_MIXTO",
    "cole_jornada_MAÑANA",
    "cole_jornada_NOCHE",
    "cole_jornada_SABATINA",
    "cole_jornada_TARDE",
    "cole_jornada_UNICA",
    "cole_naturaleza_OFICIAL",
]
 
LABEL_NAMES  = ["Bajo", "Medio", "Alto", "Muy Alto"]
LABEL_COLORS = ["#EF4444", "#F59E0B", "#3B82F6", "#10B981"]
 
# Medias y desviaciones del scaler entrenado (StandardScaler sobre X_train)
# Si guardaste el scaler con joblib, cárgalo así:
#   SCALER = joblib.load("scaler.pkl")
# Si no, usa estos valores aproximados calculados sobre el dataset:
SCALER_MEAN = np.array([0.487, 0.019, 0.033, 0.006, 0.010, 0.197, 0.092,
                         0.019, 0.951, 0.423, 0.059, 0.028, 0.157, 0.068, 0.926])
SCALER_STD  = np.array([0.500, 0.136, 0.179, 0.077, 0.098, 0.398, 0.289,
                         0.136, 0.216, 0.494, 0.236, 0.164, 0.364, 0.252, 0.262])
 
 
def encode_input(area, bilingue, calendario, caracter, genero, jornada, naturaleza):
    """Convierte las selecciones del usuario en el vector de features OHE."""
    x = np.zeros(len(FEATURE_NAMES))
 
    mapping = {
        "cole_area_ubicacion_URBANO":      area == "URBANO",
        "cole_bilingue_S":                 bilingue == "S",
        "cole_calendario_B":               calendario == "B",
        "cole_calendario_OTRO":            calendario == "OTRO",
        "cole_caracter_NO APLICA":         caracter == "NO APLICA",
        "cole_caracter_TÉCNICO":           caracter == "TÉCNICO",
        "cole_caracter_TÉCNICO/ACADÉMICO": caracter == "TÉCNICO/ACADÉMICO",
        "cole_genero_MASCULINO":           genero == "MASCULINO",
        "cole_genero_MIXTO":               genero == "MIXTO",
        "cole_jornada_MAÑANA":             jornada == "MAÑANA",
        "cole_jornada_NOCHE":              jornada == "NOCHE",
        "cole_jornada_SABATINA":           jornada == "SABATINA",
        "cole_jornada_TARDE":              jornada == "TARDE",
        "cole_jornada_UNICA":              jornada == "UNICA",
        "cole_naturaleza_OFICIAL":         naturaleza == "OFICIAL",
    }
 
    for i, feat in enumerate(FEATURE_NAMES):
        x[i] = float(mapping.get(feat, False))
 
    # Normalizar con los parámetros del scaler
    x_scaled = (x - SCALER_MEAN) / SCALER_STD
    return x_scaled.reshape(1, -1)
 
 
# ── Estilos locales ──
COLOR_AZUL  = "#1A3980"
COLOR_GRIS  = "#F5F7FA"
COLOR_BORDE = "#D9DEE7"
 
card_style = {
    "backgroundColor": "white",
    "border": f"1px solid {COLOR_BORDE}",
    "borderRadius": "12px",
    "padding": "22px",
    "boxShadow": "0px 3px 8px rgba(0,0,0,0.08)",
    "marginBottom": "20px",
}
 
label_style = {
    "fontWeight": "bold",
    "color": COLOR_AZUL,
    "marginBottom": "4px",
    "fontSize": "13px",
}
 
dropdown_style = {
    "marginBottom": "16px",
    "fontSize": "14px",
}
 
 
def make_dropdown(id_, options, placeholder):
    return dcc.Dropdown(
        id=id_,
        options=[{"label": o, "value": o} for o in options],
        placeholder=placeholder,
        clearable=False,
        style=dropdown_style,
    )

def _metric_card(titulo, valor):
    return html.Div(
        style={
            "backgroundColor": COLOR_GRIS,
            "borderRadius": "8px",
            "padding": "14px",
            "textAlign": "center",
            "border": f"1px solid {COLOR_BORDE}",
        },
        children=[
            html.P(titulo, style={"color": "#666", "fontSize": "12px",
                                   "margin": "0 0 4px 0", "fontWeight": "bold"}),
            html.P(valor,  style={"color": COLOR_AZUL, "fontSize": "20px",
                                   "margin": "0", "fontWeight": "bold"}),
        ]
    )
 
 
 
# ── Layout del tab ──
tab_pregunta_1 = html.Div(
    style={"fontFamily": "Arial", "padding": "30px", "backgroundColor": COLOR_GRIS},
    children=[
 
        # Encabezado
        html.Div(style=card_style, children=[
            html.H2("Pregunta 3: Brechas Territoriales",
                    style={"color": COLOR_AZUL, "marginBottom": "6px"}),
            html.P(
                "¿Las condiciones institucionales del colegio permiten predecir el nivel "
                "de desempeño de un estudiante en las pruebas Saber 11?",
                style={"color": "#444", "fontSize": "15px", "marginBottom": "4px"}
            ),
            html.P(
                "Ingrese las características del colegio para obtener la clasificación "
                "esperada del estudiante: Bajo, Medio, Alto o Muy Alto.",
                style={"color": "#666", "fontSize": "13px"}
            ),
        ]),
 
        # Formulario + Resultado
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"},
            children=[
 
                # ── Columna izquierda: dropdowns ──
                html.Div(style=card_style, children=[
                    html.H3("Condiciones del colegio",
                            style={"color": COLOR_AZUL, "marginBottom": "18px"}),
 
                    html.Label("Zona", style=label_style),
                    make_dropdown("p3-area", ["RURAL", "URBANO"], "Seleccione zona"),
 
                    html.Label("Naturaleza", style=label_style),
                    make_dropdown("p3-naturaleza", ["OFICIAL", "NO OFICIAL"], "Seleccione naturaleza"),
 
                    html.Label("Jornada", style=label_style),
                    make_dropdown("p3-jornada",
                                  ["MAÑANA", "TARDE", "COMPLETA", "UNICA", "NOCHE", "SABATINA"],
                                  "Seleccione jornada"),
 
                    html.Label("Calendario", style=label_style),
                    make_dropdown("p3-calendario", ["A", "B", "OTRO"], "Seleccione calendario"),
 
                    html.Label("Carácter", style=label_style),
                    make_dropdown("p3-caracter",
                                  ["ACADÉMICO", "TÉCNICO", "TÉCNICO/ACADÉMICO", "NO APLICA"],
                                  "Seleccione carácter"),
 
                    html.Label("¿Es bilingüe?", style=label_style),
                    make_dropdown("p3-bilingue", ["S", "N"], "Seleccione opción"),
 
                    html.Label("Género del colegio", style=label_style),
                    make_dropdown("p3-genero", ["MIXTO", "FEMENINO", "MASCULINO"],
                                  "Seleccione género"),
 
                    html.Button(
                        "Predecir nivel de desempeño",
                        id="p3-btn-predecir",
                        n_clicks=0,
                        style={
                            "width": "100%",
                            "padding": "12px",
                            "backgroundColor": COLOR_AZUL,
                            "color": "white",
                            "border": "none",
                            "borderRadius": "8px",
                            "fontSize": "15px",
                            "fontWeight": "bold",
                            "cursor": "pointer",
                            "marginTop": "8px",
                        }
                    ),
                ]),
 
                # ── Columna derecha: resultado ──
                html.Div(style=card_style, children=[
                    html.H3("Resultado de la predicción",
                            style={"color": COLOR_AZUL, "marginBottom": "18px"}),
                    html.Div(id="p3-resultado"),
                ]),
            ]
        ),
 
        # ── Contexto del modelo ──
        html.Div(style=card_style, children=[
            html.H3("Sobre el modelo", style={"color": COLOR_AZUL, "marginBottom": "10px"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "16px"},
                children=[
                    _metric_card("Modelo", "ModeloB — Dropout"),
                    _metric_card("F1-macro (test)", "0.355"),
                    _metric_card("Accuracy (test)", "0.350"),
                    _metric_card("Épocas entrenadas", "37 / 150"),
                ]
            ),
            html.P(
                "Nota: Un F1-macro de 0.355 supera al clasificador aleatorio (0.25), pero indica que "
                "las condiciones institucionales por sí solas explican solo una parte del desempeño. "
                "Factores del hogar como estrato, acceso a internet y educación parental son "
                "determinantes adicionales necesarios para una predicción más precisa.",
                style={"color": "#666", "fontSize": "13px", "marginTop": "14px",
                       "borderLeft": f"3px solid {COLOR_AZUL}", "paddingLeft": "12px"}
            ),
        ]),
    ]
)
 
 

# ── Callback de predicción ──
def registrar_callbacks_p1(app):
    @app.callback(
        Output("p3-resultado", "children"),
        Input("p3-btn-predecir", "n_clicks"),
        State("p3-area",       "value"),
        State("p3-naturaleza", "value"),
        State("p3-jornada",    "value"),
        State("p3-calendario", "value"),
        State("p3-caracter",   "value"),
        State("p3-bilingue",   "value"),
        State("p3-genero",     "value"),
        prevent_initial_call=True,
    )
    def predecir(n_clicks, area, naturaleza, jornada, calendario, caracter, bilingue, genero):
        # Validar que todos los campos estén completos
        campos = [area, naturaleza, jornada, calendario, caracter, bilingue, genero]
        if any(c is None for c in campos):
            return html.P("⚠ Complete todos los campos antes de predecir.",
                          style={"color": "#F59E0B", "fontWeight": "bold"})
 
        if not MODELO_CARGADO:
            return html.P("⚠ El modelo no está disponible. Verifique que modelo_ModeloB_Dropout.keras "
                          "esté en el directorio del proyecto.",
                          style={"color": "#EF4444"})
 
        # Codificar y predecir
        x = encode_input(area, bilingue, calendario, caracter, genero, jornada, naturaleza)
        probs = MODELO_B.predict(x, verbose=0)[0]
        clase = int(np.argmax(probs))
        nivel = LABEL_NAMES[clase]
        color = LABEL_COLORS[clase]
 
        # Gráfico de barras de probabilidades
        fig = go.Figure(go.Bar(
            x=LABEL_NAMES,
            y=[float(p) for p in probs],
            marker_color=LABEL_COLORS,
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside",
        ))
        fig.update_layout(
            title="Probabilidad por nivel",
            yaxis=dict(range=[0, 1], tickformat=".0%", title="Probabilidad"),
            xaxis_title="Nivel de desempeño",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=40, b=40, l=40, r=20),
            height=280,
            showlegend=False,
        )
 
        return html.Div([
            # Resultado principal
            html.Div(
                style={
                    "backgroundColor": color,
                    "borderRadius": "12px",
                    "padding": "20px",
                    "textAlign": "center",
                    "marginBottom": "20px",
                },
                children=[
                    html.P("Nivel de desempeño predicho",
                           style={"color": "white", "margin": "0", "fontSize": "13px",
                                  "fontWeight": "bold", "opacity": "0.85"}),
                    html.H1(nivel, style={"color": "white", "margin": "6px 0",
                                          "fontSize": "42px", "fontWeight": "900"}),
                    html.P(f"Confianza: {max(probs)*100:.1f}%",
                           style={"color": "white", "margin": "0", "fontSize": "14px",
                                  "opacity": "0.9"}),
                ]
            ),
 
            # Gráfico
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
 
            # Interpretación
            html.Div(
                style={"backgroundColor": "#F0F4FF", "borderRadius": "8px",
                       "padding": "12px", "marginTop": "12px"},
                children=[
                    html.P("Interpretación:", style={"fontWeight": "bold",
                                                      "color": COLOR_AZUL, "margin": "0 0 4px 0"}),
                    html.P(
                        _interpretacion(nivel, area, naturaleza, jornada),
                        style={"color": "#444", "fontSize": "13px", "margin": "0"}
                    )
                ]
            )
        ])
 
 
def _interpretacion(nivel, area, naturaleza, jornada):
    textos = {
        "Bajo":     f"Un estudiante en un colegio {area.lower()} de naturaleza {naturaleza.lower()} "
                    f"con jornada {jornada.lower()} tiene alta probabilidad de pertenecer al cuartil "
                    f"inferior de desempeño. Se recomienda revisar condiciones de apoyo pedagógico.",
        "Medio":    f"El perfil institucional ({area.lower()}, {naturaleza.lower()}, {jornada.lower()}) "
                    f"se asocia a un desempeño medio. Existe margen de mejora con intervenciones focalizadas.",
        "Alto":     f"Las condiciones del colegio ({area.lower()}, {naturaleza.lower()}, {jornada.lower()}) "
                    f"favorecen un buen desempeño. Se recomienda mantener y fortalecer las prácticas actuales.",
        "Muy Alto": f"Este perfil institucional ({area.lower()}, {naturaleza.lower()}, {jornada.lower()}) "
                    f"se asocia al cuartil superior de desempeño en el Cauca. "
                    f"Puede servir como modelo de referencia para otras instituciones.",
    }
    return textos.get(nivel, "")
 