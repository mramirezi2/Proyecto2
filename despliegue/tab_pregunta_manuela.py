
# ======================================================
# Tab Pregunta Manuela — Percentil Nacional Esperado
# Regresión: características sociodemográficas → percentil
 
import numpy as np
import joblib
from pathlib import Path
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
 
_DIR = Path(__file__).resolve().parent
_MODELOS = _DIR.parent / "modelos"
 
# ── Cargar modelo y scaler ──
try:
    import tensorflow as tf
    MODELO_MANU = tf.keras.models.load_model(str(_MODELOS / "modelo_final_manuela.keras"))
    MODELO_MANU_CARGADO = True
except Exception as e:
    MODELO_MANU_CARGADO = False
    print(f"Advertencia modelo Manuela: {e}")
 
try:
    SCALER_MANU = joblib.load(str(_MODELOS / "scaler_manuela.pkl"))
    SCALER_MANU_CARGADO = True
except Exception as e:
    SCALER_MANU_CARGADO = False
    print(f"Advertencia scaler Manuela: {e}")
 
# Orden de columnas del modelo (igual que x_mmanu en el código de Manuela)
COLUMNAS_MANU = [
    "zona", "bilingue", "naturaleza", "cuartos", "edu_ma", "edu_pa",
    "estrato", "personas", "carro", "pc", "internet", "lavadora",
    "percentil_nacional",  # se elimina antes de predecir
    "hacinamiento", "edu_prom_padres", "recursos_tecnologicos", "recursos_hogar"
]
 
COLUMNAS_X = [
    "zona", "bilingue", "naturaleza", "cuartos", "edu_ma", "edu_pa",
    "estrato", "personas", "carro", "pc", "internet", "lavadora",
    "hacinamiento", "edu_prom_padres", "recursos_tecnologicos", "recursos_hogar"
]
 
EDU_MAP = {
    "No sabe / No aplica": 0, "Ninguno": 0,
    "Primaria incompleta": 1, "Primaria completa": 2,
    "Secundaria incompleta": 3, "Secundaria completa": 4,
    "Técnica o tecnológica incompleta": 5, "Técnica o tecnológica completa": 6,
    "Educación profesional incompleta": 7, "Educación profesional completa": 8,
    "Postgrado": 9
}
 
PERSONAS_MAP = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9 o más": 9
}
 
CUARTOS_MAP = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6 o más": 6
}
 
# ── Estilos ──
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
 
dropdown_style = {"marginBottom": "14px", "fontSize": "14px"}
 
 
def _drop(id_, options, placeholder):
    return dcc.Dropdown(
        id=id_,
        options=[{"label": o, "value": o} for o in options],
        placeholder=placeholder,
        clearable=False,
        style=dropdown_style
    )
 
 
def _metric_card_manu(titulo, valor, color=COLOR_AZUL):
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
            html.P(valor,  style={"color": color, "fontSize": "20px",
                                   "margin": "0", "fontWeight": "bold"}),
        ]
    )
 
 
# ── Layout ──
tab_pregunta_manuela = html.Div(
    style={"fontFamily": "Arial", "padding": "30px", "backgroundColor": COLOR_GRIS},
    children=[
 
        # Encabezado
        html.Div(style=card_style, children=[
            html.Div(
                style={"display": "flex", "gap": "30px", "alignItems": "flex-start"},
                children=[
                    html.Div(style={"flex": "2"}, children=[
                        html.H2("Pregunta: Percentil Nacional Esperado",
                                style={"color": COLOR_AZUL, "marginTop": "0"}),
                        html.P(
                            "Las condiciones socioeconómicas del hogar constituyen uno de los factores "
                            "más influyentes en el desempeño académico. Esta herramienta estima el "
                            "percentil nacional esperado de un estudiante del Cauca en las pruebas "
                            "Saber 11 a partir de variables como el estrato, el nivel educativo de "
                            "los padres, el acceso a recursos tecnológicos y las condiciones de "
                            "habitabilidad del hogar.",
                            style={"textAlign": "justify", "lineHeight": "1.8",
                                   "fontSize": "14px", "color": "#444"}
                        ),
                        html.P(
                            "Ingrese las características del hogar y obtenga el percentil estimado.",
                            style={"fontSize": "13px", "color": "#666", "fontStyle": "italic"}
                        ),
                    ]),
                    html.Div(
                        style={
                            "flex": "1", "backgroundColor": "#EFF6FF",
                            "borderRadius": "10px", "padding": "20px",
                            "textAlign": "center", "border": "1px solid #BFDBFE"
                        },
                        children=[
                            html.P("Estudiantes analizados",
                                   style={"color": "#666", "fontSize": "12px", "margin": "0"}),
                            html.H2("3.871", style={"color": COLOR_AZUL, "margin": "6px 0"}),
                            html.P("en el departamento del Cauca",
                                   style={"color": "#666", "fontSize": "12px", "margin": "0 0 16px 0"}),
                            html.Hr(style={"borderColor": "#BFDBFE"}),
                            html.P("Variables del hogar",
                                   style={"color": "#666", "fontSize": "12px", "margin": "8px 0 0 0"}),
                            html.H3("12", style={"color": COLOR_AZUL, "margin": "4px 0"}),
                            html.P("estrato · educación padres · recursos tecnológicos · hacinamiento",
                                   style={"color": "#888", "fontSize": "11px", "margin": "0"}),
                        ]
                    )
                ]
            )
        ]),
 
        # Formulario + Resultado
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"},
            children=[
 
                # Columna izquierda
                html.Div(style=card_style, children=[
                    html.H3("Características del hogar",
                            style={"color": COLOR_AZUL, "marginBottom": "18px"}),
 
                    # Fila 1
                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
                        html.Div([html.Label("Zona del colegio", style=label_style),
                                  _drop("manu-zona", ["RURAL", "URBANO"], "Zona")]),
                        html.Div([html.Label("Colegio bilingüe", style=label_style),
                                  _drop("manu-bilingue", ["No", "Sí"], "Bilingüe")]),
                    ]),
 
                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
                        html.Div([html.Label("Naturaleza del colegio", style=label_style),
                                  _drop("manu-naturaleza", ["OFICIAL", "NO OFICIAL"], "Naturaleza")]),
                        html.Div([html.Label("Estrato de la vivienda", style=label_style),
                                  _drop("manu-estrato", ["1", "2", "3", "4", "5", "6"], "Estrato")]),
                    ]),
 
                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
                        html.Div([html.Label("Personas en el hogar", style=label_style),
                                  _drop("manu-personas", list(PERSONAS_MAP.keys()), "Personas")]),
                        html.Div([html.Label("Cuartos en el hogar", style=label_style),
                                  _drop("manu-cuartos", list(CUARTOS_MAP.keys()), "Cuartos")]),
                    ]),
 
                    html.Label("Educación de la madre", style=label_style),
                    _drop("manu-edu-ma", list(EDU_MAP.keys()), "Nivel educativo madre"),
 
                    html.Label("Educación del padre", style=label_style),
                    _drop("manu-edu-pa", list(EDU_MAP.keys()), "Nivel educativo padre"),
 
                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
                        html.Div([html.Label("¿Tiene computador?", style=label_style),
                                  _drop("manu-pc", ["No", "Sí"], "Computador")]),
                        html.Div([html.Label("¿Tiene internet?", style=label_style),
                                  _drop("manu-internet", ["No", "Sí"], "Internet")]),
                    ]),
 
                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
                        html.Div([html.Label("¿Tiene automóvil?", style=label_style),
                                  _drop("manu-carro", ["No", "Sí"], "Automóvil")]),
                        html.Div([html.Label("¿Tiene lavadora?", style=label_style),
                                  _drop("manu-lavadora", ["No", "Sí"], "Lavadora")]),
                    ]),
 
                    html.Button(
                        "Estimar percentil nacional",
                        id="manu-btn-predecir",
                        n_clicks=0,
                        style={
                            "width": "100%", "padding": "12px",
                            "backgroundColor": COLOR_AZUL, "color": "white",
                            "border": "none", "borderRadius": "8px",
                            "fontSize": "15px", "fontWeight": "bold",
                            "cursor": "pointer", "marginTop": "8px",
                        }
                    ),
                ]),
 
                # Columna derecha — resultado
                html.Div(style=card_style, children=[
                    html.H3("Resultado de la estimación",
                            style={"color": COLOR_AZUL, "marginBottom": "18px"}),
                    html.Div(id="manu-resultado"),
                ]),
            ]
        ),
 
        # Sobre el modelo
        html.Div(style=card_style, children=[
            html.H3("Sobre el modelo", style={"color": COLOR_AZUL, "marginBottom": "10px"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "16px"},
                children=[
                    _metric_card_manu("Tipo de modelo", "Regresión"),
                    _metric_card_manu("Arquitectura", "128 → 64 → 32 → 1"),
                    _metric_card_manu("Variable objetivo", "Percentil nacional"),
                    _metric_card_manu("Optimizador", "Adam"),
                ]
            ),
            html.P(
                "El modelo estima el percentil nacional esperado del estudiante (0-100) a partir de "
                "variables socioeconómicas del hogar. Un percentil de 70 indica que el estudiante "
                "superaría al 70% de los estudiantes a nivel nacional.",
                style={"color": "#666", "fontSize": "13px", "marginTop": "14px",
                       "borderLeft": f"3px solid {COLOR_AZUL}", "paddingLeft": "12px"}
            ),
        ]),
    ]
)
 
 
# ── Callback ──
def registrar_callbacks_manuela(app):
    @app.callback(
        Output("manu-resultado", "children"),
        Input("manu-btn-predecir", "n_clicks"),
        State("manu-zona",      "value"),
        State("manu-bilingue",  "value"),
        State("manu-naturaleza","value"),
        State("manu-estrato",   "value"),
        State("manu-personas",  "value"),
        State("manu-cuartos",   "value"),
        State("manu-edu-ma",    "value"),
        State("manu-edu-pa",    "value"),
        State("manu-pc",        "value"),
        State("manu-internet",  "value"),
        State("manu-carro",     "value"),
        State("manu-lavadora",  "value"),
        prevent_initial_call=True,
    )
    def predecir_manu(n_clicks, zona, bilingue, naturaleza, estrato,
                      personas, cuartos, edu_ma, edu_pa, pc, internet, carro, lavadora):
 
        campos = [zona, bilingue, naturaleza, estrato, personas,
                  cuartos, edu_ma, edu_pa, pc, internet, carro, lavadora]
        if any(c is None for c in campos):
            return html.P("⚠ Complete todos los campos antes de estimar.",
                          style={"color": "#F59E0B", "fontWeight": "bold"})
 
        if not MODELO_MANU_CARGADO:
            return html.P("⚠ El modelo no está disponible.",
                          style={"color": "#EF4444"})
 
        if not SCALER_MANU_CARGADO:
            return html.P("⚠ El scaler no está disponible. Verifique que scaler_manuela.pkl "
                          "esté en la carpeta modelos/.",
                          style={"color": "#EF4444"})
 
        # Codificar
        zona_v      = 1 if zona == "RURAL" else 0
        bilingue_v  = 1 if bilingue == "Sí" else 0
        nat_v       = 0 if naturaleza == "OFICIAL" else 1
        estrato_v   = float(estrato)
        personas_v  = PERSONAS_MAP.get(personas, 5)
        cuartos_v   = CUARTOS_MAP.get(cuartos, 3)
        edu_ma_v    = EDU_MAP.get(edu_ma, 0)
        edu_pa_v    = EDU_MAP.get(edu_pa, 0)
        pc_v        = 1 if pc == "Sí" else 0
        internet_v  = 1 if internet == "Sí" else 0
        carro_v     = 1 if carro == "Sí" else 0
        lavadora_v  = 1 if lavadora == "Sí" else 0
 
        # Variables derivadas
        hacinamiento        = personas_v / max(cuartos_v, 1)
        edu_prom_padres     = (edu_ma_v + edu_pa_v) / 2
        recursos_tecnologicos = internet_v + pc_v
        recursos_hogar      = carro_v + lavadora_v
 
        x = np.array([[zona_v, bilingue_v, nat_v, cuartos_v, edu_ma_v, edu_pa_v,
                        estrato_v, personas_v, carro_v, pc_v, internet_v, lavadora_v,
                        hacinamiento, edu_prom_padres, recursos_tecnologicos, recursos_hogar]])
 
        x_scaled    = SCALER_MANU.transform(x)
        percentil   = float(MODELO_MANU.predict(x_scaled, verbose=0)[0][0])
        percentil   = max(0, min(100, percentil))
 
        # Color según percentil
        if percentil < 25:
            color = "#EF4444"
            etiqueta = "Bajo"
        elif percentil < 50:
            color = "#F59E0B"
            etiqueta = "Medio-bajo"
        elif percentil < 75:
            color = "#3B82F6"
            etiqueta = "Medio-alto"
        else:
            color = "#10B981"
            etiqueta = "Alto"
 
        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percentil,
            number={"suffix": "°", "font": {"size": 36, "color": color}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 25],  "color": "#FEE2E2"},
                    {"range": [25, 50], "color": "#FEF3C7"},
                    {"range": [50, 75], "color": "#DBEAFE"},
                    {"range": [75, 100],"color": "#D1FAE5"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.75,
                    "value": percentil
                }
            },
            title={"text": "Percentil nacional estimado", "font": {"size": 14}}
        ))
        fig.update_layout(
            height=260,
            margin=dict(t=40, b=20, l=20, r=20),
            paper_bgcolor="white"
        )
 
        return html.Div([
            # Badge
            html.Div(
                style={
                    "backgroundColor": color, "borderRadius": "12px",
                    "padding": "14px 20px", "textAlign": "center", "marginBottom": "16px",
                },
                children=[
                    html.P("Nivel estimado", style={"color": "white", "margin": "0",
                                                     "fontSize": "12px", "opacity": "0.85"}),
                    html.H2(etiqueta, style={"color": "white", "margin": "4px 0",
                                             "fontSize": "32px", "fontWeight": "900"}),
                ]
            ),
 
            # Gauge
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
 
            # Variables derivadas calculadas
            html.Div(
                style={"backgroundColor": "#F0F4FF", "borderRadius": "8px",
                       "padding": "12px", "marginTop": "12px"},
                children=[
                    html.P("Variables calculadas automáticamente:",
                           style={"fontWeight": "bold", "color": COLOR_AZUL,
                                  "margin": "0 0 6px 0", "fontSize": "13px"}),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "6px"},
                        children=[
                            html.P(f"Hacinamiento: {hacinamiento:.2f}",
                                   style={"margin": "0", "fontSize": "12px", "color": "#444"}),
                            html.P(f"Educ. promedio padres: {edu_prom_padres:.1f}/9",
                                   style={"margin": "0", "fontSize": "12px", "color": "#444"}),
                            html.P(f"Recursos tecnológicos: {int(recursos_tecnologicos)}/2",
                                   style={"margin": "0", "fontSize": "12px", "color": "#444"}),
                            html.P(f"Recursos del hogar: {int(recursos_hogar)}/2",
                                   style={"margin": "0", "fontSize": "12px", "color": "#444"}),
                        ]
                    )
                ]
            )
        ])