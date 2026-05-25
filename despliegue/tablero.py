# ======================================================
# Tablero Proyecto 2 - Analítica de Datos
# Modelos predictivos Saber 11 - Cauca
# ======================================================

import dash
from dash import html, dcc, Input, Output, State
from pathlib import Path
import numpy as np
import tensorflow as tf

# ======================================================
# Configuración inicial
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
app = dash.Dash(
    __name__,
    assets_folder=str(BASE_DIR / "assets"),
    suppress_callback_exceptions=True)
server = app.server
PROJECT_DIR = BASE_DIR.parent
MODEL_BRECHA_PATH = PROJECT_DIR / "modelos" / "modelo_brecha_digital.keras"

modelo_brecha = tf.keras.models.load_model(MODEL_BRECHA_PATH)

# ======================================================
# Estilos generales
# ======================================================

COLOR_AZUL = "#1A3980"
COLOR_GRIS = "#F5F7FA"
COLOR_BORDE = "#D9DEE7"

card_style = {
    "backgroundColor": "white",
    "border": f"1px solid {COLOR_BORDE}",
    "borderRadius": "12px",
    "padding": "22px",
    "boxShadow": "0px 3px 8px rgba(0,0,0,0.08)",
    "marginBottom": "20px"}

button_style = {
    "backgroundColor": COLOR_AZUL,
    "color": "white",
    "border": "none",
    "borderRadius": "8px",
    "padding": "12px 24px",
    "fontWeight": "bold",
    "cursor": "pointer",
    "width": "100%",
    "marginTop": "10px"
}

# ======================================================
# Header con logos
# ======================================================

header = html.Div(
    style={
        "fontFamily": "Arial",
        "padding": "18px 28px",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "space-between",
        "borderBottom": f"2px solid {COLOR_BORDE}",
        "backgroundColor": "white"
    },
    children=[
        html.Div(
            style={
                "display": "flex",
                "gap": "18px",
                "alignItems": "center"
            },
            children=[
                html.Img(
                    src="/assets/logo_andes.png",
                    style={"height": "70px"}
                ),
                html.Img(
                    src="/assets/logo_cauca.png",
                    style={"height": "70px"}
                )
            ]
        ),

        html.H2(
            "Tablero de Análisis Educativo - Cauca",
            style={
                "color": COLOR_AZUL,
                "margin": "0px",
                "textAlign": "center"
            }
        )
    ]
)


# ======================================================
# Tab 0 - Portada
# ======================================================

tab_portada = html.Div(
    style={
        "fontFamily": "Arial",
        "padding": "30px",
        "backgroundColor": COLOR_GRIS
    },
    children=[
        html.Div(
            style=card_style,
            children=[
                html.H1(
                    "Análisis predictivo del desempeño académico en el Cauca",
                    style={
                        "color": COLOR_AZUL,
                        "textAlign": "center",
                        "marginBottom": "25px"
                    }
                ),

                html.Div(
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "gap": "40px",
                        "flexWrap": "wrap"
                    },
                    children=[
                        html.Div(
                            style={
                                "maxWidth": "720px",
                                "textAlign": "justify",
                                "lineHeight": "1.7",
                                "fontSize": "16px"
                            },
                            children=[
                                html.P(
                                    "Este tablero presenta una herramienta de analítica desarrollada "
                                    "para la Secretaría de Educación y Cultura del Cauca. Su objetivo es apoyar "
                                    "la identificación de factores asociados al desempeño académico de los "
                                    "estudiantes en las Pruebas Saber 11."
                                ),
                                html.P(
                                    "A partir de modelos predictivos, la herramienta permite explorar cómo las condiciones "
                                    "institucionales, la brecha digital, el contexto del hogar y las características "
                                    "sociodemográficas se relacionan con el desempeño esperado de los estudiantes del departamento."
                                )
                            ]
                        ),
                        html.Img(
                            src="/assets/foto_portada.jpg",
                            style={
                                "width": "330px",
                                "height": "230px",
                                "objectFit": "cover",
                                "borderRadius": "14px",
                                "boxShadow": "0px 4px 10px rgba(0,0,0,0.15)"
                            }
                        )

                    ]
                )
            ]
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
                "gap": "20px"
            },
            children=[
                html.Div(
                    style=card_style,
                    children=[
                        html.H3(
                            "¿Qué encontrará en el tablero?",
                            style={"color": COLOR_AZUL}
                        ),
                        html.P(
                            "Predicciones, visualizaciones y resultados de modelos asociados "
                            "al desempeño académico de los estudiantes del Cauca con énfasis  "
                            "en condiciones institucionales, brecha digital y características sociodemográficas."
                        )
                    ]
                ),

                html.Div(
                    style=card_style,
                    children=[
                        html.H3(
                            "Preguntas de negocio",
                            style={"color": COLOR_AZUL}
                        ),
                        html.P(
                            "Cada pestaña responde una pregunta específica:  desempeño según condiciones institucionales,"
                            "riesgo de bajo desempeño asociado a brecha digital y percentil nacional esperado según "
                            "características sociodemográficas."
                        )
                    ]
                ),

                html.Div(
                    style=card_style,
                    children=[
                        html.H3(
                            "Usuario final",
                            style={"color": COLOR_AZUL}
                        ),
                        html.P(
                            "Secretaría de Educación y Cultura del Cauca, equipos técnicos "
                            "y tomadores de decisiones del sector educativo."
                        )
                    ]
                )
            ]
        )
    ]
)


# ======================================================
# Tabs para las preguntas
# ======================================================

tab_pregunta_1 = html.Div(
    style={
        "fontFamily": "Arial",
        "padding": "30px",
        "backgroundColor": COLOR_GRIS
    },
    children=[
        html.H2("Desempeño institucional", style={"color": COLOR_AZUL}),
        html.Div(
            style=card_style,
            children=[
                html.H3("Descripción del problema", style={"color": COLOR_AZUL}),
                html.P(
                    "En esta sección se integrará el primer modelo predictivo. "
                    "Aquí se incluirán los campos de entrada, el resultado del modelo "
                    "y el hallazgo principal asociado a la pregunta de negocio."
                )
            ]
        )
    ]
)

# ======================================================
# Tab 2 - Brecha digital y hogar
# ======================================================

tab_pregunta_2 = html.Div(
    style={
        "fontFamily": "Arial",
        "padding": "30px",
        "backgroundColor": COLOR_GRIS
    },
    children=[
        html.H2(
            "Brecha digital y hogar",
            style={"color": COLOR_AZUL, "marginBottom": "20px"}
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1.35fr",
                "gap": "20px",
                "alignItems": "start"
            },
            children=[

                # ======================================================
                # Columna izquierda: descripción + inputs
                # ======================================================
                html.Div(
                    children=[

                        # 1. Descripción del problema
                        html.Div(
                            style=card_style,
                            children=[
                                html.H3(
                                    "1. Descripción del problema",
                                    style={"color": COLOR_AZUL}
                                ),

                                html.P(
                                    "Esta sección analiza si las condiciones de brecha digital y del hogar "
                                    "permiten clasificar a los estudiantes del Cauca según su probabilidad "
                                    "de presentar bajo desempeño en las pruebas Saber 11. Para esto, se requiere si " 
                                    "el estudiante tiene acceso a computador, internet, el estrato de "
                                    "la vivienda y número de personas en el hogar, con el fin de apoyar la "
                                    "identificación temprana de perfiles estudiantiles que podrían requerir "
                                    "mayor acompañamiento académico o tecnológico.",
                                    style={
                                        "textAlign": "justify",
                                        "lineHeight": "1.6",
                                        "fontSize": "15px"
                                    }
                                ),

                                html.Div(
                                    style={
                                        "display": "flex",
                                        "justifyContent": "center",
                                        "marginTop": "18px"
                                    },
                                    children=[
                                        html.Img(
                                            src="/assets/brecha.jpg",
                                            style={
                                                "width": "100%",
                                                "maxWidth": "430px",
                                                "height": "220px",
                                                "objectFit": "cover",
                                                "borderRadius": "12px",
                                                "boxShadow": "0px 4px 10px rgba(0,0,0,0.15)"
                                            }
                                        )
                                    ]
                                )
                            ]
                        ),

                        # 2. Inputs del modelo
                        html.Div(
                            style=card_style,
                            children=[
                                html.H3(
                                    "2. Ingrese los datos relacionados con el estudiante",
                                    style={"color": COLOR_AZUL}
                                ),

                                html.Label("Acceso a computador en el hogar"),
                                dcc.Dropdown(
                                    id="brecha-computador",
                                    options=[
                                        {"label": "Sí", "value": 1},
                                        {"label": "No", "value": 0}
                                    ],
                                    value=0,
                                    clearable=False,
                                    style={"marginBottom": "15px"}
                                ),

                                html.Label("Acceso a internet en el hogar"),
                                dcc.Dropdown(
                                    id="brecha-internet",
                                    options=[
                                        {"label": "Sí", "value": 1},
                                        {"label": "No", "value": 0}
                                    ],
                                    value=0,
                                    clearable=False,
                                    style={"marginBottom": "15px"}
                                ),

                                html.Label("Estrato de la vivienda"),
                                dcc.Dropdown(
                                    id="brecha-estrato",
                                    options=[
                                        {"label": str(i), "value": i}
                                        for i in range(1, 7)
                                    ],
                                    value=1,
                                    clearable=False,
                                    style={"marginBottom": "15px"}
                                ),

                                html.Label("Número de personas en el hogar"),
                                dcc.Dropdown(
                                    id="brecha-personas",
                                    options=[
                                        {"label": str(i), "value": i}
                                        for i in range(1, 11)
                                    ],
                                    value=4,
                                    clearable=False,
                                    style={"marginBottom": "20px"}
                                ),

                                html.Button(
                                    "Ejecutar modelo",
                                    id="boton-brecha",
                                    n_clicks=0,
                                    style=button_style
                                )
                            ]
                        )
                    ]
                ),

                # ======================================================
                # Columna derecha: resultado + conclusión
                # ======================================================
                html.Div(
                    children=[

                        # 3. Resultado del modelo
                        html.Div(
                            id="tarjeta-probabilidad-brecha",
                            style=card_style,
                            children=[
                                html.H3(
                                    "3. Probabilidad de bajo desempeño",
                                    style={"color": COLOR_AZUL}
                                ),
                                html.P(
                                    "Ingrese los datos del estudiante y presione 'Ejecutar modelo' "
                                    "para obtener la predicción.",
                                    style={"color": "#555", "fontSize": "16px"}
                                )
                            ]
                        ),

                        # 4. Conclusión
                        html.Div(
                            id="conclusion-brecha",
                            style=card_style,
                            children=[
                                html.H3(
                                    "4. Conclusión",
                                    style={"color": COLOR_AZUL}
                                ),
                                html.P(
                                    "La conclusión se generará después de ejecutar el modelo.",
                                    style={"color": "#555", "fontSize": "16px"}
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)



tab_pregunta_3 = html.Div(
    style={
        "fontFamily": "Arial",
        "padding": "30px",
        "backgroundColor": COLOR_GRIS
    },
    children=[
        html.H2("Percentil esperado", style={"color": COLOR_AZUL}),
        html.Div(
            style=card_style,
            children=[
                html.H3("Descripción del problema", style={"color": COLOR_AZUL}),
                html.P(
                    "En esta sección se integrará el tercer modelo predictivo. "
                    "Aquí se presentarán los parámetros requeridos, la salida del modelo "
                    "y una conclusión orientada a la toma de decisiones."
                )
            ]
        )
    ]
)


# ======================================================
# Layout principal
# ======================================================

app.layout = html.Div(
    style={
        "backgroundColor": COLOR_GRIS,
        "minHeight": "100vh"
    },
    children=[
        header,

        dcc.Tabs(
            id="tabs",
            value="portada",
            children=[
                dcc.Tab(label="Portada", value="portada"),
                dcc.Tab(label="Desempeño institucional", value="pregunta_1"),
                dcc.Tab(label="Riesgo por brecha digital", value="pregunta_2"),
                dcc.Tab(label="Percentil esperado", value="pregunta_3")
            ],
            style={"fontFamily": "Arial"}
        ),

        html.Div(id="contenido-tabs")
    ]
)


# ======================================================
# Callback para cambiar de pestaña
# ======================================================

@app.callback(
    Output("contenido-tabs", "children"),
    Input("tabs", "value")
)
def renderizar_tabs(tab):
    if tab == "portada":
        return tab_portada
    elif tab == "pregunta_1":
        return tab_pregunta_1
    elif tab == "pregunta_2":
        return tab_pregunta_2
    elif tab == "pregunta_3":
        return tab_pregunta_3

    return tab_portada


# ======================================================
# Ejecución local
# ======================================================
# ======================================================
# Callback - Modelo brecha digital
# ======================================================

@app.callback(
    Output("tarjeta-probabilidad-brecha", "children"),
    Output("tarjeta-probabilidad-brecha", "style"),
    Output("conclusion-brecha", "children"),
    Input("boton-brecha", "n_clicks"),
    State("brecha-computador", "value"),
    State("brecha-internet", "value"),
    State("brecha-estrato", "value"),
    State("brecha-personas", "value")
)
def predecir_brecha_digital(n_clicks, computador, internet, estrato, personas):

    estilo_base = card_style.copy()

    if n_clicks == 0:
        return (
            [
                html.H3(
                    "3. Probabilidad de bajo desempeño",
                    style={"color": COLOR_AZUL}
                ),
                html.P(
                    "Ingrese los datos del estudiante y presione 'Ejecutar modelo' "
                    "para obtener la predicción.",
                    style={"color": "#555", "fontSize": "16px"}
                )
            ],
            estilo_base,
            [
                html.H3("4. Conclusión", style={"color": COLOR_AZUL}),
                html.P(
                    "La conclusión se generará después de ejecutar el modelo.",
                    style={"color": "#555", "fontSize": "16px"}
                )
            ]
        )

    entrada = np.array(
        [[computador, internet, estrato, personas]],
        dtype="float32"
    )

    probabilidad = float(modelo_brecha.predict(entrada, verbose=0)[0][0])
    porcentaje = probabilidad * 100

    if probabilidad < 0.40:
        riesgo = "bajo"
        color_fondo = "#DCFCE7"   # verde claro
        color_texto = "#166534"
        conclusion = (
            "El perfil ingresado presenta un riesgo bajo de bajo desempeño. "
            "Aunque no se identifica una alerta prioritaria, se recomienda mantener "
            "seguimiento general al estudiante."
        )

    elif probabilidad < 0.60:
        riesgo = "medio"
        color_fondo = "#FEF9C3"   # amarillo claro
        color_texto = "#854D0E"
        conclusion = (
            "El perfil ingresado presenta un riesgo medio de bajo desempeño. "
            "Esto sugiere la conveniencia de realizar seguimiento preventivo y revisar "
            "posibles necesidades de acompañamiento académico o tecnológico."
        )

    else:
        riesgo = "alto"
        color_fondo = "#FEE2E2"   # rojo claro
        color_texto = "#991B1B"
        conclusion = (
            "El perfil ingresado presenta un riesgo alto de bajo desempeño. "
            "Este resultado puede servir como alerta temprana para priorizar estrategias "
            "de acompañamiento académico y apoyo en acceso a recursos tecnológicos."
        )

    estilo_resultado = card_style.copy()
    estilo_resultado["backgroundColor"] = color_fondo
    estilo_resultado["border"] = f"2px solid {color_texto}"

    contenido_resultado = [
        html.H3(
            "3. Probabilidad de bajo desempeño",
            style={"color": color_texto}
        ),

        html.Div(
            style={
                "textAlign": "center",
                "padding": "20px"
            },
            children=[
                html.H1(
                    f"{porcentaje:.1f}%",
                    style={
                        "fontSize": "64px",
                        "color": color_texto,
                        "margin": "10px 0px"
                    }
                ),

                html.H3(
                    f"Riesgo {riesgo}",
                    style={
                        "color": color_texto,
                        "textTransform": "uppercase",
                        "marginBottom": "20px"
                    }
                ),

                html.P(
                    f"El estudiante tiene un porcentaje del {porcentaje:.1f}% "
                    "de tener un bajo desempeño según su acceso a computador e internet, "
                    "su estrato y las personas con las que vive.",
                    style={
                        "fontSize": "17px",
                        "lineHeight": "1.6",
                        "textAlign": "justify"
                    }
                ),

                html.P(
                    f"Esto se puede interpretar como un riesgo {riesgo}.",
                    style={
                        "fontSize": "18px",
                        "fontWeight": "bold",
                        "color": color_texto,
                        "marginTop": "18px"
                    }
                )
            ]
        )
    ]

    contenido_conclusion = [
        html.H3("4. Conclusión", style={"color": COLOR_AZUL}),
        html.P(
            conclusion,
            style={
                "fontSize": "16px",
                "lineHeight": "1.6",
                "textAlign": "justify"
            }
        )
    ]

    return contenido_resultado, estilo_resultado, contenido_conclusion

if __name__ == "__main__":
    app.run(debug=True)